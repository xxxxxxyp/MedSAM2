import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluate_pipeline import (
    convert_bbox_prompt,
    evaluate_case,
    resolve_model_cfg_path,
    run_case_inference,
)


class FakePredictor:
    def __init__(self, propagated_masks):
        self.device = "cpu"
        self.propagated_masks = propagated_masks
        self.add_calls = []
        self.reset_called = False

    def init_state(self, images, video_height, video_width):
        self.init_args = (images, video_height, video_width)
        return {"video_height": video_height, "video_width": video_width}

    def add_new_points_or_box(self, inference_state, frame_idx, obj_id, box):
        self.add_calls.append((frame_idx, obj_id, np.asarray(box)))
        return frame_idx, [obj_id], self.propagated_masks[frame_idx]

    def propagate_in_video(self, inference_state):
        for frame_idx, mask_logits in sorted(self.propagated_masks.items()):
            yield frame_idx, list(range(1, len(mask_logits) + 1)), mask_logits

    def reset_state(self, inference_state):
        self.reset_called = True


class EvaluatePipelineTests(unittest.TestCase):
    def test_convert_bbox_prompt_uses_expected_coordinate_order(self):
        z_mid, box = convert_bbox_prompt([5, 1, 7, 2, 8, 3], (6, 8, 9))
        self.assertEqual(z_mid, 3)
        np.testing.assert_array_equal(box, np.array([3.0, 2.0, 8.0, 7.0], dtype=np.float32))

    def test_run_case_inference_merges_all_object_masks(self):
        frame_masks = {
            1: np.array(
                [
                    [[[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]],
                    [[[0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 0]]],
                ],
                dtype=np.float32,
            )
        }
        predictor = FakePredictor(frame_masks)
        prediction = run_case_inference(
            img_3d=np.zeros((3, 4, 4), dtype=np.uint8),
            bboxes=[[0, 2, 0, 1, 0, 1], [1, 1, 1, 2, 2, 3]],
            predictor=predictor,
            prepared_frames=np.zeros((3, 3, 4, 4), dtype=np.float32),
        )

        expected = np.zeros((3, 4, 4), dtype=bool)
        expected[1, :2, :2] = True
        expected[1, 1:3, 2:4] = True
        np.testing.assert_array_equal(prediction, expected)
        self.assertTrue(predictor.reset_called)
        self.assertEqual([call[:2] for call in predictor.add_calls], [(1, 1), (1, 2)])

    def test_evaluate_case_computes_binary_dice_from_npz(self):
        frame_masks = {
            1: np.array([[[[1, 0], [0, 1]]]], dtype=np.float32),
        }
        predictor = FakePredictor(frame_masks)

        imgs = np.zeros((3, 2, 2), dtype=np.uint8)
        gts = np.zeros((3, 2, 2), dtype=np.uint8)
        gts[1, 0, 0] = 1
        gts[1, 1, 1] = 2

        with tempfile.TemporaryDirectory() as tmpdir:
            npz_path = Path(tmpdir) / "case_a.npz"
            np.savez_compressed(npz_path, imgs=imgs, gts=gts)
            dsc = evaluate_case(
                npz_path,
                [[1, 1, 0, 1, 0, 1]],
                predictor,
                prepared_frames=np.zeros((3, 3, 2, 2), dtype=np.float32),
            )

        self.assertAlmostEqual(dsc, 1.0)

    def test_default_model_cfg_alias_resolves_to_existing_file(self):
        resolved = resolve_model_cfg_path("sam2.1_hiera_tiny512_FL.yaml")
        self.assertTrue(resolved.endswith("sam2/configs/sam2.1_hiera_tiny512_FLARE_RECIST.yaml"))


if __name__ == "__main__":
    unittest.main()
