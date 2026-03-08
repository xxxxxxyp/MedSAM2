import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_NPZ_DIR = "data/FL_dataset/val"
DEFAULT_MODEL_CFG = "sam2.1_hiera_tiny512_FL.yaml"
DEFAULT_CKPT_PATH = "exp_log/MedSAM2_FL_Finetune/checkpoints/checkpoint.pt"
MODEL_CFG_ALIASES = {
    "sam2.1_hiera_tiny512_FL.yaml": "sam2.1_hiera_tiny512_FLARE_RECIST.yaml",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz_dir", type=str, default=DEFAULT_NPZ_DIR)
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--model_cfg", type=str, default=DEFAULT_MODEL_CFG)
    parser.add_argument("--ckpt_path", type=str, default=DEFAULT_CKPT_PATH)
    return parser.parse_args()


def resolve_model_cfg_path(model_cfg):
    repo_root = Path(__file__).resolve().parent
    model_cfg_path = Path(model_cfg)
    candidates = []

    if model_cfg_path.is_absolute():
        candidates.append(model_cfg_path)
    else:
        candidates.extend(
            [
                repo_root / model_cfg_path,
                repo_root / "sam2" / "configs" / model_cfg_path.name,
            ]
        )
        alias = MODEL_CFG_ALIASES.get(model_cfg_path.name)
        if alias is not None:
            candidates.append(repo_root / "sam2" / "configs" / alias)

    for candidate in candidates:
        if candidate.exists():
            return f"//{candidate.resolve()}"

    searched_paths = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Unable to locate model config '{model_cfg}'. Searched: {searched_paths}"
    )


def load_bbox_cases(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list):
        raw_cases = {}
        for item in raw_data:
            if not isinstance(item, dict) or "case_id" not in item:
                raise ValueError("JSON list entries must contain a 'case_id' field.")
            raw_cases[item["case_id"]] = item
    elif isinstance(raw_data, dict):
        raw_cases = raw_data
    else:
        raise ValueError("Prompt JSON must be either a dict or a list of case entries.")

    bbox_cases = {}
    for case_id, case_value in raw_cases.items():
        if isinstance(case_value, dict):
            if "bboxes" in case_value:
                bboxes = case_value["bboxes"]
            elif "boxes" in case_value:
                bboxes = case_value["boxes"]
            else:
                raise ValueError(f"Case '{case_id}' is missing a 'bboxes' field.")
        elif isinstance(case_value, list):
            bboxes = case_value
        else:
            raise ValueError(f"Unsupported JSON value for case '{case_id}'.")
        bbox_cases[str(case_id)] = bboxes

    return bbox_cases


def resolve_case_npz_path(npz_dir, case_id):
    case_path = Path(npz_dir) / case_id
    if case_path.suffix == ".npz" and case_path.exists():
        return case_path

    if case_path.exists():
        return case_path

    npz_path = case_path.with_suffix(".npz")
    if npz_path.exists():
        return npz_path

    raise FileNotFoundError(f"Unable to locate NPZ for case '{case_id}' in '{npz_dir}'.")


def resize_grayscale_to_rgb_and_resize(array, image_size):
    depth = array.shape[0]
    resized_array = np.zeros((depth, 3, image_size, image_size), dtype=np.float32)

    for idx in range(depth):
        img_pil = Image.fromarray(array[idx].astype(np.uint8))
        img_rgb = img_pil.convert("RGB")
        img_resized = img_rgb.resize((image_size, image_size))
        resized_array[idx] = np.asarray(img_resized, dtype=np.float32).transpose(2, 0, 1)

    return resized_array


def prepare_video_frames(img_3d, predictor, image_size=512):
    import torch

    if img_3d.shape[1] != image_size or img_3d.shape[2] != image_size:
        img_resized = resize_grayscale_to_rgb_and_resize(img_3d, image_size)
    else:
        img_resized = np.repeat(img_3d[:, None], 3, axis=1).astype(np.float32)

    img_resized = torch.from_numpy(img_resized / 255.0).to(predictor.device)
    img_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32, device=predictor.device)[:, None, None]
    img_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32, device=predictor.device)[:, None, None]
    img_resized -= img_mean
    img_resized /= img_std
    return img_resized


def convert_bbox_prompt(bbox, volume_shape):
    if len(bbox) != 6:
        raise ValueError(f"Each bbox must contain 6 values, got {bbox}")

    depth, height, width = volume_shape
    z_min, z_max, y_min, y_max, x_min, x_max = [int(round(coord)) for coord in bbox]
    z_min, z_max = sorted((z_min, z_max))
    y_min, y_max = sorted((y_min, y_max))
    x_min, x_max = sorted((x_min, x_max))

    z_min = np.clip(z_min, 0, depth - 1)
    z_max = np.clip(z_max, 0, depth - 1)
    y_min = np.clip(y_min, 0, height - 1)
    y_max = np.clip(y_max, 0, height - 1)
    x_min = np.clip(x_min, 0, width - 1)
    x_max = np.clip(x_max, 0, width - 1)

    z_mid = int((z_min + z_max) // 2)
    box = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)
    return z_mid, box


def merge_mask_logits(mask_logits):
    merged_mask = None

    for single_mask_logits in mask_logits:
        if hasattr(single_mask_logits, "detach"):
            mask = single_mask_logits.detach().cpu().numpy()
        else:
            mask = np.asarray(single_mask_logits)
        mask = mask > 0.0
        while mask.ndim > 2:
            mask = mask[0]
        merged_mask = mask if merged_mask is None else np.logical_or(merged_mask, mask)

    if merged_mask is None:
        raise ValueError("No masks were produced during propagation.")

    return merged_mask


def dice_score(pred_mask, gt_mask):
    pred_mask = np.asarray(pred_mask).astype(bool)
    gt_mask = np.asarray(gt_mask).astype(bool)
    denominator = pred_mask.sum(dtype=np.float64) + gt_mask.sum(dtype=np.float64)
    if denominator == 0:
        return 1.0
    intersection = np.logical_and(pred_mask, gt_mask).sum(dtype=np.float64)
    return float((2.0 * intersection) / denominator)


def run_case_inference(img_3d, bboxes, predictor, prepared_frames=None):
    if np.max(img_3d) >= 256:
        raise ValueError(
            f"Input images should be in range [0, 255], but got max value {np.max(img_3d)}."
        )

    if prepared_frames is None:
        prepared_frames = prepare_video_frames(img_3d, predictor)

    prediction = np.zeros(img_3d.shape, dtype=bool)
    inference_state = predictor.init_state(
        prepared_frames,
        video_height=img_3d.shape[1],
        video_width=img_3d.shape[2],
    )

    try:
        for obj_id, bbox in enumerate(bboxes, start=1):
            z_mid, box = convert_bbox_prompt(bbox, img_3d.shape)
            predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=z_mid,
                obj_id=obj_id,
                box=box,
            )

        if bboxes:
            for frame_idx, _, mask_logits in predictor.propagate_in_video(inference_state):
                prediction[frame_idx] = np.logical_or(
                    prediction[frame_idx], merge_mask_logits(mask_logits)
                )
    finally:
        predictor.reset_state(inference_state)

    return prediction


def evaluate_case(npz_path, bboxes, predictor, prepared_frames=None):
    npz_data = np.load(npz_path, allow_pickle=True)
    if "imgs" not in npz_data or "gts" not in npz_data:
        raise KeyError(f"NPZ file '{npz_path}' must contain both 'imgs' and 'gts'.")

    img_3d = npz_data["imgs"]
    gt_mask = npz_data["gts"] > 0
    pred_mask = run_case_inference(
        img_3d,
        bboxes,
        predictor,
        prepared_frames=prepared_frames,
    )
    return dice_score(pred_mask, gt_mask)


def build_predictor(model_cfg, ckpt_path):
    import torch

    from sam2.build_sam import build_sam2_video_predictor_npz

    torch.set_float32_matmul_precision("high")
    return build_sam2_video_predictor_npz(
        resolve_model_cfg_path(model_cfg),
        str(Path(ckpt_path).expanduser()),
    )


def main():
    args = parse_args()
    bbox_cases = load_bbox_cases(args.json_path)
    predictor = build_predictor(args.model_cfg, args.ckpt_path)

    case_scores = []
    for case_id, bboxes in bbox_cases.items():
        npz_path = resolve_case_npz_path(args.npz_dir, case_id)
        dsc = evaluate_case(npz_path, bboxes, predictor)
        case_scores.append(dsc)
        print(f"{Path(npz_path).stem}: DSC={dsc:.4f}")

    if not case_scores:
        raise ValueError("No cases were found in the prompt JSON.")

    avg_dsc = float(np.mean(case_scores))
    print(f"Average DSC: {avg_dsc:.4f}")


if __name__ == "__main__":
    main()
