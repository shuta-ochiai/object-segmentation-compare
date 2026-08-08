"""Common inference wrappers for YOLO26-seg and SAM3.1.

Both wrappers normalize predictions to a shared format so they can be fed into
the same COCO-style evaluation and speed-benchmark code:

    [{"image_id": int, "category_id": int, "segmentation": RLE, "score": float}, ...]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pycocotools.mask as mask_utils

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"

# ultralytics detects SAM3 by checking for "sam3" in the checkpoint filename
# stem (models/sam/model.py), so the actual facebook/sam3.1 filename works as-is.
SAM3_CHECKPOINT = CHECKPOINT_DIR / "sam3.1_multiplex.pt"


def _load_coco_categories(annotations_path: Path) -> dict[int, str]:
    coco = json.loads(annotations_path.read_text())
    return {c["id"]: c["name"] for c in coco["categories"]}


def _mask_to_rle(binary_mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
    rle["counts"] = rle["counts"].decode("utf-8")
    return rle


YOLO_CHECKPOINT = CHECKPOINT_DIR / "yolo26n-seg.pt"


def run_yolo(
    image_dir: Path | str,
    annotations_path: Path | str,
    weights: Path | str = YOLO_CHECKPOINT,
    conf: float = 0.25,
) -> list[dict]:
    """Run YOLO26 instance segmentation over every image in image_dir."""
    from ultralytics import YOLO

    image_dir = Path(image_dir)
    categories = _load_coco_categories(Path(annotations_path))
    name_to_id = {name: cid for cid, name in categories.items()}

    model = YOLO(str(weights))
    image_paths = sorted(image_dir.glob("*.jpg"))

    predictions: list[dict] = []
    for img_path in image_paths:
        image_id = int(img_path.stem)
        # retina_masks=True upsamples masks to the original image resolution;
        # without it masks.data stays at the stride-padded inference resolution,
        # which silently breaks RLE/IoU comparisons against COCO ground truth.
        results = model.predict(source=str(img_path), conf=conf, verbose=False, retina_masks=True)
        r = results[0]
        if r.masks is None:
            continue
        for mask, box, cls_idx, score in zip(
            r.masks.data.cpu().numpy(),
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
        ):
            cls_name = model.names[int(cls_idx)]
            category_id = name_to_id.get(cls_name)
            if category_id is None:
                continue
            binary_mask = (mask > 0.5).astype(np.uint8)
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": _mask_to_rle(binary_mask),
                    "score": float(score),
                }
            )
    return predictions


def run_sam31(
    image_dir: Path | str,
    annotations_path: Path | str,
    conf: float = 0.25,
) -> list[dict]:
    """Run SAM3.1 concept segmentation, prompting with the COCO category names
    present in each image's ground truth (closed-set framing for a fair
    accuracy comparison against YOLO26)."""
    from ultralytics.models.sam import SAM3SemanticPredictor

    if not SAM3_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{SAM3_CHECKPOINT} not found. Complete the HF access request in README.md, "
            "then download the checkpoint into checkpoints/."
        )

    image_dir = Path(image_dir)
    coco = json.loads(Path(annotations_path).read_text())
    categories = {c["id"]: c["name"] for c in coco["categories"]}
    cats_by_image: dict[int, set[int]] = {}
    for ann in coco["annotations"]:
        cats_by_image.setdefault(ann["image_id"], set()).add(ann["category_id"])

    # SAM3's text-prompted concept segmentation is only exposed through
    # SAM3SemanticPredictor (plain `SAM(...).predict(text=...)` rejects `text`
    # as an unknown cfg override), following the ultralytics docs pattern:
    # set_image(...) then predictor(text=[...]).
    predictor = SAM3SemanticPredictor(
        overrides={
            "conf": conf,
            "task": "segment",
            "mode": "predict",
            "model": str(SAM3_CHECKPOINT),
            "save": False,
            "verbose": False,
        }
    )

    predictions: list[dict] = []
    for img_path in sorted(image_dir.glob("*.jpg")):
        image_id = int(img_path.stem)
        cat_ids = sorted(cats_by_image.get(image_id, set()))
        if not cat_ids:
            continue
        text_prompts = [categories[c] for c in cat_ids]
        name_to_id = {categories[c]: c for c in cat_ids}

        predictor.set_image(str(img_path))
        results = predictor(text=text_prompts)
        r = results[0]
        if r.masks is None:
            continue
        for mask, cls_idx, score in zip(
            r.masks.data.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
        ):
            cls_name = text_prompts[int(cls_idx)] if int(cls_idx) < len(text_prompts) else None
            category_id = name_to_id.get(cls_name)
            if category_id is None:
                continue
            binary_mask = (mask > 0.5).astype(np.uint8)
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": _mask_to_rle(binary_mask),
                    "score": float(score),
                }
            )
    return predictions
