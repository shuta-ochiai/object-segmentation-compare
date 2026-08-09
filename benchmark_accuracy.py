"""Evaluate YOLO26-seg, SAM3.1, and RF-DETR-Seg mask predictions against COCO ground truth."""

import json
from functools import partial
from pathlib import Path

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models import CHECKPOINT_DIR, run_rfdetr, run_sam31, run_yolo

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
IMAGE_DIR = DATA_DIR / "subset_accuracy"
ANNOTATIONS_PATH = DATA_DIR / "subset_accuracy_annotations.json"

# Standard COCO mAP evaluation protocol passes (almost) unfiltered detections
# into COCOeval and lets it sweep the full precision-recall curve itself;
# pre-filtering with a "serving" confidence threshold (e.g. 0.25) truncates the
# low-confidence/high-recall end of that curve and depresses AP/AR versus
# officially published numbers. Use a near-zero threshold here for closer
# apples-to-apples comparison with published benchmarks (benchmark_speed.py
# intentionally keeps conf=0.25, a realistic serving threshold, since that's
# what affects real-world latency).
EVAL_CONF = 0.001

# (display name, slug used in results/accuracy_<slug>_predictions.json, run function)
MODELS = [
    ("YOLO26-seg (n)", "yolo26n", partial(run_yolo, weights=CHECKPOINT_DIR / "yolo26n-seg.pt", conf=EVAL_CONF)),
    ("YOLO26-seg (m)", "yolo26m", partial(run_yolo, weights=CHECKPOINT_DIR / "yolo26m-seg.pt", conf=EVAL_CONF)),
    ("YOLO26-seg (x)", "yolo26x", partial(run_yolo, weights=CHECKPOINT_DIR / "yolo26x-seg.pt", conf=EVAL_CONF)),
    ("SAM3.1", "sam31", run_sam31),
    ("RF-DETR-Seg (Nano)", "rfdetrnano", partial(run_rfdetr, model_size="Nano", conf=EVAL_CONF)),
    ("RF-DETR-Seg (Medium)", "rfdetrmedium", partial(run_rfdetr, model_size="Medium", conf=EVAL_CONF)),
    ("RF-DETR-Seg (Large)", "rfdetrlarge", partial(run_rfdetr, model_size="Large", conf=EVAL_CONF)),
]


def evaluate(gt: COCO, predictions: list[dict]) -> dict:
    if not predictions:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0, "AR": 0.0, "n_predictions": 0}
    dt = gt.loadRes(predictions)
    coco_eval = COCOeval(gt, dt, iouType="segm")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return {
        "AP": coco_eval.stats[0],
        "AP50": coco_eval.stats[1],
        "AP75": coco_eval.stats[2],
        "AR": coco_eval.stats[8],
        "n_predictions": len(predictions),
    }


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    gt = COCO(str(ANNOTATIONS_PATH))

    rows = []
    for name, slug, run_fn in MODELS:
        print(f"running {name}...")
        preds = run_fn(IMAGE_DIR, ANNOTATIONS_PATH)
        (RESULTS_DIR / f"accuracy_{slug}_predictions.json").write_text(json.dumps(preds))
        rows.append({"model": name, **evaluate(gt, preds)})

    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "accuracy_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
