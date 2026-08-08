"""Evaluate YOLO26-seg and SAM3.1 mask predictions against COCO ground truth."""

import json
from pathlib import Path

import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models import run_sam31, run_yolo

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
IMAGE_DIR = DATA_DIR / "subset_accuracy"
ANNOTATIONS_PATH = DATA_DIR / "subset_accuracy_annotations.json"


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

    print("running YOLO26-seg...")
    yolo_preds = run_yolo(IMAGE_DIR, ANNOTATIONS_PATH)
    (RESULTS_DIR / "accuracy_yolo26_predictions.json").write_text(json.dumps(yolo_preds))
    yolo_metrics = evaluate(gt, yolo_preds)

    print("running SAM3.1...")
    sam_preds = run_sam31(IMAGE_DIR, ANNOTATIONS_PATH)
    (RESULTS_DIR / "accuracy_sam31_predictions.json").write_text(json.dumps(sam_preds))
    sam_metrics = evaluate(gt, sam_preds)

    summary = pd.DataFrame(
        [{"model": "YOLO26-seg", **yolo_metrics}, {"model": "SAM3.1", **sam_metrics}]
    )
    summary.to_csv(RESULTS_DIR / "accuracy_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
