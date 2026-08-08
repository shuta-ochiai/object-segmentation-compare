"""Measure per-image latency, throughput, and peak VRAM for YOLO26-seg and SAM3.1."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from models import SAM3_CHECKPOINT, YOLO_CHECKPOINT

DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
IMAGE_DIR = DATA_DIR / "subset_speed"
ANNOTATIONS_PATH = DATA_DIR / "subset_accuracy_annotations.json"

N_WARMUP = 2


def _timed_predict(predict_fn, image_paths, n_warmup: int = N_WARMUP) -> dict:
    for img_path in image_paths[:n_warmup]:
        predict_fn(img_path)

    torch.cuda.reset_peak_memory_stats()
    latencies = []
    for img_path in image_paths:
        torch.cuda.synchronize()
        start = time.perf_counter()
        predict_fn(img_path)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - start)

    latencies = np.array(latencies)
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
    return {
        "mean_latency_s": float(latencies.mean()),
        "median_latency_s": float(np.median(latencies)),
        "p95_latency_s": float(np.percentile(latencies, 95)),
        "throughput_img_s": float(1.0 / latencies.mean()),
        "peak_vram_gb": peak_mem_gb,
        "n_images": len(image_paths),
    }


def benchmark_yolo(image_paths: list[Path]) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(YOLO_CHECKPOINT))

    def predict_fn(img_path: Path):
        model.predict(source=str(img_path), verbose=False, retina_masks=True)

    return _timed_predict(predict_fn, image_paths)


def benchmark_sam31(image_paths: list[Path]) -> dict:
    from ultralytics.models.sam import SAM3SemanticPredictor

    if not SAM3_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"{SAM3_CHECKPOINT} not found. See README.md for the HF access + download steps."
        )

    predictor = SAM3SemanticPredictor(
        overrides={
            "conf": 0.25,
            "task": "segment",
            "mode": "predict",
            "model": str(SAM3_CHECKPOINT),
            "save": False,
            "verbose": False,
        }
    )
    # Prompting with all 80 COCO category names at once overflows 8GB VRAM
    # (SAM3's grounding attention scales with the number of text prompts), and
    # isn't representative of how SAM3.1 is actually used. Use the same
    # per-image ground-truth categories as benchmark_accuracy.py instead.
    coco = json.loads(ANNOTATIONS_PATH.read_text())
    categories_by_id = {c["id"]: c["name"] for c in coco["categories"]}
    cats_by_image: dict[int, list[str]] = {}
    for ann in coco["annotations"]:
        cats_by_image.setdefault(ann["image_id"], set()).add(ann["category_id"])
    cats_by_image = {
        image_id: sorted(categories_by_id[c] for c in cat_ids) for image_id, cat_ids in cats_by_image.items()
    }

    def predict_fn(img_path: Path):
        image_id = int(img_path.stem)
        text_prompts = cats_by_image.get(image_id) or ["object"]
        predictor.set_image(str(img_path))
        predictor(text=text_prompts)
        torch.cuda.empty_cache()

    return _timed_predict(predict_fn, image_paths)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    if not image_paths:
        raise SystemExit(f"no images found in {IMAGE_DIR} — run download_data.py first")

    print(f"benchmarking on {len(image_paths)} images")

    print("benchmarking YOLO26-seg...")
    yolo_stats = benchmark_yolo(image_paths)

    print("benchmarking SAM3.1...")
    sam_stats = benchmark_sam31(image_paths)

    summary = pd.DataFrame([{"model": "YOLO26-seg", **yolo_stats}, {"model": "SAM3.1", **sam_stats}])
    summary.to_csv(RESULTS_DIR / "speed_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
