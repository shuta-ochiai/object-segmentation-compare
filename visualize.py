"""Generate comparison charts and qualitative mask overlays for the report."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"
IMAGE_DIR = DATA_DIR / "subset_accuracy"

COLORS = {
    "YOLO26-seg (n)": "#4C72B0",
    "YOLO26-seg (m)": "#7A9FC9",
    "YOLO26-seg (x)": "#A9C2DF",
    "SAM3.1": "#DD8452",
    "RF-DETR-Seg (Nano)": "#55A868",
    "RF-DETR-Seg (Medium)": "#89BF9B",
    "RF-DETR-Seg (Large)": "#B8DBC3",
}
# (display name, slug matching results/accuracy_<slug>_predictions.json)
# One representative size per family (+SAM3.1) to keep the qualitative PDF readable;
# see accuracy_comparison.png / speed_comparison.png for the full size sweep.
QUALITATIVE_MODELS = [
    ("YOLO26-seg (m)", "yolo26m"),
    ("SAM3.1", "sam31"),
    ("RF-DETR-Seg (Medium)", "rfdetrmedium"),
]
ROWS_PER_PAGE = 5
# accuracy_*_predictions.json is generated at EVAL_CONF=0.001 (benchmark_accuracy.py)
# for standard COCO mAP scoring, which is far too noisy to look at directly — filter
# down to a realistic serving-style confidence for the overlay images.
QUALITATIVE_MIN_SCORE = 0.25


def plot_speed(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(4 + 1.1 * len(df), 5))

    axes[0].bar(df["model"], df["mean_latency_s"], color=[COLORS[m] for m in df["model"]])
    axes[0].set_ylabel("mean latency (s, log scale)")
    axes[0].set_yscale("log")
    axes[0].set_title("Inference latency")
    axes[0].tick_params(axis="x", rotation=45)
    for label in axes[0].get_xticklabels():
        label.set_horizontalalignment("right")

    axes[1].bar(df["model"], df["peak_vram_gb"], color=[COLORS[m] for m in df["model"]])
    axes[1].set_ylabel("peak VRAM (GB)")
    axes[1].set_title("Peak GPU memory")
    axes[1].tick_params(axis="x", rotation=45)
    for label in axes[1].get_xticklabels():
        label.set_horizontalalignment("right")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "speed_comparison.png", dpi=150)
    plt.close(fig)


def plot_accuracy(df: pd.DataFrame) -> None:
    metrics = ["AP", "AP50", "AP75", "AR"]
    x = np.arange(len(metrics))
    n_models = len(df["model"])
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, model in enumerate(df["model"]):
        values = df.loc[df["model"] == model, metrics].values.flatten()
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, values, width, label=model, color=COLORS[model])

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score")
    ax.set_title("Mask accuracy (COCOeval, segm)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)


def _overlay_masks(image: np.ndarray, predictions: list[dict]) -> np.ndarray:
    overlay = image.copy()
    rng = np.random.default_rng(0)
    for pred in predictions:
        mask = mask_utils.decode(pred["segmentation"]).astype(bool)
        color = rng.integers(64, 256, size=3)
        overlay[mask] = (0.5 * overlay[mask] + 0.5 * color).astype(np.uint8)
    return overlay


def plot_qualitative_examples() -> None:
    preds_by_image: dict[str, dict[int, list[dict]]] = {}
    for name, slug in QUALITATIVE_MODELS:
        path = RESULTS_DIR / f"accuracy_{slug}_predictions.json"
        if not path.exists():
            print(f"skip qualitative overlays: {path} not found (run benchmark_accuracy.py first)")
            return
        by_image: dict[int, list[dict]] = {}
        for p in json.loads(path.read_text()):
            if p["score"] < QUALITATIVE_MIN_SCORE:
                continue
            by_image.setdefault(p["image_id"], []).append(p)
        preds_by_image[name] = by_image

    image_ids = sorted(set.intersection(*(set(d) for d in preds_by_image.values())))
    if not image_ids:
        print("skip qualitative overlays: no overlapping image_ids between predictions")
        return

    n_cols = 1 + len(QUALITATIVE_MODELS)  # input image + one column per model
    out_path = RESULTS_DIR / "qualitative_examples.pdf"
    with PdfPages(out_path) as pdf:
        for page_start in range(0, len(image_ids), ROWS_PER_PAGE):
            page_ids = image_ids[page_start : page_start + ROWS_PER_PAGE]

            fig, axes = plt.subplots(len(page_ids), n_cols, figsize=(4 * n_cols, 4 * len(page_ids)))
            if len(page_ids) == 1:
                axes = axes[np.newaxis, :]

            for row, image_id in enumerate(page_ids):
                img_path = IMAGE_DIR / f"{image_id:012d}.jpg"
                image = np.array(Image.open(img_path).convert("RGB"))

                axes[row, 0].imshow(image)
                axes[row, 0].set_title("input" if row == 0 else "")
                axes[row, 0].axis("off")

                for col, (name, _slug) in enumerate(QUALITATIVE_MODELS, start=1):
                    axes[row, col].imshow(_overlay_masks(image, preds_by_image[name][image_id]))
                    axes[row, col].set_title(name if row == 0 else "")
                    axes[row, col].axis("off")

            fig.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

    print(f"wrote {out_path} ({len(image_ids)} images, {-(-len(image_ids) // ROWS_PER_PAGE)} pages)")


def main() -> None:
    speed_path = RESULTS_DIR / "speed_summary.csv"
    accuracy_path = RESULTS_DIR / "accuracy_summary.csv"

    if speed_path.exists():
        plot_speed(pd.read_csv(speed_path))
        print(f"wrote {RESULTS_DIR / 'speed_comparison.png'}")
    else:
        print(f"skip speed plot: {speed_path} not found (run benchmark_speed.py first)")

    if accuracy_path.exists():
        plot_accuracy(pd.read_csv(accuracy_path))
        print(f"wrote {RESULTS_DIR / 'accuracy_comparison.png'}")
    else:
        print(f"skip accuracy plot: {accuracy_path} not found (run benchmark_accuracy.py first)")

    plot_qualitative_examples()


if __name__ == "__main__":
    main()
