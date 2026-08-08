"""Generate comparison charts and qualitative mask overlays for the report."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pycocotools.mask as mask_utils
from PIL import Image

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"
IMAGE_DIR = DATA_DIR / "subset_accuracy"

COLORS = {"YOLO26-seg": "#4C72B0", "SAM3.1": "#DD8452"}
N_QUALITATIVE_EXAMPLES = 4


def plot_speed(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(df["model"], df["mean_latency_s"], color=[COLORS[m] for m in df["model"]])
    axes[0].set_ylabel("mean latency (s, log scale)")
    axes[0].set_yscale("log")
    axes[0].set_title("Inference latency")

    axes[1].bar(df["model"], df["peak_vram_gb"], color=[COLORS[m] for m in df["model"]])
    axes[1].set_ylabel("peak VRAM (GB)")
    axes[1].set_title("Peak GPU memory")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "speed_comparison.png", dpi=150)
    plt.close(fig)


def plot_accuracy(df: pd.DataFrame) -> None:
    metrics = ["AP", "AP50", "AP75", "AR"]
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(7, 4))
    for i, model in enumerate(df["model"]):
        values = df.loc[df["model"] == model, metrics].values.flatten()
        ax.bar(x + (i - 0.5) * width, values, width, label=model, color=COLORS[model])

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
    yolo_path = RESULTS_DIR / "accuracy_yolo26_predictions.json"
    sam_path = RESULTS_DIR / "accuracy_sam31_predictions.json"
    if not (yolo_path.exists() and sam_path.exists()):
        print("skip qualitative overlays: run benchmark_accuracy.py first")
        return

    yolo_preds = json.loads(yolo_path.read_text())
    sam_preds = json.loads(sam_path.read_text())

    image_ids = sorted({p["image_id"] for p in yolo_preds} & {p["image_id"] for p in sam_preds})
    image_ids = image_ids[:N_QUALITATIVE_EXAMPLES]
    if not image_ids:
        print("skip qualitative overlays: no overlapping image_ids between predictions")
        return

    fig, axes = plt.subplots(len(image_ids), 2, figsize=(8, 4 * len(image_ids)))
    if len(image_ids) == 1:
        axes = axes[np.newaxis, :]

    for row, image_id in enumerate(image_ids):
        img_path = IMAGE_DIR / f"{image_id:012d}.jpg"
        image = np.array(Image.open(img_path).convert("RGB"))

        yolo_for_img = [p for p in yolo_preds if p["image_id"] == image_id]
        sam_for_img = [p for p in sam_preds if p["image_id"] == image_id]

        axes[row, 0].imshow(_overlay_masks(image, yolo_for_img))
        axes[row, 0].set_title("YOLO26-seg" if row == 0 else "")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(_overlay_masks(image, sam_for_img))
        axes[row, 1].set_title("SAM3.1" if row == 0 else "")
        axes[row, 1].axis("off")

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "qualitative_examples.png", dpi=150)
    plt.close(fig)
    print(f"wrote {RESULTS_DIR / 'qualitative_examples.png'}")


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
