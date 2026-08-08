"""Download a stratified subset of COCO val2017 for the SAM3.1 vs YOLO26 comparison."""

import json
import random
import shutil
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).parent / "data"
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMAGES_BASE_URL = "http://images.cocodataset.org/val2017"

N_ACCURACY = 100
N_SPEED = 20
SEED = 0


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"skip (exists): {dest}")
        return
    print(f"downloading {url} -> {dest}")
    urlretrieve(url, dest)


def ensure_annotations() -> Path:
    zip_path = DATA_DIR / "annotations_trainval2017.zip"
    ann_path = DATA_DIR / "annotations" / "instances_val2017.json"
    if not ann_path.exists():
        download(ANNOTATIONS_URL, zip_path)
        print("extracting annotations...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extract("annotations/instances_val2017.json", DATA_DIR)
        zip_path.unlink()
    return ann_path


def stratified_sample(coco_json: dict, n: int, seed: int) -> list[dict]:
    """Pick images so that COCO categories are represented as evenly as possible."""
    rng = random.Random(seed)
    img_by_id = {img["id"]: img for img in coco_json["images"]}
    cats_by_img: dict[int, set[int]] = {}
    for ann in coco_json["annotations"]:
        cats_by_img.setdefault(ann["image_id"], set()).add(ann["category_id"])

    cat_ids = sorted({c["id"] for c in coco_json["categories"]})
    rng.shuffle(cat_ids)

    chosen: list[int] = []
    chosen_set: set[int] = set()
    covered_cats: set[int] = set()

    imgs_by_cat: dict[int, list[int]] = {c: [] for c in cat_ids}
    for img_id, cats in cats_by_img.items():
        for c in cats:
            imgs_by_cat[c].append(img_id)
    for c in imgs_by_cat:
        rng.shuffle(imgs_by_cat[c])

    # Round-robin over categories so the subset spans as many classes as possible.
    while len(chosen) < n and len(covered_cats) < len(cat_ids):
        progressed = False
        for c in cat_ids:
            if len(chosen) >= n:
                break
            for img_id in imgs_by_cat[c]:
                if img_id not in chosen_set:
                    chosen.append(img_id)
                    chosen_set.add(img_id)
                    covered_cats.add(c)
                    progressed = True
                    break
        if not progressed:
            break

    # Fill remainder randomly if not enough images yet.
    all_ids = list(img_by_id.keys())
    rng.shuffle(all_ids)
    for img_id in all_ids:
        if len(chosen) >= n:
            break
        if img_id not in chosen_set:
            chosen.append(img_id)
            chosen_set.add(img_id)

    return [img_by_id[i] for i in chosen[:n]]


def download_images(images: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images, 1):
        dest = out_dir / img["file_name"]
        if dest.exists():
            continue
        url = f"{IMAGES_BASE_URL}/{img['file_name']}"
        print(f"[{i}/{len(images)}] {img['file_name']}")
        urlretrieve(url, dest)


def write_subset_annotations(coco_json: dict, images: list[dict], dest: Path) -> None:
    img_ids = {img["id"] for img in images}
    subset = {
        "images": images,
        "annotations": [a for a in coco_json["annotations"] if a["image_id"] in img_ids],
        "categories": coco_json["categories"],
    }
    dest.write_text(json.dumps(subset))


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    ann_path = ensure_annotations()
    coco_json = json.loads(ann_path.read_text())

    accuracy_images = stratified_sample(coco_json, N_ACCURACY, SEED)
    speed_images = accuracy_images[:N_SPEED]

    download_images(accuracy_images, DATA_DIR / "subset_accuracy")
    for img in speed_images:
        src = DATA_DIR / "subset_accuracy" / img["file_name"]
        dst_dir = DATA_DIR / "subset_speed"
        dst_dir.mkdir(exist_ok=True)
        shutil.copy(src, dst_dir / img["file_name"])

    write_subset_annotations(coco_json, accuracy_images, DATA_DIR / "subset_accuracy_annotations.json")

    print(f"accuracy subset: {len(accuracy_images)} images -> {DATA_DIR / 'subset_accuracy'}")
    print(f"speed subset: {len(speed_images)} images -> {DATA_DIR / 'subset_speed'}")
    print(f"annotations: {DATA_DIR / 'subset_accuracy_annotations.json'}")


if __name__ == "__main__":
    main()
