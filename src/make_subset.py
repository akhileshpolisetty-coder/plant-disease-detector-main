# src/make_subset.py
import os
import shutil
import random
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_BASE = BASE_DIR / "data" / "raw" / "New Plant Diseases Dataset(Augmented)" / "New Plant Diseases Dataset(Augmented)"
SOURCE_TRAIN = RAW_BASE / "train"
SOURCE_VALID = RAW_BASE / "valid"

DEST_TRAIN = BASE_DIR / "data" / "processed" / "train"
DEST_VALID = BASE_DIR / "data" / "processed" / "valid"

def make_subset(source_dir, dest_dir, images_per_class, selected_classes):
    os.makedirs(dest_dir, exist_ok=True)
    for cls in selected_classes:
        src_cls_dir = os.path.join(source_dir, cls)
        dst_cls_dir = os.path.join(dest_dir, cls)
        os.makedirs(dst_cls_dir, exist_ok=True)

        images = os.listdir(src_cls_dir)
        random.shuffle(images)
        selected_images = images[:images_per_class]

        for img in selected_images:
            shutil.copy(os.path.join(src_cls_dir, img), os.path.join(dst_cls_dir, img))

        print(f"Copied {len(selected_images)} images for class: {cls}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a subset of the PlantVillage dataset.")
    parser.add_argument("--num-classes",  type=int, default=38,  help="Number of classes to sample (default: 38)")
    parser.add_argument("--train-imgs",   type=int, default=300, help="Max training images per class (default: 300)")
    parser.add_argument("--valid-imgs",   type=int, default=75,  help="Max validation images per class (default: 75)")
    parser.add_argument("--seed",         type=int, default=42,  help="Random seed (default: 42)")
    parser.add_argument("--source-base",  type=str, default=None,
                        help="Override raw dataset root (folder that contains train/ and valid/). "
                             "Useful when running on Google Colab with kagglehub.")
    args = parser.parse_args()

    # Allow overriding the source root (e.g. from kagglehub download path)
    if args.source_base:
        src_train = Path(args.source_base) / "train"
        src_valid = Path(args.source_base) / "valid"
    else:
        src_train = SOURCE_TRAIN
        src_valid = SOURCE_VALID

    if not src_train.is_dir() or not src_valid.is_dir():
        parser.error(
            "Dataset must contain train/ and valid/ folders; "
            f"checked {src_train.parent}"
        )

    random.seed(args.seed)

    all_classes = sorted(
        entry.name for entry in src_train.iterdir() if entry.is_dir()
    )
    if args.num_classes < 1 or args.num_classes > len(all_classes):
        parser.error(f"--num-classes must be between 1 and {len(all_classes)}")

    selected_classes = random.sample(all_classes, args.num_classes)
    missing_valid_classes = [
        cls for cls in selected_classes if not (src_valid / cls).is_dir()
    ]
    if missing_valid_classes:
        parser.error(
            "Validation data is missing selected classes: "
            + ", ".join(missing_valid_classes)
        )

    print(f"Selected {args.num_classes} classes: {selected_classes}")

    for dest_dir in (DEST_TRAIN, DEST_VALID):
        if dest_dir.exists():
            shutil.rmtree(dest_dir)

    print("\nCreating train subset...")
    make_subset(src_train, DEST_TRAIN, args.train_imgs, selected_classes)

    print("\nCreating valid subset...")
    make_subset(src_valid, DEST_VALID, args.valid_imgs, selected_classes)

    print("\nSubset creation complete.")
    print(f"  Train: {args.num_classes} classes x up to {args.train_imgs} images")
    print(f"  Valid: {args.num_classes} classes x up to {args.valid_imgs} images")