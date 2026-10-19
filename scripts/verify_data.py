"""Verify a locally prepared dataset before NumPy/PyTorch experiments.

Run: python scripts/verify_data.py --data-dir data
Checks every image, the manifest, counts, labels, split fingerprint and preprocessing.
"""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_data import split_fingerprint


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify_dataset(data_dir, expected_fingerprint=None):
    root = Path(data_dir)
    stats = json.loads((root / "stats.json").read_text(encoding="utf-8"))
    with (root / "split_manifest.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    require(bool(records), "The split manifest is empty")
    filenames = [row["filename"] for row in records]
    require(len(filenames) == len(set(filenames)), "Duplicate filenames in the manifest")
    train = {row["filename"] for row in records if row["split"] == "train"}
    test = {row["filename"] for row in records if row["split"] == "test"}
    require(not train & test, "Training and test filenames overlap")

    counts = Counter()
    preprocessing_checks = set()
    for index, row in enumerate(records, 1):
        require(row["split"] in {"train", "test"}, "Unexpected split in manifest")
        require(row["label"] in {"0", "1"}, "Unexpected label in manifest")
        require(row["class"] == {"0": "cat", "1": "dog"}[row["label"]],
                f"Label/class mismatch: {row['filename']}")
        require(Path(row["filename"]).name == row["filename"], "Invalid image filename")
        folder = f"{row['split']}/{row['class']}"
        path = root / folder / row["filename"]
        with Image.open(path) as image:
            image.load()
            require(image.mode == "RGB", f"Image is not RGB: {path}")
            require(image.size == (int(row["width"]), int(row["height"])),
                    f"Image dimensions differ from manifest: {path}")
            if folder not in preprocessing_checks:
                resized = image.resize((64, 64), Image.Resampling.BILINEAR)
                array = np.asarray(resized, dtype=np.float32) / 255.0
                require(array.shape == (64, 64, 3) and np.isfinite(array).all(),
                        f"Invalid preprocessed array: {path}")
                require(0 <= array.min() <= array.max() <= 1,
                        f"Pixels are outside [0, 1]: {path}")
                preprocessing_checks.add(folder)
        counts[folder] += 1
        if index % 5000 == 0:
            print(f"Verified {index}/{len(records)} images", flush=True)

    expected_folders = {f"{split}/{label}" for split in ["train", "test"]
                        for label in ["cat", "dog"]}
    require(set(counts) == expected_folders, "A class or split is missing")
    require(dict(counts) == stats["counts"], "Manifest and stats counts differ")
    disk_counts = {folder: sum(p.is_file() for p in (root / folder).iterdir())
                   for folder in expected_folders}
    require(disk_counts == dict(counts), "Folder and manifest counts differ")
    require(stats["train_total"] == len(train) and stats["test_total"] == len(test),
            "Train/test totals differ from stats")
    require(stats["images_used"] == len(records), "Image total differs from stats")
    fingerprint = split_fingerprint(records)
    require(fingerprint == stats["split_fingerprint"], "Split fingerprint does not match stats")
    if expected_fingerprint is not None:
        require(fingerprint == expected_fingerprint, "Split differs from the expected fingerprint")
    return dict(data_dir=str(root.resolve()), images_verified=len(records),
                counts=dict(counts), train_total=len(train), test_total=len(test),
                split_fingerprint=fingerprint, filename_overlap=0,
                all_images_decoded=True, all_images_rgb=True,
                preprocessing_checked_folders=sorted(preprocessing_checks),
                preprocessing="RGB, 64x64 bilinear, float32 / 255", passed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--expected-fingerprint")
    parser.add_argument("--output", type=Path, default=Path("results/data_verification.json"))
    args = parser.parse_args()
    report = verify_dataset(args.data_dir, args.expected_fingerprint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Verification saved to {args.output}")


if __name__ == "__main__":
    main()
