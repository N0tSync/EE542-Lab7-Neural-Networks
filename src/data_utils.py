"""Local manifest selection and shared RGB/64x64/bilinear preprocessing."""

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_data_info(data_dir):
    """Validate the manifest, actual filenames, labels, counts and fingerprint."""
    root = Path(data_dir)
    stats = json.loads((root / "stats.json").read_text(encoding="utf-8"))
    with (root / "split_manifest.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not records or len(records) != stats["images_used"]:
        raise ValueError("Manifest image count does not match stats.json")
    expected = {}
    names = set()
    counts = Counter()
    for row in records:
        label = {"cat": "0", "dog": "1"}.get(row["class"])
        if (row["split"] not in {"train", "test"} or row["label"] != label
                or Path(row["filename"]).name != row["filename"]
                or row["filename"] in {"", ".", ".."}):
            raise ValueError("Invalid manifest record")
        if row["filename"] in names:
            raise ValueError("Duplicate or overlapping train/test filenames")
        names.add(row["filename"])
        relative = f"{row['split']}/{row['class']}/{row['filename']}"
        expected[relative] = int(row["label"])
        counts[f"{row['split']}/{row['class']}"] += 1
    actual = {}
    for split in ["train", "test"]:
        for label, name in enumerate(["cat", "dog"]):
            folder = root / split / name
            if not folder.is_dir():
                raise ValueError(f"Missing image directory: {folder}")
            for path in folder.iterdir():
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                    actual[f"{split}/{name}/{path.name}"] = label
    if actual != expected:
        raise ValueError("Image files/labels differ from split_manifest.csv")
    if (dict(counts) != stats["counts"]
            or sum(r["split"] == "train" for r in records) != stats["train_total"]
            or sum(r["split"] == "test" for r in records) != stats["test_total"]
            or manifest_fingerprint(records) != stats["split_fingerprint"]):
        raise ValueError("Dataset counts/fingerprint differ from stats.json")
    return stats, records


def manifest_fingerprint(records):
    """Hash the effective (filename, split) pairs, including subset selections."""
    digest = hashlib.sha256()
    for row in sorted(records, key=lambda r: r["filename"]):
        digest.update(f"{row['filename']},{row['split']}\n".encode())
    return digest.hexdigest()[:12]


def select_samples(records, split, per_class=0, seed=0):
    """Select a reproducible subset; zero uses the entire original split."""
    if split not in {"train", "test"} or per_class < 0:
        raise ValueError("Expected train/test split and a nonnegative per-class count")
    rng = np.random.default_rng(seed)
    selected = []
    for label in ["0", "1"]:
        candidates = sorted((r for r in records if r["split"] == split and r["label"] == label),
                            key=lambda r: r["filename"])
        if not candidates or len(candidates) < per_class:
            raise ValueError(f"Not enough {split} images for label {label}")
        if per_class:
            selected.extend(candidates[i] for i in rng.permutation(len(candidates))[:per_class])
        else:
            selected.extend(candidates)
    return selected


def load_images(data_dir, records, uint8=False):
    """Preallocate NHWC images as uint8 [0,255] or normalized float32 [0,1]."""
    if not records:
        raise ValueError("No images selected")
    root = Path(data_dir)
    images = np.empty((len(records), 64, 64, 3), dtype=np.uint8 if uint8 else np.float32)
    labels = np.empty(len(records), dtype=np.int64)
    for index, row in enumerate(records):
        expected_class = {"0": "cat", "1": "dog"}.get(row["label"])
        if (row["class"] != expected_class or row["split"] not in {"train", "test"}
                or Path(row["filename"]).name != row["filename"]):
            raise ValueError("Invalid manifest record")
        path = root / row["split"] / row["class"] / row["filename"]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            array = np.asarray(image, dtype=np.uint8)
            images[index] = array if uint8 else array.astype(np.float32) / 255.0
        labels[index] = int(row["label"])
    return images, labels
