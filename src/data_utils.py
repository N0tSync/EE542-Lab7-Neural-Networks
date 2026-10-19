"""Local manifest selection and shared RGB/64x64/bilinear preprocessing."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_data_info(data_dir):
    root = Path(data_dir)
    stats = json.loads((root / "stats.json").read_text(encoding="utf-8"))
    with (root / "split_manifest.csv").open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    if not records or len(records) != stats["images_used"]:
        raise ValueError("Manifest image count does not match stats.json")
    return stats, records


def select_samples(records, split, per_class, seed=0):
    """Select a balanced, reproducible subset within an existing split."""
    if split not in {"train", "test"} or per_class < 1:
        raise ValueError("Expected train/test split and a positive per-class count")
    rng = np.random.default_rng(seed)
    selected = []
    for label in ["0", "1"]:
        candidates = sorted((r for r in records if r["split"] == split and r["label"] == label),
                            key=lambda r: r["filename"])
        if len(candidates) < per_class:
            raise ValueError(f"Not enough {split} images for label {label}")
        selected.extend(candidates[i] for i in rng.permutation(len(candidates))[:per_class])
    return selected


def load_images(data_dir, records):
    """Return NHWC float32 images in [0, 1] and integer labels."""
    if not records:
        raise ValueError("No images selected")
    root = Path(data_dir)
    images, labels = [], []
    for row in records:
        expected_class = {"0": "cat", "1": "dog"}.get(row["label"])
        if (row["class"] != expected_class or row["split"] not in {"train", "test"}
                or Path(row["filename"]).name != row["filename"]):
            raise ValueError("Invalid manifest record")
        path = root / row["split"] / row["class"] / row["filename"]
        with Image.open(path) as image:
            image = image.convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.float32) / 255.0)
        labels.append(int(row["label"]))
    return np.stack(images), np.asarray(labels, dtype=np.int64)
