"""
Quick tests for prepare_data.py and minipq.py (runs in a few seconds).

Usage (from the repository root):
    python tests/test_prepare_data.py             # tests 1-6 (no download needed)
    python tests/test_prepare_data.py --raw raw   # also test 7: read the real Parquet files

Each test prints PASS or FAIL. The last line says whether everything passed.
"""
import argparse
import glob
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_data as P  # noqa: E402

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append(bool(condition))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def jpeg_bytes(color, size=(120, 90), fmt="JPEG", mode="RGB"):
    img = Image.new("RGB", size, color)
    if mode != "RGB":
        img = img.convert(mode)
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def fake_rows():
    """60 cats + 40 dogs, plus 4 broken rows. Same layout as the real Parquet rows."""
    rows = []
    rng = np.random.default_rng(0)
    for i in range(100):
        label = 0 if i < 60 else 1               # cats first, dogs after (like the real files)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        if i % 10 == 3:
            raw = jpeg_bytes(color, fmt="PNG", mode="L")      # grayscale PNG -> must become RGB
        else:
            raw = jpeg_bytes(color)
        rows.append(("train-00000-of-00002.parquet", i, raw, label))
    good = jpeg_bytes((10, 10, 10))
    rows.append(("train-00001-of-00002.parquet", 0, good[:200], 0))       # truncated file
    rows.append(("train-00001-of-00002.parquet", 1, None, 1))             # missing bytes
    rows.append(("train-00001-of-00002.parquet", 2, b"not an image", 0))  # garbage
    rows.append(("train-00001-of-00002.parquet", 3, jpeg_bytes((1, 2, 3), size=(4, 4)), 1))  # 4x4 image
    return rows


def run_fake_tests(tmp):
    out1, out2, out3 = (os.path.join(tmp, n) for n in ("a", "b", "c"))
    stats = P.build_dataset(iter(fake_rows()), out1, test_frac=0.2, seed=0, overwrite=True)
    print()

    # 1. bad images are skipped and logged
    with open(os.path.join(out1, "bad_images.csv")) as f:
        n_bad = sum(1 for _ in f) - 1
    check("1. broken images are skipped and logged", stats["bad_rows"] == 4 and n_bad == 4,
          f"{n_bad} rows in bad_images.csv, expected 4")

    # 2. four folders with the right stratified counts (60 cats, 40 dogs, 20% test)
    want = {"train/cat": 48, "test/cat": 12, "train/dog": 32, "test/dog": 8}
    on_disk = {k: len(os.listdir(os.path.join(out1, k))) for k in want}
    check("2. stratified 80/20 split per class", on_disk == want, f"got {on_disk}")

    # 3. every saved image is RGB and opens cleanly
    files = glob.glob(os.path.join(out1, "*", "*", "*.jpg"))
    modes = {Image.open(f).mode for f in files}
    check("3. all saved images are RGB", modes == {"RGB"} and len(files) == 100,
          f"{len(files)} images, modes {modes}")

    # 4. no overlap, and folder matches label
    with open(os.path.join(out1, "split_manifest.csv")) as f:
        import csv
        rows = list(csv.DictReader(f))
    train = {r["filename"] for r in rows if r["split"] == "train"}
    test = {r["filename"] for r in rows if r["split"] == "test"}
    mismatched = [r for r in rows if (r["class"] == "cat") != (r["label"] == "0")]
    check("4. train/test do not overlap; folders match labels",
          not (train & test) and not mismatched, f"overlap {len(train & test)}, mismatches {len(mismatched)}")

    # 5. same seed -> same split; different seed -> different split
    s2 = P.build_dataset(iter(fake_rows()), out2, seed=0, overwrite=True)
    s3 = P.build_dataset(iter(fake_rows()), out3, seed=1, overwrite=True)
    print()
    check("5. same seed gives the same split, new seed a new one",
          stats["split_fingerprint"] == s2["split_fingerprint"] != s3["split_fingerprint"],
          f'{stats["split_fingerprint"]} / {s2["split_fingerprint"]} / {s3["split_fingerprint"]}')

    # 6. stats.json and sample image are written
    with open(os.path.join(out1, "stats.json")) as f:
        js = json.load(f)
    check("6. stats.json and sample_check.png are written",
          js["train_total"] == 80 and js["test_total"] == 20,
          "sample_check.png " + ("found" if os.path.exists(os.path.join(out1, "sample_check.png"))
                                 else "missing (matplotlib not installed - optional)"))


def run_real_parquet_test(raw_dir):
    files = P.find_parquet_files(raw_dir)
    labels, n = set(), 0
    for src, row, raw, label in P.iter_parquet_rows(files[:1]):
        img = Image.open(io.BytesIO(raw))
        img.load()
        labels.add(label)
        n += 1
        if n == 200:
            break
    check("7. real Parquet file can be read and its images decoded",
          n == 200 and labels <= {0, 1}, f"read {n} rows from {os.path.basename(files[0])}, labels {labels}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", help="folder with the real .parquet files (optional)")
    args = ap.parse_args()

    tmp = tempfile.mkdtemp(prefix="test_prepare_")
    try:
        print("Running prepare_data.py on 104 fake rows (100 good, 4 broken)...\n")
        run_fake_tests(tmp)
        if args.raw:
            run_real_parquet_test(args.raw)
        else:
            print("[SKIP] 7. real Parquet test (run with --raw <folder with the .parquet files>)")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(RESULTS)
    print(f"\n{passed}/{len(RESULTS)} tests passed - " + ("ALL GOOD" if passed == len(RESULTS) else "SOMETHING FAILED"))
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
