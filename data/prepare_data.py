"""
Lab 7 - Dataset preparation (shared by Part II and Part III)

Turns the Hugging Face "microsoft/cats_vs_dogs" parquet files (23,410 rows, one 'train'
split, label 0 = cat, 1 = dog) into one fixed, reproducible train/test split:

    data/
      train/cat/  train/dog/          training images
      test/cat/   test/dog/           unseen test images
      split_manifest.csv              every image: file, split, label, source file + row, size
      bad_images.csv                  rows that could not be decoded (skipped)
      stats.json                      image counts per folder, settings, split fingerprint
      sample_check.png                a few images with their labels, for a visual check

Steps
  1. read the parquet files batch by batch (images + labels)
  2. decode every image and convert it to RGB; record and skip any that fail
  3. split cats and dogs separately (stratified), e.g. 80% train / 20% test, fixed seed
  4. write the four folders, the manifest, the bad-image log and the counts

The images are saved at their original size. Resizing to 64x64 and dividing by 255 is done
when the models load them (see DATA.md), so both models apply exactly the same rule.

Usage
    pip install pillow matplotlib   (pyarrow optional - minipq.py is used without it;
                                     huggingface_hub needed only for --download)
    python prepare_data.py --download --raw raw --out data
    python prepare_data.py --raw raw --out data      # parquet files already in raw/
    python prepare_data.py --raw raw --out data_small --max-per-class 2000   # smaller subset
Re-running with the same --seed gives the identical split (same fingerprint in stats.json).
"""
import argparse
import csv
import glob
import hashlib
import io
import json
import os
import shutil
import time

import numpy as np
from PIL import Image

REPO_ID = "microsoft/cats_vs_dogs"
CLASS_NAMES = {0: "cat", 1: "dog"}


# ---------------------------------------------------------------------------
# Step 0 (optional): download
# ---------------------------------------------------------------------------
def download(raw_dir):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub, or download the .parquet files from\n"
                         f"https://huggingface.co/datasets/{REPO_ID}/tree/main/data into {raw_dir}/")
    print(f"Downloading {REPO_ID} parquet files (~722 MB) into {raw_dir}/ ...")
    snapshot_download(repo_id=REPO_ID, repo_type="dataset",
                      allow_patterns=["data/*.parquet"], local_dir=raw_dir)


def find_parquet_files(raw_dir):
    files = sorted(glob.glob(os.path.join(raw_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise SystemExit(f"No .parquet files found under {raw_dir}/ (use --download, or copy them there)")
    return files


# ---------------------------------------------------------------------------
# Step 1: read rows from the parquet files
# ---------------------------------------------------------------------------
def iter_parquet_rows(files, batch_size=256):
    """Yield (source_file, row_in_file, image_bytes, label) without loading everything at once.
    Uses pyarrow when installed; otherwise the bundled pure-Python reader minipq.py."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        import minipq
        print("  (pyarrow not installed - using the bundled pure-Python reader minipq.py)")
        for path in files:
            for row, r in enumerate(minipq.iter_rows(path, ["image.bytes", "labels"])):
                yield os.path.basename(path), row, r["image.bytes"], r["labels"]
        return
    for path in files:
        pf = pq.ParquetFile(path)
        names = pf.schema_arrow.names
        image_col = "image" if "image" in names else names[0]
        label_col = "labels" if "labels" in names else ("label" if "label" in names else None)
        if label_col is None:
            raise SystemExit(f"{path}: no 'labels' column (columns: {names})")
        row = 0
        for batch in pf.iter_batches(batch_size=batch_size, columns=[image_col, label_col]):
            images = batch.column(image_col).to_pylist()
            labels = batch.column(label_col).to_pylist()
            for cell, label in zip(images, labels):
                raw = cell.get("bytes") if isinstance(cell, dict) else cell
                yield os.path.basename(path), row, raw, label
                row += 1


# ---------------------------------------------------------------------------
# Step 2: decode, convert to RGB, save (or record as bad)
# ---------------------------------------------------------------------------
def decode_and_save(raw, label, out_path):
    """Return (width, height) on success; raise on a corrupt image."""
    if raw is None:
        raise ValueError("no image bytes")
    if label not in CLASS_NAMES:
        raise ValueError(f"unexpected label {label!r}")
    img = Image.open(io.BytesIO(raw))
    img.load()                                   # forces a full decode; truncated files fail here
    if img.width < 8 or img.height < 8:
        raise ValueError(f"image too small ({img.width}x{img.height})")
    if img.format == "JPEG" and img.mode == "RGB":
        with open(out_path, "wb") as f:          # already an RGB JPEG: keep the original bytes
            f.write(raw)
    else:
        img.convert("RGB").save(out_path, "JPEG", quality=95)
    return img.width, img.height


def export_all(rows, staging_dir, max_rows=0):
    """Decode every row into staging_dir/<class>/; return (good records, bad records)."""
    for name in CLASS_NAMES.values():
        os.makedirs(os.path.join(staging_dir, name), exist_ok=True)
    good, bad = [], []
    t0 = time.perf_counter()
    for k, (src, row, raw, label) in enumerate(rows):
        if max_rows and k >= max_rows:
            break
        stem = os.path.splitext(src)[0].split("-of-")[0]          # e.g. "train-00000"
        name = CLASS_NAMES.get(label, "unknown")
        filename = f"{name}_{stem}_{row:05d}.jpg"
        out_path = os.path.join(staging_dir, name, filename)
        try:
            w, h = decode_and_save(raw, label, out_path)
            good.append({"filename": filename, "label": int(label), "class": name,
                         "source_file": src, "source_row": row, "width": w, "height": h})
        except Exception as e:
            if os.path.exists(out_path):
                os.remove(out_path)
            bad.append({"source_file": src, "source_row": row, "label": label,
                        "error": f"{type(e).__name__}: {e}"})
        if (k + 1) % 2000 == 0:
            print(f"  {k+1} rows processed ({time.perf_counter()-t0:.0f}s), {len(bad)} bad so far")
    return good, bad


# ---------------------------------------------------------------------------
# Step 3: stratified split (each class shuffled and split on its own)
# ---------------------------------------------------------------------------
def stratified_split(records, test_frac, max_per_class, seed):
    rng = np.random.default_rng(seed)
    for label in sorted(CLASS_NAMES):
        recs = sorted((r for r in records if r["label"] == label), key=lambda r: r["filename"])
        order = rng.permutation(len(recs))
        if max_per_class:
            order = order[:max_per_class]
        n_test = int(round(len(order) * test_frac))
        for i in order[:n_test]:
            recs[i]["split"] = "test"
        for i in order[n_test:]:
            recs[i]["split"] = "train"
    return [r for r in records if "split" in r]          # records beyond max_per_class are dropped


# ---------------------------------------------------------------------------
# Step 4: folders, manifest, counts, sample image
# ---------------------------------------------------------------------------
def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def split_fingerprint(records):
    """Short hash of (filename, split) pairs: equal fingerprints mean identical splits."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda r: r["filename"]):
        h.update(f'{r["filename"]},{r["split"]}\n'.encode())
    return h.hexdigest()[:12]


def save_sample_grid(out_dir, records, seed, n=12):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    rng = np.random.default_rng(seed)
    picks = [records[i] for i in rng.choice(len(records), size=min(n, len(records)), replace=False)]
    cols = 6
    rows = (len(picks) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.2 * cols, 2.5 * rows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, r in zip(np.atleast_1d(axes).ravel(), picks):
        img = Image.open(os.path.join(out_dir, r["split"], r["class"], r["filename"]))
        ax.imshow(img)
        ax.set_title(f'label={r["label"]} ({r["class"]})\n{r["split"]}', fontsize=8)
    fig.suptitle("Spot check: does each picture match its label?")
    fig.tight_layout()
    path = os.path.join(out_dir, "sample_check.png")
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def build_dataset(rows, out_dir, test_frac=0.2, max_per_class=0, seed=0,
                  max_rows=0, source_files=(), overwrite=False):
    if os.path.exists(out_dir) and os.listdir(out_dir):
        if not overwrite:
            raise SystemExit(f"{out_dir}/ is not empty. Use --overwrite to rebuild it.")
        shutil.rmtree(out_dir)
    staging = os.path.join(out_dir, "_staging")
    t0 = time.perf_counter()

    print("Steps 1-2: reading parquet rows, decoding and converting to RGB ...")
    good, bad = export_all(rows, staging, max_rows)
    print(f"  decoded {len(good)} images, skipped {len(bad)} bad rows")

    print(f"Step 3: stratified split, test_frac={test_frac}, seed={seed}")
    records = stratified_split(good, test_frac, max_per_class, seed)

    print("Step 4: moving images into train/ and test/ folders")
    for split in ("train", "test"):
        for name in CLASS_NAMES.values():
            os.makedirs(os.path.join(out_dir, split, name), exist_ok=True)
    for r in records:
        os.replace(os.path.join(staging, r["class"], r["filename"]),
                   os.path.join(out_dir, r["split"], r["class"], r["filename"]))
    shutil.rmtree(staging)

    fields = ["filename", "split", "label", "class", "source_file", "source_row", "width", "height"]
    records.sort(key=lambda r: (r["split"] != "train", r["label"], r["filename"]))
    write_csv(os.path.join(out_dir, "split_manifest.csv"), records, fields)
    write_csv(os.path.join(out_dir, "bad_images.csv"), bad,
              ["source_file", "source_row", "label", "error"])

    counts = {f"{s}/{n}": sum(1 for r in records if r["split"] == s and r["class"] == n)
              for s in ("train", "test") for n in CLASS_NAMES.values()}
    # Re-count from disk, so the numbers are what is actually in the folders.
    on_disk = {k: len(os.listdir(os.path.join(out_dir, k))) for k in counts}
    assert on_disk == counts, f"folder counts {on_disk} differ from manifest {counts}"
    train_names = {r["filename"] for r in records if r["split"] == "train"}
    test_names = {r["filename"] for r in records if r["split"] == "test"}
    assert not train_names & test_names, "train and test overlap"

    stats = {
        "dataset": REPO_ID, "source_files": list(source_files),
        "rows_read": len(good) + len(bad), "bad_rows": len(bad), "images_used": len(records),
        "counts": counts,
        "train_total": counts["train/cat"] + counts["train/dog"],
        "test_total": counts["test/cat"] + counts["test/dog"],
        "test_frac": test_frac, "seed": seed, "max_per_class": max_per_class,
        "labels": {str(k): v for k, v in CLASS_NAMES.items()},
        "preprocessing_for_models": "RGB, resize to 64x64 (bilinear), pixel / 255 -> [0,1]",
        "split_fingerprint": split_fingerprint(records),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    sample = save_sample_grid(out_dir, records, seed)

    print("\n| folder | images |\n|---|---|")
    for k, v in counts.items():
        print(f"| {out_dir}/{k}/ | {v} |")
    print(f"| total | {len(records)} |")
    print(f"\nbad rows skipped: {len(bad)}   split fingerprint: {stats['split_fingerprint']}")
    print(f"wrote {out_dir}/split_manifest.csv, bad_images.csv, stats.json"
          + (", sample_check.png" if sample else ""))
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default="raw", help="folder with the .parquet files")
    ap.add_argument("--out", default="data")
    ap.add_argument("--download", action="store_true", help="download the parquet files first")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = use every image")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-rows", type=int, default=0, help="read only the first N rows (quick test)")
    ap.add_argument("--overwrite", action="store_true", help="delete and rebuild --out")
    args = ap.parse_args()

    if args.download:
        download(args.raw)
    files = find_parquet_files(args.raw)
    print(f"Parquet files: {[os.path.basename(f) for f in files]}")
    build_dataset(iter_parquet_rows(files), args.out, args.test_frac, args.max_per_class,
                  args.seed, args.max_rows, [os.path.basename(f) for f in files], args.overwrite)


if __name__ == "__main__":
    main()
