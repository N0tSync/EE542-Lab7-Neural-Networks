# Data and Results Conventions (owned by C, followed by A and B)

## 1. Data source

- Hugging Face `microsoft/cats_vs_dogs`: 23,410 samples in a single `train` split
- The download is two Parquet files under `raw/` (about 722 MB). Each row has two columns:
  - `image`: the encoded image bytes
  - `labels`: `0` = cat, `1` = dog
- The dataset has no test split, so we make our own train/test split (see section 3)

## 2. Building the data (C runs this once, then shares `data/` with A and B)

Run commands from the repository root. Data preparation code lives in `scripts/`;
`data/` contains images and generated records.

```bash
pip install pillow matplotlib huggingface_hub   # pyarrow optional; minipq.py is used without it
python scripts/prepare_data.py --download --raw raw --out data      # download + export + split
# Parquet files already downloaded:  python scripts/prepare_data.py --raw raw --out data
# Tests without real data:          python tests/test_prepare_data.py
# Include real Parquet reading:     python tests/test_prepare_data.py --raw raw
```

The output directory must be empty. If `data/` already contains records, use a new
output directory such as `--out data_restored`, or explicitly rebuild it with
`--overwrite` (which deletes and recreates the selected output directory).

What the script does:

1. Reads the Parquet files in batches, so it never loads everything into memory
2. Fully decodes every image and converts it to RGB; images that fail are logged in
   `bad_images.csv` and skipped
3. Shuffles cats and dogs separately and splits each 80% / 20% (stratified split, `seed=0`)
4. Writes the four folders and three record files, then verifies that the image counts on
   disk match the manifest and that train and test do not overlap

Running again with the same `--seed` produces exactly the same split.

## 3. Output

| Path | Contents |
|---|---|
| `data/train/cat/`, `data/train/dog/` | Training images (original size, RGB JPEG) |
| `data/test/cat/`, `data/test/dog/` | Test images; never use these for training |
| `data/split_manifest.csv` | For every image: file name, split, label, source Parquet file and row, original width and height |
| `data/bad_images.csv` | Skipped rows and the reason |
| `data/stats.json` | Image counts per folder, split settings, and the **split fingerprint** `split_fingerprint` |
| `data/sample_check.png` | 12 random images with their labels, for a visual label check |

Image counts (actual run on 2026-10-16: 23,410 rows read, 23,409 images used, about 85 seconds):

| Folder | Images |
|---|---|
| `data/train/cat/` | 9,392 |
| `data/train/dog/` | 9,335 |
| `data/test/cat/` | 2,348 |
| `data/test/dog/` | 2,334 |
| Train total / test total | 18,727 / 4,682 |
| Bad images skipped | 1 (`train-00000`, row 7223, only 4×4 pixels) |
| Split fingerprint | `5deb0f27f6c5` |

The saved manifest has no train/test filename overlap, and its labels agree with the
class names. These records describe the earlier export; verify that the image files
exist locally after restoring the data. `sample_check.png` is available for visual checks.

## 4. Shared preprocessing rules (A and B must match)

| Item | Rule |
|---|---|
| Color | RGB, 3 channels |
| Size | Resize to 64×64 with **bilinear interpolation** |
| Pixels | Divide by 255, range [0, 1] |
| Labels | `cat = 0`, `dog = 1` |
| Decision | Model output > 0.5 means dog |

The bilinear rule is easy to get wrong: PIL's `resize()` defaults to bicubic, while
torchvision's `Resize` defaults to bilinear. Both model implementations must set bilinear explicitly:

- A (NumPy): `Image.open(p).convert("RGB").resize((64, 64), Image.BILINEAR)`, then `/ 255.0`
- B (PyTorch): `transforms.Resize((64, 64), interpolation=InterpolationMode.BILINEAR)` + `transforms.ToTensor()`
  (`ImageFolder` sorts folder names alphabetically, which gives `cat = 0`, `dog = 1`)

## 5. Results format

Each run writes its results to a JSON file in `results/` (A: `part2_numpy_results.json`,
B: `part3_torch_cuda.json`; CPU runs must use a CPU filename and device value). Required fields:

| Field | Meaning |
|---|---|
| `model`, `device` | Model name; device used (give the GPU model) |
| `data_split_fingerprint` | Read from `data/stats.json`; shows both models used the same split |
| `n_train`, `n_test`, `epochs`, `seed`, `image_size` | Run configuration |
| `train_seconds`, `train_images_per_sec` | Total training time over all epochs; images per second |
| `test_seconds`, `test_images_per_sec` | Total test time; images per second |
| `test_accuracy` | Test-set accuracy (0–1) |
| `epochs_log` | Loss and time for each epoch |
| `test_blocks` | Image count, accuracy and time for each test block (1,000 images by default) |

Timing rules:

- Training time: from the start of the first epoch to the end of the last, excluding image loading
- Test time: inference over all test blocks; the GPU version also records pure model compute time
  `test_compute_seconds`
- On the GPU, call `torch.cuda.synchronize()` before and after timing, or the times come out too low

Planned integration tools (not yet implemented): C will add `scripts/run_experiments.py`
to run both models and save logs, and `scripts/compare.py` to check the split fingerprint
and experiment settings and produce `results/comparison.md` and `results/comparison.png`
for the report.

## 6. Keeping the comparison fair

- Both models use the same `data/` folder (same fingerprint) and the same number of epochs
- The NumPy version is slow (about 100 images/s; 10 epochs on the full set take about 35 minutes).
  If A needs a subset, C builds one with `--max-per-class` (e.g. `data_small/`), and **both A and B
  use that same subset**
- State the CPU and GPU models in the report
