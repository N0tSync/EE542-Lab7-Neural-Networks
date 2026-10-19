# EE542-Lab7-Neural-Networks

## Python environment

Use a project-local virtual environment for every install and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-part1.txt
```

## Part I: NumPy MLP examples

```bash
python src/part1_mlp.py --output-dir results/part1
```

This saves CSV metrics, loss curves, a regression fit and classification decision
regions. Classification metrics in Part I are training accuracy.

## Part III (team member B): PyTorch CNN

The dependency file targets Python 3.12, PyTorch 2.9.1, torchvision 0.24.1
and CUDA 12.6 wheels from the [official version matrix](https://pytorch.org/get-started/previous-versions/#v291).
Part III is currently an implementation handoff: the tests and real-data runs
below still need to be executed by the teammate responsible for verification.

```bash
source .venv/bin/activate
python -m pip install -r requirements-part3.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
python -m unittest -v test_part3_torch
```

The tests use generated fixtures and require no downloaded dataset. For training,
prepare `data/train/{cat,dog}`, `data/test/{cat,dog}`, `data/stats.json` and
`data/split_manifest.csv` according to [DATA.md](DATA.md). The script rejects
missing images or a mismatch between actual files and the recorded split.

First verify the pipeline using a reproducible subset of the real images:

```bash
python src/part3_torch.py --device cuda --epochs 2 \
  --max-train-per-class 128 --max-test-per-class 64 \
  --output results/part3_smoke_cuda.json
```

Then run the full experiment:

```bash
set -o pipefail
mkdir -p results
python -u src/part3_torch.py --device cuda --epochs 10 \
  --output results/part3_torch_cuda.json | tee results/part3_torch_cuda.log
```

Use `--device cpu` for CPU execution or omit the device flag for automatic device
selection. An explicit `--device cuda` fails if CUDA is unavailable. Both the model
and tensors run on the selected device.

The model is `Conv(3,16) -> ReLU -> MaxPool -> Conv(16,32) -> ReLU -> MaxPool ->
FC(8192,64) -> ReLU -> FC(64,1) -> ReLU`, trained with MSE and Adam at 0.001. The
output layer starts with small weights and bias 0.5 to reduce inactive output
ReLUs. All convolution and fully connected parameters are trainable. This differs
from Part II's fixed convolution filter; comparisons must account for that.

Images are RGB, resized to 64x64 with bilinear interpolation, and divided by 255.
They are preloaded into CPU memory before timing (about 1.1 GiB for the full set).
Training batches are shuffled; test order is fixed. Labels are float32 columns
matching the `[N,1]` output. A prediction greater than 0.5 means dog.

The result JSON follows DATA.md and includes the effective split fingerprint,
source fingerprint, device, configuration, per-epoch loss/times, accuracy, test
blocks of 1,000 images, and throughput. Subset runs have their own fingerprint.
Training time spans all epochs and includes batching, transfers, forward/backward,
Adam, metrics and epoch logging; it excludes disk decoding, optimizer setup,
warmup and saving. Per-epoch times are also recorded separately.
Test time includes batching, transfers, inference, metrics and synchronization.
`test_compute_seconds` measures only forward calls, with CUDA synchronization.

Alongside the result JSON, a directory with the same stem contains `epochs.csv`,
`loss.png` and `model.pt` (model/optimizer state, configuration and label mapping).
Reload the model with:

```python
import torch
from src.part3_torch import CatDogCNN

checkpoint = torch.load("results/part3_torch_cuda/model.pt", map_location="cpu", weights_only=True)
model = CatDogCNN()
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
```

Virtual environments, downloaded images, model weights and `results/` are ignored
by Git, following the repository's current `.gitignore` settings.
