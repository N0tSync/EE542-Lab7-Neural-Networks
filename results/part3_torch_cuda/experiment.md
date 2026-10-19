# PyTorch CUDA full experiment

The existing `src/part3_torch.py` was used without changing model, training,
evaluation, or timing code. Source commit: `f5f78723311859a6a7e39c7bd789913729940fee`.
Model source SHA256: `16db2ba20e5ffbce70b51327af2601368a2e0a47b732a48b82cb8aa7417b55a3`.

## Environment and reproduction

The existing `ee508` Conda environment was reused without installing or upgrading
packages. Python 3.10.20, PyTorch 2.11.0+cu130,
torchvision 0.26.0+cu130, CUDA runtime 13.0,
NumPy 2.2.6, Pillow 12.2.0, matplotlib 3.10.9.
GPU: NVIDIA GeForce RTX 5080, 15.92 GiB VRAM,
compute capability 12.0.
NVIDIA driver: 610.88.
CUDA forward/backward and all five Part III behavior tests passed.
The restricted agent sandbox hides GPU devices; CUDA runs used host execution.

```bash
conda activate ee508
python -m unittest discover -s tests -p 'test_part3_torch.py' -v
python src/part3_torch.py --data-dir data --device cuda --epochs 10 --lr 0.001 \
  --seed 3 --batch-size 64 --test-block-size 1000 \
  --output results/part3_torch_cuda.json
python scripts/check_results.py
```

The formal console log was captured as `part3_torch_cuda_run.log`.
The separate 2-epoch smoke run used 256 training and 128 test images; its 50%
test accuracy is only a workflow check. The formal configuration was fixed before
the full test; no hyperparameters were changed after observing formal accuracy.

## Configuration

- Dataset: microsoft/cats_vs_dogs, stratified 80/20 split, dataset split seed 0.
- Split fingerprint: `5deb0f27f6c5`.
- Training images: 18,727; test images: 4,682.
- Preprocessing: RGB, 64x64 bilinear resize, float32 pixels divided by 255.
- Architecture: `3->Conv16(3x3,pad1)->ReLU->Pool2->Conv32(3x3,pad1)->ReLU->Pool2->Flatten8192->FC64->ReLU->FC1->ReLU`.
- All 529,505 model parameters are trainable.
- Loss: MSE; optimizer: Adam; learning rate: 0.001.
- Epochs: 10; model/shuffle seed: 3; batch size: 64.
- CPU threads: 4; preload workers: 0.
- Test block sizes: 1000, 1000, 1000, 1000, 682.
- Output ReLU and the strict `score > 0.5` decision rule follow the existing code.

## Measured results

| Metric | Value |
|---|---:|
| Test correct / total | 3684 / 4682 |
| Test accuracy | 78.6843% |
| Test MSE | 0.16209933 |
| Epoch 1 online training MSE | 0.22020275 |
| Epoch 10 online training MSE | 0.07208424 |
| Epoch 10 online training accuracy | 91.4081% |
| Image decode and preload time | 27.484290 s |
| Training time, all 10 epochs | 5.645574 s |
| Training throughput | 33171.12 image presentations/s |
| Total test time | 0.158814 s |
| Forward-only test time | 0.036439 s |
| Test throughput | 29481.10 images/s |

Times exclude disk decoding, preload, warmup, and result export. Training includes
batching, host-to-device transfers, forward/backward, Adam, metrics, and epoch logging.
Total test time includes batching, transfers, inference, metrics, and synchronization.
Forward-only time excludes transfers and metrics. CUDA is synchronized at timing
boundaries. These are single-run measurements on this machine.
Training loss and accuracy are online averages before the corresponding updates.

## Checkpoint verification

The saved checkpoint reloaded successfully. All model values and optimizer state
were finite, with 2930 Adam updates per parameter tensor. A separate
untimed pass reproduced the correct counts for all five blocks, the total of
3684 correct predictions, class prediction counts, and MSE within float32
rounding tolerance. The formal JSON and its timing values were preserved.

| Actual / predicted | Cat | Dog |
|---|---:|---:|
| Cat | 1952 | 396 |
| Dog | 602 | 1732 |

The NumPy and CUDA runs match in full sample counts, split fingerprint, image size,
epochs, learning rate, and model/shuffle seed. They use different architectures,
optimizers, precision, and training batch sizes; a later comparison must describe
those differences rather than attribute the entire performance difference to CUDA.
The final comparison files now include both formal runs; see
[comparison.md](../comparison.md) for the accuracy, throughput, and timing comparison.

## Artifacts

- `../part3_torch_cuda.json`: original formal experiment result.
- `../part3_torch_cuda_run.log`: original console log.
- `loss.png`, `epochs.csv`: loss plot and epoch history.
- `model.pt`: local model and Adam checkpoint.
- `environment.json`: environment, GPU, driver, and source provenance.
- `verification.json`: independent verification outcomes.
- `test_predictions.csv`: all predictions from checkpoint verification.

Raw downloads and dataset images remain excluded from Git. Smoke outputs remain
local. Formal report artifacts are explicitly allowed by `.gitignore`; the binary
PyTorch checkpoint remains local under the existing checkpoint ignore rule.
