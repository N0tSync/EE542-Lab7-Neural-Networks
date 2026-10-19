# NumPy CNN full experiment

The full NumPy experiment completed using source commit `01142c734805d3d5daa4f06a78dbe2edf18e626a`.

## Reproduce

Run from the repository root in the project Python environment:

```bash
python src/part2_numpy_cnn.py --data-dir data --epochs 10 --lr 0.001 --seed 3 \
  --test-block-size 1000 --inference-batch-size 64 --output-dir results
python scripts/check_results.py
python scripts/compare.py
```

Prepare and verify the local data first using the commands in README.md.
Raw downloads, exported images, and dataset metadata are ignored by Git.
The formal NumPy artifacts listed below and the available comparison outputs
are committed; validation runs and other generated outputs remain ignored.

## Configuration

- Dataset: `microsoft/cats_vs_dogs`; stratified 80/20 split with dataset split seed 0.
- Split fingerprint: `5deb0f27f6c5`.
- Training: 18,727 images (9,392 cats, 9,335 dogs).
- Testing: 4,682 images (2,348 cats, 2,334 dogs).
- Preprocessing: RGB, bilinear resize to 64x64, pixels divided by 255.
- Architecture: `64x64x3 -> fixed conv3 -> ReLU -> pool2 -> 961 -> FC1 -> ReLU`.
- Parameters: 962 trainable FC values, 27 fixed convolution values.
- Training: per-image SGD, ordinary MSE, 10 epochs, learning rate 0.001, model/shuffle seed 3.
- Testing: logical blocks of 1000, forward batches of at most 64.
- Hardware: AMD Ryzen 7 9800X3D 8-Core Processor.
- Runtime: Python 3.12.13, NumPy 2.4.4; Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.39.

## Measured results

| Metric | Value |
|---|---:|
| Test correct / total | 2540 / 4682 |
| Test accuracy | 54.2503% |
| Majority-class test baseline | 50.1495% |
| Test MSE | 0.24904945 |
| Epoch 1 online training MSE | 0.24882524 |
| Epoch 10 online training MSE | 0.23937690 |
| Epoch 10 online training accuracy | 58.4290% |
| Data validation, decoding and preload | 39.351478 s |
| Training time, all 10 epochs | 26.309508 s |
| Training throughput | 7117.96 image presentations/s |
| Total testing time | 0.493379 s |
| Forward-only testing time | 0.444566 s |
| Testing throughput | 9489.66 images/s |
| Preloaded image storage | 274.32 MiB |

Training loss and accuracy are online averages before individual SGD updates;
they do not evaluate one fixed checkpoint over the complete training set.
Training and test timers exclude preload and artifact export. They include
normalization, model computation and metrics. Training also includes shuffle,
gradient updates and epoch logging. Forward-only test timing excludes normalization
and metrics. These are measurements from one run on this machine.

### Test confusion matrix

Rows are actual labels; columns are predicted labels. Cat = 0, dog = 1.

| Actual / predicted | Cat | Dog |
|---|---:|---:|
| Cat | 1015 | 1333 |
| Dog | 809 | 1525 |

The model predicted 1,824 cats and 2,858 dogs.
Accuracy is modest: 4.10 percentage points above
the majority-class baseline. The fixed single-filter feature extractor and output
ReLU follow the handout. No hyperparameters were changed after this formal test.

## Verification and artifacts

Result schema validation passed. Independent checks confirmed all saved test labels,
predictions, MSE, accuracy, confusion counts and weighted block aggregation,
including the final 682-image block. Epoch CSV records match the JSON.
The convolution checkpoint exactly matches initialization, the FC weights changed,
and all parameters and recorded values are finite. Loading the saved checkpoint
reproduced predictions on nine test images across class and block boundaries.

- `part2_numpy_results.json`: machine-readable full result.
- `part2_numpy_loss.png`: training loss plot.
- `part2_numpy_run.log`: loading, epoch and block timing log.
- `part2_numpy_epochs.csv`: epoch loss, accuracy and timing.
- `part2_numpy_test_predictions.csv`: all ordered test scores and predictions.
- `part2_numpy_model.npz`: fixed convolution and trained FC checkpoint.

At the time of this NumPy experiment, no formal PyTorch result was available.
A subsequent CUDA experiment is recorded in `part3_torch_cuda.json` and
`part3_torch_cuda/experiment.md`. The final comparison table and chart now include
both formal runs; see [comparison.md](comparison.md).
The two models have different architectures and trainable parameters, which must
be stated when interpreting the comparison.
