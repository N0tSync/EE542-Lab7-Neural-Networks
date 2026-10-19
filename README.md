# EE542-Lab7-Neural-Networks

USC EE 542 Lab 7: NumPy multilayer perceptrons, a manually implemented NumPy CNN,
and a PyTorch CNN with CPU/GPU execution.

## Repository layout

```text
.
├── README.md
├── DATA.md                      # Dataset, preprocessing, and result conventions
├── requirements-part1.txt       # NumPy, plotting, and image dependencies
├── requirements-part3.txt       # PyTorch and torchvision dependencies
├── src/
│   ├── part1_mlp.py              # Three MLP examples
│   ├── part2_numpy_cnn.py        # NumPy training, block evaluation, and timing
│   ├── data_utils.py             # Manifest loading, sampling, and preprocessing
│   ├── part3_torch.py            # PyTorch training and block evaluation
│   └── lab7_common.py            # Shared timing and result validation tools
├── scripts/
│   ├── prepare_data.py           # Download, decode, and split the dataset
│   ├── minipq.py                 # Parquet reader used without pyarrow
│   ├── verify_data.py            # Verify all images and dataset records
│   ├── validate_part2_numpy.py   # Small real-image NumPy validation
│   ├── run_experiments.py       # Sequential model runs and comparison
│   ├── check_results.py         # Result schema and split checks
│   ├── compare.py               # Tables, charts, and throughput comparison
│   └── pipeline_check.py        # Lightweight shared-pipeline check
├── tests/
│   ├── test_prepare_data.py
│   ├── test_part2_numpy_cnn.py
│   ├── test_part2_experiment.py
│   ├── test_part3_torch.py
│   └── test_results_pipeline.py
├── raw/                         # Local downloads; ignored by Git
├── data/                        # Local images and generated records; ignored
└── results/                     # Selected formal results; other outputs ignored
```

Run the commands below from the repository root.
Use English for repository documentation, code comments, and command-line messages.

## Environment

Use Python 3.12 in a virtual environment or an equivalent Conda environment.
For a project-local virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-part1.txt
```

Install the optional download dependency before using `--download`:

```bash
python -m pip install huggingface_hub
```

## Prepare local data

Download the dataset specified in the handout and generate a reproducible,
stratified 80/20 split:

```bash
python scripts/prepare_data.py --download --raw raw --out data --seed 0 --test-frac 0.2
```

If the Parquet files are already available:

```bash
python scripts/prepare_data.py --raw raw --out data --seed 0 --test-frac 0.2
```

The output directory must be empty. For another local export, use an ignored
path such as `--out data/restored`. `--overwrite` deletes and rebuilds the
selected output directory.

Each checkout prepares its own data using the same settings. Raw downloads,
exported images, manifests, statistics, and preview images stay local and are
excluded from Git. The reference full split contains 18,727 training images and
4,682 test images, with fingerprint `5deb0f27f6c5`.

Verify every exported image, the labels, counts, split, and preprocessing:

```bash
python scripts/verify_data.py --data-dir data --expected-fingerprint 5deb0f27f6c5
```

The verification report is saved to `results/data_verification.json`.
Models use RGB images resized to 64x64 with bilinear interpolation and pixels
scaled to [0, 1]. Labels are `cat = 0` and `dog = 1`.
See [DATA.md](DATA.md) for the complete conventions.

## Part I: NumPy MLP

Run regression, binary classification, and multiclass classification:

```bash
python src/part1_mlp.py --output-dir results/part1
```

This writes loss histories, plots, predictions, and a summary CSV.
Classification metrics in Part I are training accuracy.

## Part II: NumPy CNN

The network is:

```text
64x64x3 -> fixed 3x3x3 convolution -> ReLU -> 2x2 max pool
        -> 961 features -> fully connected -> ReLU
```

Only the fully connected weights and bias are updated, using ordinary MSE and
per-image SGD. Convolution and pooling use vectorized NumPy operations.
The output is a nonnegative score; a score greater than 0.5 predicts dog.

Run the synthetic forward-pass and single-update demonstration:

```bash
python src/part2_numpy_cnn.py --demo
```

Run reproducible validation on 32 training images and 16 separate test images:

```bash
python scripts/validate_part2_numpy.py --data-dir data --epochs 20 --seed 3 --data-seed 0
```

The default learning rate is 0.001. Output files in `results/part2_validation/`
include `validation.json`, `loss.csv`, `loss.png`, `test_predictions.csv`, and
`run.log`. Each epoch's loss is the average MSE measured before each SGD update.
The initial and final losses are evaluated on the entire training subset.

Output ReLU can block gradients for some initializations. The validation loop
detects an inactive positive class and reports the problem. The small-sample
run checks computation and learning behavior; it is not a full benchmark.

Run the experiment interface on a small reproducible subset:

```bash
python src/part2_numpy_cnn.py --data-dir data --epochs 2 --seed 3 \
  --max-train-per-class 16 --max-test-per-class 8 \
  --test-block-size 7 --inference-batch-size 4
```

Subset runs default to `results/part2_smoke/`. Their result JSON records the
effective subset fingerprint separately from the original dataset fingerprint.
The sampling seed defaults to `--seed`; use `--data-seed` to override it.

For a full experiment:

```bash
python src/part2_numpy_cnn.py --data-dir data --epochs 10 --lr 0.001 --seed 3 \
  --test-block-size 1000 --inference-batch-size 64 --output-dir results
```

Set `--max-train-per-class 0` and `--max-test-per-class 0` to use the complete
training and test splits; these are the defaults.
`--output` accepts a JSON filename instead of an output directory.

The output directory contains:

```text
part2_numpy_results.json
part2_numpy_loss.png
part2_numpy_run.log
part2_numpy_epochs.csv
part2_numpy_test_predictions.csv
part2_numpy_model.npz
```

The committed full run trained on 18,727 images for 10 epochs and tested on
4,682 images. Test accuracy was 54.25%, with 26.31 seconds of training and
0.493 seconds of testing, excluding data preload. See the
[result JSON](results/part2_numpy_results.json),
[loss plot](results/part2_numpy_loss.png),
[run log](results/part2_numpy_run.log), and
[experiment report](results/part2_numpy_experiment.md) for the configuration,
timing definitions, verification, and limitations.
These formal artifacts and the available comparison outputs are tracked;
dataset files, smoke runs, and other generated outputs remain ignored.

Images are preloaded as uint8 arrays before timing, using about 274 MiB for the
full dataset. Normalization is performed per training image and per inference
batch, so no full float64 image array is allocated. Training includes shuffle,
normalization, convolution, FC gradients, updates, metrics, and epoch logging.
Test time includes normalization, inference, and metrics; forward-only time is
also recorded. Disk decoding and result export are excluded from these timers.
`--test-block-size` controls reporting blocks; `--inference-batch-size` bounds
the images processed in one forward call. Overall accuracy is computed from
total correct predictions, including the final partial block.

Validate generated model results and build the available comparison:

```bash
python scripts/check_results.py
python scripts/compare.py
```

The comparison uses the result files currently available. GPU speedups require
a real PyTorch GPU result with the same data split and compatible settings.

## Part III: PyTorch CNN

Install the PyTorch dependencies and check CUDA availability:

```bash
python -m pip install -r requirements-part3.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

The network uses two convolution layers with ReLU and max pooling, followed by
fully connected layers and output ReLU. Training uses MSE and Adam at 0.001.
All convolution and fully connected parameters are trainable. Account for this
architectural difference when comparing it with the fixed-filter NumPy model.

Start with a small real-image run:

```bash
python src/part3_torch.py --device cuda --epochs 2 \
  --max-train-per-class 128 --max-test-per-class 64 \
  --output results/part3_smoke_cuda.json
```

Run the full GPU experiment:

```bash
python src/part3_torch.py --device cuda --epochs 10 \
  --output results/part3_torch_cuda.json
```

Use `--device cpu` for CPU execution, or omit the flag for automatic selection.
Explicit CUDA execution fails if CUDA is unavailable. Default output filenames
reflect the actual device; choose a CPU filename when overriding the output path
for a CPU run. The script records total test time and
model compute time, with CUDA synchronization around GPU measurements.
It also saves a loss plot, epoch CSV, and checkpoint beside the result JSON.
PyTorch runtime tests and formal GPU experiments have not been verified in
this checkout.

## Tests

```bash
python tests/test_prepare_data.py
python tests/test_prepare_data.py --raw raw
python tests/test_part2_numpy_cnn.py
python tests/test_part2_experiment.py
python tests/test_results_pipeline.py
```

With the PyTorch dependencies installed:

```bash
python -m unittest discover -s tests -p 'test_part3_torch.py' -v
```

The NumPy checks cover reference convolution results, layer shapes, numerical
gradients, reproducible SGD, inactive ReLU outputs, and a fixed convolution filter.
Experiment checks cover uint8 normalization, partial blocks, weighted accuracy,
manifest validation, and CLI artifacts using temporary image fixtures.
