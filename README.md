# EE542 Lab 7: Neural Networks

USC EE 542 Lab 7 explores neural networks through three implementations:
NumPy MLP examples, a manually implemented NumPy CNN, and a PyTorch CNN with
CUDA support. Both CNNs classify cats and dogs using the same prepared dataset.

## Setup

Run commands from the repository root. Use Python 3.12 for the NumPy experiments
and Python 3.10 with a CUDA-capable PyTorch installation for the GPU experiment.
Install NumPy dependencies in the active environment:

```bash
python -m pip install -r requirements-part1.txt
python -m pip install huggingface_hub
```

For PyTorch, reuse the existing `ee508` Conda environment or install
`requirements-part3.txt` in a separate Python 3.10 environment.

## Prepare data

Each checkout prepares its own local data. Use an empty output directory:

```bash
python scripts/prepare_data.py --download --raw raw --out data --seed 0 --test-frac 0.2
python scripts/verify_data.py --data-dir data
```

See [DATA.md](DATA.md) for dataset and preprocessing conventions.

## Run experiments

Part I: regression, binary classification, and multiclass MLP examples:

```bash
python src/part1_mlp.py --output-dir results/part1
```

Part II: full NumPy CNN experiment:

```bash
python src/part2_numpy_cnn.py --data-dir data --epochs 10 --lr 0.001 --seed 3
```

Part III: full PyTorch CUDA experiment:

```bash
conda activate ee508
python src/part3_torch.py --data-dir data --device cuda --epochs 10 --lr 0.001 --seed 3
```

## Check results

```bash
python scripts/check_results.py
python scripts/compare.py
```

Formal results, loss plots, and logs are saved under `results/`.
See the [comparison](results/comparison.md) for the completed experiments.
Raw downloads, dataset images, temporary outputs, and PyTorch checkpoints stay
local; selected formal results are tracked in Git.
