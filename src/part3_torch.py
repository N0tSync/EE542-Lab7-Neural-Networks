#!/usr/bin/env python3
"""Lab 7 Part III: train and evaluate the handout's ReLU/MSE CNN.

Run from the repository root with the project's virtual environment activated:
    python src/part3_torch.py --data-dir data --device cuda
Images are decoded into CPU tensors before timing, following DATA.md. Timed
training includes batching, CPU-to-device transfers, forward/backward and Adam.
Timed testing includes batching, transfers, inference and metrics; forward-only
time is also recorded separately. No test images are used for optimization.
"""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


class CatDogCNN(nn.Module):
    """64x64 RGB -> conv16/pool -> conv32/pool -> FC64 -> ReLU scalar."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 1)
        # Start near the label midpoint to reduce the risk of a dead output ReLU.
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.001)
        nn.init.constant_(self.fc2.bias, 0.5)

    def forward(self, images):
        x = F.max_pool2d(F.relu(self.conv1(images)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.relu(self.fc1(torch.flatten(x, 1)))
        return F.relu(self.fc2(x))


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def select_device(requested):
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Check GPU access/driver, or use --device cpu.")
    return torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                        else "cpu" if requested == "auto" else requested)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def split_fingerprint(pairs):
    digest = hashlib.sha256()
    for name, split in sorted(pairs):
        digest.update(f"{name},{split}\n".encode())
    return digest.hexdigest()[:12]


def cpu_model_name():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def load_image_datasets(data_dir):
    """Check the actual files against the preparation manifest before training."""
    if not all((data_dir / split / name).is_dir()
               for split in ("train", "test") for name in ("cat", "dog")):
        raise ValueError(f"Missing {data_dir}/train|test/cat|dog images. Run prepare_data.py first.")
    transform = transforms.Compose([
        transforms.Resize((64, 64), interpolation=InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])
    train = datasets.ImageFolder(data_dir / "train", transform=transform)
    test = datasets.ImageFolder(data_dir / "test", transform=transform)
    if train.class_to_idx != {"cat": 0, "dog": 1} or test.class_to_idx != train.class_to_idx:
        raise ValueError("Expected exactly cat=0 and dog=1 in both splits.")
    with (data_dir / "stats.json").open() as stream:
        stats = json.load(stream)
    with (data_dir / "split_manifest.csv").open(newline="") as stream:
        manifest = list(csv.DictReader(stream))
    expected = {f"{r['split']}/{r['class']}/{r['filename']}": int(r["label"])
                for r in manifest}
    actual = {f"{split}/{Path(path).parent.name}/{Path(path).name}": label
              for split, dataset in (("train", train), ("test", test))
              for path, label in dataset.samples}
    if len(expected) != len(manifest) or actual != expected:
        raise ValueError("Image files/labels differ from split_manifest.csv; rebuild or restore the dataset.")
    names = [{Path(path).name for path, _ in dataset.samples} for dataset in (train, test)]
    if names[0] & names[1]:
        raise ValueError("Training and test filenames overlap.")
    pairs = [(Path(path).name, split) for split, dataset in (("train", train), ("test", test))
             for path, _ in dataset.samples]
    fingerprint = split_fingerprint(pairs)
    counts = {f"{split}/{name}": dataset.targets.count(label)
              for split, dataset in (("train", train), ("test", test))
              for name, label in train.class_to_idx.items()}
    if (fingerprint != stats["split_fingerprint"] or counts != stats["counts"]
            or len(train) != stats["train_total"] or len(test) != stats["test_total"]):
        raise ValueError("Actual data counts/fingerprint differ from stats.json.")
    return train, test, stats


def select_subset(dataset, max_per_class, seed):
    if not max_per_class:
        return dataset, list(range(len(dataset)))
    rng = np.random.default_rng(seed)
    indices = []
    for label in (0, 1):
        candidates = np.flatnonzero(np.asarray(dataset.targets) == label)
        indices.extend(rng.permutation(candidates)[:max_per_class].tolist())
    indices.sort()
    return Subset(dataset, indices), indices


def preload(dataset, batch_size, num_workers):
    """Decode once; keep all 64x64 float32 images in CPU memory, not GPU memory."""
    images = torch.empty((len(dataset), 3, 64, 64), dtype=torch.float32)
    labels = torch.empty(len(dataset), dtype=torch.int64)
    offset = 0
    for batch, targets in DataLoader(dataset, batch_size=batch_size, num_workers=num_workers):
        count = len(targets)
        images[offset:offset + count].copy_(batch)
        labels[offset:offset + count].copy_(targets)
        offset += count
    return TensorDataset(images, labels)


def train_model(model, dataset, device, epochs, lr, batch_size, seed):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        pin_memory=device.type == "cuda")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    logs = []
    synchronize(device)
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.int64)
        active = torch.zeros((), device=device, dtype=torch.int64)
        synchronize(device)
        started = time.perf_counter()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True).view(-1, 1)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach() * len(images)
            correct += ((outputs.detach() > 0.5) == labels.bool()).sum()
            active += (outputs.detach() > 0).sum()
        synchronize(device)
        seconds = time.perf_counter() - started
        mean_loss = loss_sum.item() / len(dataset)
        if not math.isfinite(mean_loss):
            raise RuntimeError(f"Non-finite training loss at epoch {epoch}.")
        if active.item() == 0:
            raise RuntimeError("All output ReLUs were inactive for an epoch; training has stalled.")
        entry = dict(epoch=epoch, loss=mean_loss, seconds=seconds,
                     train_accuracy=correct.item() / len(dataset),
                     output_active_fraction=active.item() / len(dataset))
        logs.append(entry)
        print(f"Epoch {epoch:2d}/{epochs} | MSE {mean_loss:.6f} | "
              f"train accuracy {entry['train_accuracy']:.2%} | {seconds:.2f}s", flush=True)
    synchronize(device)
    train_seconds = time.perf_counter() - training_started
    return logs, optimizer, train_seconds


@torch.no_grad()
def evaluate(model, dataset, device, batch_size, block_size):
    model.eval()
    blocks = []
    total_correct = total_dogs = 0
    total_compute_seconds = 0.0
    synchronize(device)
    test_started = time.perf_counter()
    for start in range(0, len(dataset), block_size):
        end = min(start + block_size, len(dataset))
        loader = DataLoader(Subset(dataset, range(start, end)), batch_size=batch_size,
                            pin_memory=device.type == "cuda")
        correct = dogs = 0
        squared_error = 0.0
        compute_seconds = 0.0
        synchronize(device)
        block_started = time.perf_counter()
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32, non_blocking=True).view(-1, 1)
            synchronize(device)  # Finish transfers before starting forward-only timing.
            compute_started = time.perf_counter()
            outputs = model(images)
            synchronize(device)
            compute_seconds += time.perf_counter() - compute_started
            if not torch.isfinite(outputs).all().item():
                raise RuntimeError("Non-finite test prediction.")
            predictions = outputs > 0.5
            correct += (predictions == labels.bool()).sum().item()
            dogs += predictions.sum().item()
            squared_error += F.mse_loss(outputs, labels, reduction="sum").item()
        synchronize(device)
        seconds = time.perf_counter() - block_started
        count = end - start
        blocks.append(dict(block=len(blocks) + 1, start_index=start, n_images=count,
                           correct=correct, accuracy=correct / count, mse=squared_error / count,
                           seconds=seconds, images_per_sec=count / seconds,
                           compute_seconds=compute_seconds, predicted_dogs=dogs))
        total_correct += correct
        total_dogs += dogs
        total_compute_seconds += compute_seconds
    synchronize(device)
    seconds = time.perf_counter() - test_started
    return dict(test_accuracy=total_correct / len(dataset), test_correct=total_correct,
                test_mse=sum(b["mse"] * b["n_images"] for b in blocks) / len(dataset),
                test_seconds=seconds, test_images_per_sec=len(dataset) / seconds,
                test_compute_seconds=total_compute_seconds, test_blocks=blocks,
                test_predicted_cats=len(dataset) - total_dogs, test_predicted_dogs=total_dogs)


def save_artifacts(result, model, optimizer, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = output.parent / output.stem
    artifacts.mkdir(parents=True, exist_ok=True)
    checkpoint = artifacts / "model.pt"
    torch.save(dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
                    config=result["config"], data_split_fingerprint=result["data_split_fingerprint"],
                    class_to_idx={"cat": 0, "dog": 1}), checkpoint)
    with (artifacts / "epochs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result["epochs_log"][0]))
        writer.writeheader()
        writer.writerows(result["epochs_log"])
    # Import plotting outside the benchmark, using a writable local config directory.
    os.environ.setdefault("MPLCONFIGDIR", str(Path(".venv/matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["epoch"] for row in result["epochs_log"]],
            [row["loss"] for row in result["epochs_log"]], marker="o")
    ax.set(xlabel="Epoch", ylabel="Training MSE", title="PyTorch CatDogCNN training loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(artifacts / "loss.png", dpi=150)
    plt.close(fig)
    result["checkpoint"] = str(checkpoint)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


def run(args):
    seed_everything(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = select_device(args.device)
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print(f"Using device: {device} ({device_name})", flush=True)
    train_images, test_images, stats = load_image_datasets(args.data_dir)
    train_subset, train_indices = select_subset(train_images, args.max_train_per_class, args.seed)
    test_subset, test_indices = select_subset(test_images, args.max_test_per_class, args.seed)
    pairs = [(Path(dataset.samples[index][0]).name, split)
             for split, dataset, indices in (("train", train_images, train_indices),
                                              ("test", test_images, test_indices)) for index in indices]
    fingerprint = split_fingerprint(pairs)
    print(f"Preloading RGB 64x64 images: train={len(train_subset)}, test={len(test_subset)}, "
          f"split={fingerprint}", flush=True)
    started = time.perf_counter()
    train = preload(train_subset, args.batch_size, args.num_workers)
    test = preload(test_subset, args.batch_size, args.num_workers)
    load_seconds = time.perf_counter() - started
    model = CatDogCNN().to(device)
    print(f"Batch shape: {tuple(train.tensors[0][:args.batch_size].shape)}; "
          f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    # Warm up kernels and device allocations, outside the timed experiment.
    with torch.no_grad():
        model(train.tensors[0][:args.batch_size].to(device))
    synchronize(device)
    logs, optimizer, train_seconds = train_model(
        model, train, device, args.epochs, args.lr, args.batch_size, args.seed)
    evaluation = evaluate(model, test, device, args.batch_size, args.test_block_size)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    result = dict(model="CatDogCNN", device=device.type, device_name=device_name,
                  data_split_fingerprint=fingerprint, source_data_split_fingerprint=stats["split_fingerprint"],
                  n_train=len(train), n_test=len(test), epochs=args.epochs, seed=args.seed,
                  image_size=[64, 64], batch_size=args.batch_size, learning_rate=args.lr,
                  optimizer="Adam", loss="MSELoss", output_activation="ReLU", decision_threshold=0.5,
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  architecture="3->Conv16(3x3,pad1)->ReLU->Pool2->Conv32(3x3,pad1)->ReLU->Pool2"
                               "->Flatten8192->FC64->ReLU->FC1->ReLU",
                  output_initialization="weight normal(0,0.001), bias 0.5",
                  preprocessing="RGB, resize 64x64 bilinear, float32 / 255",
                  load_seconds=load_seconds, train_seconds=train_seconds,
                  train_images_per_sec=args.epochs * len(train) / train_seconds,
                  epochs_log=logs, **evaluation, config=config,
                  timing_notes={"train": "Wall time across all epochs: batching, H2D, forward, backward, Adam, metrics and epoch logging; excludes disk decoding, optimizer setup and warmup.",
                                "test": "Batching, H2D, inference, metrics and synchronization; excludes disk decoding and warmup.",
                                "test_compute": "Forward-only wall time with synchronization; excludes transfers and metrics."},
                  environment=dict(python=platform.python_version(), torch=torch.__version__,
                                   torchvision=torchvision.__version__, numpy=np.__version__,
                                   cuda_runtime=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                                   cpu_model=cpu_model_name(), cpu_threads=args.cpu_threads,
                                   platform=platform.platform()))
    output = args.output or Path(f"results/part3_torch_{device.type}.json")
    save_artifacts(result, model, optimizer, output)
    for block in evaluation["test_blocks"]:
        print(f"Test block {block['block']}: {block['n_images']} images, "
              f"accuracy {block['accuracy']:.2%}, {block['seconds']:.3f}s", flush=True)
    print(f"Test accuracy: {evaluation['test_accuracy']:.2%}; "
          f"test time: {evaluation['test_seconds']:.3f}s; "
          f"throughput: {evaluation['test_images_per_sec']:.1f} images/s", flush=True)
    if not evaluation["test_predicted_cats"] or not evaluation["test_predicted_dogs"]:
        print("WARNING: the model predicted only one class on the test set.", flush=True)
    print(f"Results: {output.resolve()}", flush=True)
    return result


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-block-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0, help="Workers for initial image decoding only.")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--max-train-per-class", type=int, default=0, help="0 uses all training images.")
    parser.add_argument("--max-test-per-class", type=int, default=0, help="0 uses all test images.")
    parser.add_argument("--output", type=Path, help="Default: results/part3_torch_<device>.json")
    args = parser.parse_args()
    if any(value <= 0 for value in (args.epochs, args.batch_size, args.test_block_size, args.cpu_threads)):
        parser.error("Epochs, batch size, test block size and CPU threads must be positive.")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Learning rate must be finite and positive.")
    if min(args.num_workers, args.max_train_per_class, args.max_test_per_class, args.seed) < 0:
        parser.error("Worker count, subset sizes and seed must be nonnegative.")
    return args


if __name__ == "__main__":
    try:
        run(parse_args())
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
