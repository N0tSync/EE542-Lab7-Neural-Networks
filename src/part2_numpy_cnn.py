#!/usr/bin/env python3
"""EE 542 Lab 7, Part II: NumPy CNN forward pass and FC-only updates.

Architecture: 64x64x3 -> valid convolution -> ReLU -> 2x2 max pool
              -> 961 features -> fully connected -> ReLU.
The single 3x3x3 convolution filter is fixed; only W_fc and b_fc are trained.
Loss is ordinary MSE (without Part I's factor of 0.5).
An output ReLU has zero gradient at nonpositive logits; some initializations
can therefore produce all-zero scores and stall learning.

Run an experiment: python src/part2_numpy_cnn.py --data-dir data --epochs 10
Run a synthetic forward/update demonstration: python src/part2_numpy_cnn.py --demo
Real-image validation: python scripts/validate_part2_numpy.py
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import platform
import sys
import time

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    """Use derivative 0 at the ReLU kink x = 0, as in the handout."""
    return (x > 0).astype(float)


def _mse_error(y_true, y_pred):
    targets = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    if targets.shape != predictions.shape or predictions.size == 0:
        raise ValueError("MSE targets and predictions must have the same nonempty shape")
    return predictions - targets


def mse_loss(y_true, y_pred):
    diff = _mse_error(y_true, y_pred)
    return float(np.mean(diff * diff))


def mse_grad(y_true, y_pred):
    diff = _mse_error(y_true, y_pred)
    return 2.0 * diff / diff.size


def conv2d(images, kernel):
    """Valid, stride-1 cross-correlation, summing all RGB channels.

    Input: (H, W, C) or (N, H, W, C); kernel: (KH, KW, C).
    Output: (H-KH+1, W-KW+1) or (N, H-KH+1, W-KW+1).
    A window view avoids copying every image patch into a separate array.
    """
    images = np.asarray(images, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if images.ndim not in (3, 4) or kernel.ndim != 3:
        raise ValueError("Expected HWC/NHWC images and an HWC kernel")
    kh, kw, channels = kernel.shape
    if (min(kh, kw, channels) < 1 or images.shape[-1] != channels
            or images.shape[-3] < kh or images.shape[-2] < kw):
        raise ValueError("Kernel dimensions must fit the image and match its channels")
    windows = sliding_window_view(images, (kh, kw), axis=(-3, -2))
    # Window layout: (..., output_height, output_width, channels, kh, kw).
    return np.einsum("...hwcij,ijc->...hw", windows, kernel, optimize=False)


def maxpool(images, size=2):
    """Non-overlapping max pooling; discard incomplete bottom/right blocks."""
    images = np.asarray(images, dtype=float)
    if images.ndim not in (2, 3):
        raise ValueError("Expected HW/NHW feature maps")
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("Pool size must be a positive integer")
    h, w = images.shape[-2:]
    new_h, new_w = h // size, w // size
    if min(new_h, new_w) < 1:
        raise ValueError("Pool size must fit the feature map")
    cropped = images[..., :new_h * size, :new_w * size]
    blocks = cropped.reshape(*images.shape[:-2], new_h, size, new_w, size)
    return blocks.max(axis=(-3, -1))


class NumPyCNN:
    """One fixed convolution filter and one trainable output neuron.

    forward accepts (64, 64, 3) or (N, 64, 64, 3), returning (N, 1).
    Layer values are cached for inspection and for backward's FC gradients.
    Output is a nonnegative score; later classification uses score > 0.5.
    """

    input_shape = (64, 64, 3)
    n_features = 31 * 31

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.conv_filter = rng.normal(0.0, 0.1, size=(3, 3, 3))
        self.W_fc = rng.normal(0.0, 0.01, size=(self.n_features, 1))
        self.b_fc = np.zeros((1,))
        self.y_pred = None

    def forward(self, images):
        images = np.asarray(images, dtype=float)
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        if images.ndim != 4 or images.shape[1:] != self.input_shape or len(images) == 0:
            raise ValueError("Expected a 64x64x3 image or a nonempty NHWC batch")
        self.conv_out = conv2d(images, self.conv_filter)
        self.relu_out = relu(self.conv_out)
        self.pooled = maxpool(self.relu_out)
        self.flat = self.pooled.reshape(len(images), self.n_features)
        self.z_fc = self.flat @ self.W_fc + self.b_fc
        self.y_pred = relu(self.z_fc)
        return self.y_pred

    def _targets(self, y):
        if self.y_pred is None:
            raise RuntimeError("Call forward before loss or backward")
        targets = np.asarray(y, dtype=float)
        if targets.ndim == 0:
            targets = targets.reshape(1, 1)
        elif targets.ndim == 1:
            targets = targets[:, np.newaxis]
        if targets.shape != self.y_pred.shape:
            raise ValueError("Expected one target per image, with shape (N,) or (N, 1)")
        return targets

    def loss(self, y):
        return mse_loss(self._targets(y), self.y_pred)

    def backward(self, y):
        grad_z = mse_grad(self._targets(y), self.y_pred) * relu_derivative(self.z_fc)
        dW_fc = self.flat.T @ grad_z
        db_fc = grad_z.sum(axis=0)
        # No convolution gradients: the feature extractor remains fixed.
        return dW_fc, db_fc

    def step(self, grads, lr=1e-3):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be finite and positive")
        dW_fc, db_fc = grads
        if dW_fc.shape != self.W_fc.shape or db_fc.shape != self.b_fc.shape:
            raise ValueError("Gradient shapes must match W_fc and b_fc")
        self.W_fc -= lr * dW_fc
        self.b_fc -= lr * db_fc


def _dataset_arrays(images, labels):
    """Validate without expanding a whole uint8 dataset to floating point."""
    images = np.asarray(images)
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if (images.ndim != 4 or images.shape[1:] != (64, 64, 3)
            or len(images) == 0 or labels.shape != (len(images),)):
        raise ValueError("Expected nonempty NHWC images and one label per image")
    if images.dtype != np.uint8 and (images.dtype.kind != "f" or not np.isfinite(images).all()):
        raise ValueError("Images must be uint8 pixels or finite normalized floating-point values")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 or 1")
    return images, labels


def _normalized(images):
    result = np.asarray(images, dtype=float)
    return result / 255.0 if images.dtype == np.uint8 else result


def train(model, images, labels, epochs=20, lr=1e-3, seed=0, verbose=True, log=None,
          timed=False):
    """Shuffled per-image SGD, matching the handout's training scheme.

    Each epoch's loss and accuracy are online averages, measured immediately
    before each sample's update. Only the FC parameters are updated. uint8
    images are normalized one at a time. timed=True returns (history, seconds)
    and adds per-epoch timing; otherwise the reproducible history is returned.
    """
    images, labels = _dataset_arrays(images, labels)
    if not isinstance(epochs, (int, np.integer)) or epochs < 1:
        raise ValueError("Epochs must be a positive integer")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    rng = np.random.default_rng(seed)
    positive_count = int(np.count_nonzero(labels == 1))
    history = []
    report = log if log is not None else print
    training_started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        total_loss, correct, active_positive = 0.0, 0, 0
        for index in rng.permutation(len(images)):
            score = float(model.forward(_normalized(images[index]))[0, 0])
            loss = model.loss(labels[index])
            if not np.isfinite(loss):
                raise FloatingPointError(f"Non-finite MSE in epoch {epoch}, image {index}")
            total_loss += loss
            correct += int((score > 0.5) == labels[index])
            active_positive += int(labels[index] == 1 and model.z_fc[0, 0] > 0)
            model.step(model.backward(labels[index]), lr=lr)
        if not np.isfinite(model.W_fc).all() or not np.isfinite(model.b_fc).all():
            raise FloatingPointError(f"Non-finite FC parameters after epoch {epoch}")
        if positive_count and active_positive == 0:
            raise RuntimeError(
                f"All positive-class outputs were inactive in epoch {epoch}; "
                "the output ReLU blocks their gradients. Try a different model seed."
            )
        record = dict(epoch=epoch, loss=total_loss / len(images),
                      train_accuracy=correct / len(images),
                      active_positive_fraction=(active_positive / positive_count
                                                if positive_count else None))
        if timed:
            record["seconds"] = time.perf_counter() - epoch_started
            record["images_per_sec"] = len(images) / record["seconds"]
        history.append(record)
        if verbose:
            message = (f"epoch {epoch:3d}/{epochs} online_mse={record['loss']:.6f} "
                       f"online_accuracy={record['train_accuracy']:.2%}")
            if timed:
                message += f" seconds={record['seconds']:.3f}"
            report(message)
    training_seconds = time.perf_counter() - training_started
    return (history, training_seconds) if timed else history


def evaluate(model, images, labels, block_size=1000, batch_size=64):
    """Return metrics and ordered scores; decode images before calling this.

    Logical test blocks are subdivided into small forward batches. Timing
    includes normalization, inference and metrics; forward time is also recorded.
    """
    images, labels = _dataset_arrays(images, labels)
    if block_size < 1 or batch_size < 1:
        raise ValueError("Test block and inference batch sizes must be positive")
    blocks = []
    scores = np.empty(len(images), dtype=float)
    total_correct = total_dogs = 0
    total_error = total_compute = 0.0
    confusion = np.zeros((2, 2), dtype=np.int64)
    test_started = time.perf_counter()
    for start in range(0, len(images), block_size):
        end = min(start + block_size, len(images))
        block_started = time.perf_counter()
        correct = dogs = 0
        error = compute = 0.0
        for offset in range(start, end, batch_size):
            stop = min(offset + batch_size, end)
            batch = _normalized(images[offset:stop])
            targets = labels[offset:stop]
            compute_started = time.perf_counter()
            output = np.asarray(model.forward(batch))
            compute += time.perf_counter() - compute_started
            if output.shape != (stop - offset, 1) or not np.isfinite(output).all():
                raise FloatingPointError("Invalid or non-finite test predictions")
            values = output[:, 0]
            scores[offset:stop] = values
            predictions = (values > 0.5).astype(np.int64)
            correct += int(np.count_nonzero(predictions == targets))
            dogs += int(predictions.sum())
            error += float(np.sum((values - targets) ** 2))
            np.add.at(confusion, (targets.astype(np.int64), predictions), 1)
        seconds = time.perf_counter() - block_started
        count = end - start
        blocks.append(dict(block=len(blocks) + 1, start_index=start, n_images=count,
                           correct=correct, accuracy=correct / count, mse=error / count,
                           seconds=seconds, compute_seconds=compute,
                           images_per_sec=count / seconds))
        total_correct += correct
        total_dogs += dogs
        total_error += error
        total_compute += compute
    test_seconds = time.perf_counter() - test_started
    metrics = dict(test_seconds=test_seconds, test_compute_seconds=total_compute,
                   test_images_per_sec=len(images) / test_seconds,
                   test_correct=total_correct, test_accuracy=total_correct / len(images),
                   test_mse=total_error / len(images), test_predicted_dogs=total_dogs,
                   test_predicted_cats=len(images) - total_dogs,
                   confusion_matrix=confusion.tolist(), test_blocks=blocks)
    return metrics, scores


def cpu_model_name():
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def run_experiment(args):
    from data_utils import load_images, manifest_fingerprint, read_data_info, select_samples

    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.stem.removesuffix("_results")
    log_path = output.with_name(f"{prefix}_run.log")
    logger = logging.getLogger("part2_numpy_experiment")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(log_path, mode="w")]
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
    try:
        logger.info("NumPy CNN on CPU: %s", cpu_model_name())
        load_started = time.perf_counter()
        stats, records = read_data_info(args.data_dir)
        train_records = select_samples(records, "train", args.max_train_per_class, args.data_seed)
        test_records = select_samples(records, "test", args.max_test_per_class, args.data_seed)
        images, labels = load_images(args.data_dir, train_records, uint8=True)
        test_images, test_labels = load_images(args.data_dir, test_records, uint8=True)
        load_seconds = time.perf_counter() - load_started
        fingerprint = manifest_fingerprint(train_records + test_records)
        logger.info("Loaded %d train / %d test images in %.3fs; split=%s",
                    len(images), len(test_images), load_seconds, fingerprint)
        logger.info("Image storage: %.2f MiB, uint8; normalization is timed",
                    (images.nbytes + test_images.nbytes) / 2**20)
        logger.info("epochs=%d lr=%g seed=%d data_seed=%d test_block_size=%d inference_batch_size=%d",
                    args.epochs, args.lr, args.seed, args.data_seed,
                    args.test_block_size, args.inference_batch_size)
        model = NumPyCNN(seed=args.seed)
        original_filter = model.conv_filter.copy()
        history, training_seconds = train(
            model, images, labels, epochs=args.epochs, lr=args.lr, seed=args.seed,
            log=logger.info, timed=True)
        evaluation, scores = evaluate(model, test_images, test_labels,
                                      block_size=args.test_block_size,
                                      batch_size=args.inference_batch_size)
        if not np.array_equal(original_filter, model.conv_filter):
            raise RuntimeError("The fixed convolution filter changed during the experiment")
        for block in evaluation["test_blocks"]:
            logger.info("Test block %d: %d images, accuracy=%.2f%%, seconds=%.4f",
                        block["block"], block["n_images"], 100 * block["accuracy"], block["seconds"])
        result = dict(
            model="NumPyCNN", device="cpu", device_name=cpu_model_name(),
            dataset=stats["dataset"], data_dir=str(args.data_dir.resolve()),
            data_split_fingerprint=fingerprint,
            source_data_split_fingerprint=stats["split_fingerprint"],
            run_type=("full_dataset" if len(train_records) + len(test_records) == len(records)
                      else "subset_validation"),
            n_train=len(images), n_test=len(test_images), epochs=args.epochs,
            lr=args.lr, seed=args.seed, data_seed=args.data_seed, image_size=[64, 64],
            batch_size=1, test_block_size=args.test_block_size,
            inference_batch_size=args.inference_batch_size,
            max_train_per_class=args.max_train_per_class,
            max_test_per_class=args.max_test_per_class,
            train_seconds=training_seconds,
            train_images_per_sec=len(images) * args.epochs / training_seconds,
            load_seconds=load_seconds, preloaded_image_bytes=images.nbytes + test_images.nbytes,
            epochs_log=history, **evaluation,
            train_accuracy_definition="Online accuracy before each SGD update",
            loss_definition="Ordinary MSE; epoch values are online averages before SGD updates",
            timing_definition="Preload excluded; normalization, model computation and metrics included",
            architecture="64x64x3 -> fixed conv3 -> ReLU -> pool2 -> 961 -> FC1 -> ReLU",
            trainable_parameters=962, frozen_parameters=27,
            convolution_filter_fixed=True, numpy_version=np.__version__,
            python_version=platform.python_version(), platform=platform.platform(),
            machine=f"{platform.platform()} / Python {platform.python_version()}",
            config=dict(epochs=args.epochs, lr=args.lr, seed=args.seed, data_seed=args.data_seed,
                        batch_size=1, test_block_size=args.test_block_size,
                        inference_batch_size=args.inference_batch_size,
                        max_train_per_class=args.max_train_per_class,
                        max_test_per_class=args.max_test_per_class,
                        trained_layers="fully connected only", convolution="NumPy vectorized einsum"))
        checkpoint = output.with_name(f"{prefix}_model.npz")
        np.savez(checkpoint, conv_filter=model.conv_filter, W_fc=model.W_fc, b_fc=model.b_fc)
        result["checkpoint"] = str(checkpoint)
        epochs_path = output.with_name(f"{prefix}_epochs.csv")
        with epochs_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        with output.with_name(f"{prefix}_test_predictions.csv").open(
                "w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["filename", "label", "score", "prediction"])
            writer.writerows((row["filename"], int(label), float(score), int(score > 0.5))
                             for row, label, score in zip(test_records, test_labels, scores))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([row["epoch"] for row in history], [row["loss"] for row in history])
        ax.set(xlabel="Epoch", ylabel="Mean online training MSE", title="NumPy CNN training")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(output.with_name(f"{prefix}_loss.png"), dpi=150)
        plt.close(fig)
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        logger.info("Training: %.3fs, %.1f images/s", training_seconds, result["train_images_per_sec"])
        logger.info("Testing: %.3fs, %.1f images/s, accuracy=%.2f%%",
                    result["test_seconds"], result["test_images_per_sec"], 100 * result["test_accuracy"])
        logger.info("Test predictions: cat=%d dog=%d",
                    result["test_predicted_cats"], result["test_predicted_dogs"])
        if not result["test_predicted_cats"] or not result["test_predicted_dogs"]:
            logger.warning("The model predicted only one class on the test set")
        logger.info("Results saved to %s", output)
        return result
    except Exception:
        logger.exception("Experiment failed")
        raise
    finally:
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)


def demo():
    """Show one forward pass and one gradient update on synthetic images."""
    images = np.random.default_rng(1).uniform(0.0, 1.0, size=(2, 64, 64, 3))
    labels = np.array([0, 1])
    # Seed 3 gives active output ReLUs for this demonstration's synthetic data.
    model = NumPyCNN(seed=3)
    original_filter = model.conv_filter.copy()
    model.forward(images)
    print("Synthetic demonstration: data seed=1, model seed=3")
    print("Synthetic images:", images.shape)
    print("Convolution:", model.conv_out.shape)
    print("Pooling:", model.pooled.shape)
    print("Flatten:", model.flat.shape)
    print("W_fc:", model.W_fc.shape, "b_fc:", model.b_fc.shape)
    print("Output scores:", model.y_pred.ravel())
    print(f"MSE before update: {model.loss(labels):.8f}")
    model.step(model.backward(labels), lr=1e-3)
    model.forward(images)
    print(f"MSE after update:  {model.loss(labels):.8f}")
    print("Convolution filter unchanged:", np.array_equal(original_filter, model.conv_filter))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", action="store_true", help="run only the synthetic update demonstration")
    parser.add_argument("--data-dir", "--data", type=Path, default=Path("data"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--data-seed", type=int, help="subset seed; defaults to the model seed")
    parser.add_argument("--test-block-size", type=int, default=1000)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--max-train-per-class", type=int, default=0, help="0 uses all training images")
    parser.add_argument("--max-test-per-class", type=int, default=0, help="0 uses all test images")
    outputs = parser.add_mutually_exclusive_group()
    outputs.add_argument("--output-dir", "--out", type=Path,
                         help="directory for results and companion artifacts")
    outputs.add_argument("--output", type=Path, help="explicit result JSON path")
    args = parser.parse_args(argv)
    if min(args.epochs, args.test_block_size, args.inference_batch_size) < 1:
        parser.error("Epochs, test block size and inference batch size must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("Learning rate must be finite and positive")
    if args.data_seed is None:
        args.data_seed = args.seed
    if min(args.seed, args.data_seed, args.max_train_per_class, args.max_test_per_class) < 0:
        parser.error("Seeds and subset counts must be nonnegative")
    if args.output is None:
        subset = bool(args.max_train_per_class or args.max_test_per_class)
        folder = args.output_dir or Path("results/part2_smoke" if subset else "results")
        args.output = folder / "part2_numpy_results.json"
    if args.output.suffix.lower() != ".json":
        parser.error("The output file must have a .json extension")
    return args


def main():
    args = parse_args()
    if args.demo:
        demo()
        return 0
    try:
        run_experiment(args)
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
