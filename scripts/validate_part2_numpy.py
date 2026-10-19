"""Step 3: train and evaluate the NumPy CNN on a small real cat/dog subset.

Run: python scripts/validate_part2_numpy.py --data-dir data
This records a small validation run; formal experiment timing comes later.
"""

import argparse
import csv
import json
import logging
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_utils import load_images, read_data_info, select_samples  # noqa: E402
from part2_numpy_cnn import NumPyCNN, train  # noqa: E402


def validate(args):
    logger = logging.getLogger("part2_validation")
    stats, records = read_data_info(args.data_dir)
    train_records = select_samples(records, "train", args.train_per_class, args.data_seed)
    test_records = select_samples(records, "test", args.test_per_class, args.data_seed)
    train_names = {r["filename"] for r in train_records}
    test_names = {r["filename"] for r in test_records}
    if train_names & test_names:
        raise ValueError("Selected train/test filenames overlap")
    images, labels = load_images(args.data_dir, train_records)
    test_images, test_labels = load_images(args.data_dir, test_records)
    model = NumPyCNN(seed=args.seed)
    original_filter, original_weights = model.conv_filter.copy(), model.W_fc.copy()
    original_bias = model.b_fc.copy()
    model.forward(images)
    initial_loss = model.loss(labels)
    shapes = {"input": list(images.shape), "convolution": list(model.conv_out.shape),
              "pooling": list(model.pooled.shape), "flatten": list(model.flat.shape),
              "output": list(model.y_pred.shape)}
    logger.info("Real cat/dog validation: train=%d, test=%d, split=%s",
                len(images), len(test_images), stats["split_fingerprint"])
    logger.info("Seeds: model/shuffle=%d, sample selection=%d", args.seed, args.data_seed)
    logger.info("Layer shapes: %s", shapes)
    logger.info("Initial full-training-subset MSE: %.8f", initial_loss)
    history = train(model, images, labels, epochs=args.epochs, lr=args.lr,
                    seed=args.seed, log=logger.info)
    scores = model.forward(images)[:, 0].copy()
    final_loss = model.loss(labels)
    train_accuracy = float(np.mean((scores > 0.5) == labels))
    test_scores = model.forward(test_images)[:, 0].copy()
    test_loss = model.loss(test_labels)
    test_accuracy = float(np.mean((test_scores > 0.5) == test_labels))
    checks = dict(
        layer_shapes=(shapes["convolution"] == [len(images), 62, 62]
                      and shapes["pooling"] == [len(images), 31, 31]
                      and shapes["flatten"] == [len(images), 961]
                      and shapes["output"] == [len(images), 1]),
        finite_losses=bool(np.isfinite([initial_loss, final_loss, test_loss]
                                      + [r["loss"] for r in history]).all()),
        training_loss_decreased=final_loss < initial_loss,
        fc_weights_updated=not np.array_equal(original_weights, model.W_fc),
        fc_bias_updated=not np.array_equal(original_bias, model.b_fc),
        convolution_filter_fixed=np.array_equal(original_filter, model.conv_filter),
        no_train_test_overlap=not bool(train_names & test_names))
    summary = dict(
        run_type="small_sample_validation", dataset=stats["dataset"],
        data_split_fingerprint=stats["split_fingerprint"],
        n_train=len(images), n_test=len(test_images), epochs=args.epochs,
        lr=args.lr, seed=args.seed, data_seed=args.data_seed,
        initial_loss=initial_loss, final_loss=final_loss, test_loss=test_loss,
        train_accuracy=train_accuracy, test_accuracy=test_accuracy,
        shapes=shapes, checks=checks, passed=all(checks.values()),
        epoch_loss_definition="Mean per-image MSE immediately before each SGD update",
        epochs_log=history, train_filenames=sorted(train_names), test_filenames=sorted(test_names))
    (args.output_dir / "validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    with (args.output_dir / "loss.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    with (args.output_dir / "test_predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["filename", "label", "score", "prediction"])
        writer.writerows((r["filename"], int(y), float(s), int(s > 0.5))
                         for r, y, s in zip(test_records, test_labels, test_scores))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["epoch"] for r in history], [r["loss"] for r in history])
    ax.set(xlabel="Epoch", ylabel="Mean online training MSE",
           title="NumPy CNN: small cat/dog validation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.output_dir / "loss.png", dpi=150)
    plt.close(fig)
    logger.info("Final full-training-subset MSE: %.8f", final_loss)
    logger.info("Train accuracy: %.2f%%; held-out subset accuracy: %.2f%%",
                100 * train_accuracy, 100 * test_accuracy)
    logger.info("Validation checks: %s", checks)
    logger.info("Results saved to %s", args.output_dir)
    return summary["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--train-per-class", type=int, default=16)
    parser.add_argument("--test-per-class", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/part2_validation"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True,
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.output_dir / "run.log", mode="w")])
    try:
        return 0 if validate(args) else 1
    except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
        logging.error("Validation failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
