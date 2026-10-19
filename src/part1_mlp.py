#!/usr/bin/env python3
"""EE 542 Lab 7, Part I: the three MLP examples from the handout.

Install: python -m pip install numpy matplotlib
Run:     python src/part1_mlp.py --output-dir results/part1
One run: python src/part1_mlp.py --example regression --output-dir results/part1

The handout's MLP, synthetic data, seeds, and training settings are preserved.
Added: a command-line entry point, CSV logs, and plots saved without a display.
Classification accuracy below is TRAINING accuracy, as in Part I's examples.
The regression loss is 0.5 * mean(sum(squared error over output dimensions)).
Logged loss excludes the weight-decay penalty, matching the handout.
"""

import argparse
import csv
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def relu(x):
    return np.maximum(0.0, x)


def d_relu(x):
    return (x > 0).astype(x.dtype)


def softmax(logits):
    logits = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(logits)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y, num_classes):
    result = np.zeros((y.size, num_classes))
    result[np.arange(y.size), y] = 1.0
    return result


class MLP2:
    """Two trainable layers: input -> ReLU hidden layer -> output."""

    def __init__(self, in_dim, hidden_dim, out_dim, mode="regression", seed=0):
        if mode not in ("regression", "classification"):
            raise ValueError("mode must be regression or classification")
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, math.sqrt(2.0 / in_dim), (in_dim, hidden_dim))
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.normal(0, math.sqrt(2.0 / hidden_dim), (hidden_dim, out_dim))
        self.b2 = np.zeros((1, out_dim))
        self.mode = mode

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = relu(self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        if self.mode == "classification":
            self.probs = softmax(self.z2)
            return self.probs
        return self.z2

    def loss(self, y):
        if self.mode == "regression":
            diff = self.z2 - y
            return 0.5 * np.mean(np.sum(diff * diff, axis=1))
        return np.mean(-np.log(self.probs[np.arange(y.shape[0]), y] + 1e-12))

    def backward(self, y):
        N = self.X.shape[0]
        if self.mode == "regression":
            dz2 = (self.z2 - y) / N
        else:
            dz2 = (self.probs - one_hot(y, self.z2.shape[1])) / N
        dW2 = self.h1.T @ dz2
        db2 = dz2.sum(axis=0, keepdims=True)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * d_relu(self.z1)
        dW1 = self.X.T @ dz1
        db1 = dz1.sum(axis=0, keepdims=True)
        return dW1, db1, dW2, db2

    def step(self, grads, lr=1e-2, weight_decay=0.0):
        dW1, db1, dW2, db2 = grads
        if weight_decay > 0:
            dW1 += weight_decay * self.W1
            dW2 += weight_decay * self.W2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1
        self.W2 -= lr * dW2
        self.b2 -= lr * db2


def batch_iter(X, y, batch_size, shuffle=True, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(X.shape[0])
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        indices = idx[i : i + batch_size]
        yield X[indices], y[indices]


def train(model, X, y, epochs=200, lr=1e-2, batch_size=64,
          weight_decay=0.0, verbose=True):
    losses = []
    for epoch in range(1, epochs + 1):
        for Xb, yb in batch_iter(X, y, batch_size, seed=epoch):
            model.forward(Xb)
            grads = model.backward(yb)
            model.step(grads, lr=lr, weight_decay=weight_decay)
        model.forward(X)
        loss = float(model.loss(y))
        if not np.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at epoch {epoch}")
        losses.append(loss)
        if verbose and epoch % max(1, epochs // 10) == 0:
            print(f"epoch {epoch:4d}/{epochs} loss={loss:.6f}")
    return losses


def target(x):
    return np.maximum(0, 0.5 * x + 0.2) + 0.3 * np.maximum(0, -x + 0.5)


def rings(n=600, inner_r=0.5, gap=0.2, noise=0.06, seed=0):
    rng = np.random.default_rng(seed)
    n2 = n // 2
    theta1 = rng.uniform(0, 2 * np.pi, n2)
    r1 = inner_r + noise * rng.normal(size=n2)
    x1 = np.c_[r1 * np.cos(theta1), r1 * np.sin(theta1)]
    theta2 = rng.uniform(0, 2 * np.pi, n - n2)
    r2 = inner_r + gap + noise * rng.normal(size=n - n2)
    x2 = np.c_[r2 * np.cos(theta2), r2 * np.sin(theta2)]
    return np.vstack([x1, x2]), np.array([0] * n2 + [1] * (n - n2))


def three_blobs(n=900, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([[0, 0], [2.5, 0.5], [-2.0, 1.5]])
    cov = np.array([[0.4, 0.0], [0.0, 0.4]])
    Xs, ys = [], []
    for k, mean in enumerate(means):
        Xs.append(rng.multivariate_normal(mean, cov, size=n // 3))
        ys.append(np.full(n // 3, k))
    return np.vstack(Xs), np.hstack(ys)


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_plots(name, model, X, y, losses, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(np.arange(len(losses)), losses)
    ax.set(xlabel="Epoch (0 = before training)", ylabel="Training loss",
           title=f"{name.capitalize()}: training loss")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_loss.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    if name == "regression":
        grid = np.linspace(-2, 2, 400).reshape(-1, 1)
        ax.scatter(X[:, 0], y[:, 0], s=9, alpha=0.3, label="Noisy training data")
        ax.plot(grid[:, 0], target(grid)[:, 0], label="Target function", linewidth=2)
        ax.plot(grid[:, 0], model.forward(grid)[:, 0], "--", label="MLP prediction")
        ax.set(xlabel="x", ylabel="y", title="Regression: learned function")
        ax.legend()
        suffix = "fit"
    else:
        x1 = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 180)
        x2 = np.linspace(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3, 180)
        xx, yy = np.meshgrid(x1, x2)
        grid = np.c_[xx.ravel(), yy.ravel()]
        # Bound temporary activation memory when plotting the grid.
        predictions = np.concatenate([
            model.forward(grid[i : i + 2048]).argmax(axis=1)
            for i in range(0, len(grid), 2048)
        ]).reshape(xx.shape)
        colors = ["#2563eb", "#e67e22", "#16a34a"]
        ax.contourf(xx, yy, predictions, levels=np.arange(y.max() + 2) - 0.5,
                    colors=colors[: y.max() + 1], alpha=0.16)
        for label in np.unique(y):
            points = X[y == label]
            ax.scatter(points[:, 0], points[:, 1], s=10, alpha=0.65,
                       color=colors[label], label=f"Class {label}")
        ax.set(xlabel="x1", ylabel="x2",
               title=f"{name.capitalize()}: decision regions and training samples")
        ax.legend()
        suffix = "boundary"
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}_{suffix}.png", dpi=150)
    plt.close(fig)


def run_example(name, output_dir):
    if name == "regression":
        rng = np.random.default_rng(0)
        X = rng.uniform(-2, 2, (400, 1))
        y = target(X) + 0.05 * rng.normal(size=(400, 1))
        model = MLP2(1, 32, 1, mode="regression", seed=1)
        data_seed, model_seed = 0, 1
        settings = dict(epochs=500, lr=1e-2, batch_size=64, weight_decay=1e-4)
    elif name == "binary":
        X, y = rings(n=800, inner_r=0.6, gap=0.5, noise=0.07, seed=1)
        model = MLP2(2, 64, 2, mode="classification", seed=2)
        data_seed, model_seed = 1, 2
        settings = dict(epochs=200, lr=5e-3, batch_size=64, weight_decay=1e-4)
    else:
        X, y = three_blobs(n=900, seed=4)
        model = MLP2(2, 64, 3, mode="classification", seed=3)
        data_seed, model_seed = 4, 3
        settings = dict(epochs=200, lr=5e-3, batch_size=64, weight_decay=0.0)

    print(f"\n=== {name} ===")
    model.forward(X)
    initial_loss = float(model.loss(y))
    print(f"initial loss={initial_loss:.6f}")
    start = time.perf_counter()
    losses = train(model, X, y, **settings)
    train_seconds = time.perf_counter() - start
    outputs = model.forward(X)
    accuracy = ""
    if model.mode == "classification":
        accuracy = float(np.mean(outputs.argmax(axis=1) == y))
        print(f"TRAIN accuracy={accuracy:.4%}")
    else:
        xt = np.linspace(-2, 2, 9).reshape(-1, 1)
        values = np.hstack([xt, target(xt), model.forward(xt)])
        print("Columns: x, target(x), prediction")
        print(np.array2string(values, precision=6))
        write_csv(output_dir / "regression_predictions.csv",
                  ["x", "target", "prediction"],
                  [dict(zip(["x", "target", "prediction"], row)) for row in values])

    full_losses = [initial_loss] + losses
    write_csv(output_dir / f"{name}_loss.csv", ["epoch", "loss"],
              [{"epoch": epoch, "loss": loss} for epoch, loss in enumerate(full_losses)])
    save_plots(name, model, X, y, full_losses, output_dir)
    return dict(example=name, n_samples=len(X), in_dim=model.W1.shape[0],
                hidden_dim=model.W1.shape[1], out_dim=model.W2.shape[1],
                data_seed=data_seed, model_seed=model_seed, **settings,
                initial_loss=initial_loss, final_loss=losses[-1],
                train_accuracy=accuracy, train_seconds=train_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", choices=["all", "regression", "binary", "multiclass"],
                        default="all")
    parser.add_argument("--output-dir", type=Path, default=Path("results/part1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    names = ["regression", "binary", "multiclass"] if args.example == "all" else [args.example]
    rows = [run_example(name, args.output_dir) for name in names]
    # Keep metrics from other examples when running one example at a time.
    summary_path = args.output_dir / "summary.csv"
    if summary_path.exists() and args.example != "all":
        with summary_path.open(newline="", encoding="utf-8") as stream:
            previous = list(csv.DictReader(stream))
        rows = [row for row in previous if row["example"] not in names] + rows
    write_csv(summary_path, list(rows[-1]), rows)
    print(f"\nResults saved to: {args.output_dir.resolve()}")
    print("Classification metrics are training accuracy, not held-out test accuracy.")


if __name__ == "__main__":
    main()
