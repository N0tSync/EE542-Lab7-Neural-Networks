"""
End-to-end check of the shared pipeline, and a reference example for the model scripts.

Trains a small logistic-regression model (NumPy only) on data_small/ twice - once one image
at a time, once in batches - using lab7_common for loading, timing and logging. Then the
results go through scripts/check_results.py and scripts/compare.py exactly like the real runs.

These are pipeline-test numbers, not lab results: they are written to results_check/,
never to results/.

Usage:
    python scripts/prepare_data.py --raw raw --out data_small --max-per-class 1000   # once
    python scripts/pipeline_check.py --data data_small
    python scripts/compare.py --results results_check
"""
import argparse
import os
import platform
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # lab7_common, models

import numpy as np

import lab7_common as lc


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def run(name, X_tr, y_tr, X_te, y_te, data_root, out_dir, epochs, lr, batch, block_size, seed):
    rng = np.random.default_rng(seed)
    Xf_tr = X_tr.reshape(len(X_tr), -1)
    Xf_te = X_te.reshape(len(X_te), -1)
    mean = Xf_tr.mean(axis=0)                          # center features (learned on train only)
    Xf_tr, Xf_te = Xf_tr - mean, Xf_te - mean
    w = np.zeros(Xf_tr.shape[1], dtype=np.float32)
    b = 0.0
    cpu = platform.processor() or platform.machine()
    log = lc.RunLogger(name, f"CPU ({cpu})", data_root=data_root,
                       config={"epochs": epochs, "lr": lr, "batch_size": batch, "seed": seed,
                               "loss": "MSE on sigmoid output"})
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(Xf_tr))
        total = 0.0
        with lc.Timer() as t:
            for i in range(0, len(order), batch):
                idx = order[i:i + batch]
                xb, yb = Xf_tr[idx], y_tr[idx].astype(np.float32)
                p = sigmoid(xb @ w + b)
                err = p - yb
                total += float(np.sum(err ** 2))
                g = 2 * err * p * (1 - p) / len(idx)    # d(MSE)/dz through the sigmoid
                w -= lr * (xb.T @ g)
                b -= lr * float(g.sum())
        train_acc = float(np.mean((sigmoid(Xf_tr @ w + b) > lc.THRESHOLD) == y_tr))
        log.epoch(epoch, loss=total / len(Xf_tr), seconds=t.seconds, images=len(Xf_tr),
                  train_accuracy=train_acc)
    for i in range(0, len(Xf_te), block_size):
        xb, yb = Xf_te[i:i + block_size], y_te[i:i + block_size]
        with lc.Timer() as t:
            if batch == 1:
                pred = np.array([sigmoid(x @ w + b) > lc.THRESHOLD for x in xb])
            else:
                pred = sigmoid(xb @ w + b) > lc.THRESHOLD
        log.test_block(images=len(xb), correct=int(np.sum(pred == yb)), seconds=t.seconds)
    fname = "check_" + name.split("(")[-1].rstrip(")").replace(", ", "_").replace(" ", "_").lower() + ".json"
    return log.save(os.path.join(out_dir, fname))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_small")
    ap.add_argument("--out", default="results_check")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X_tr, y_tr = lc.load_arrays(args.data, "train", seed=args.seed)
    X_te, y_te = lc.load_arrays(args.data, "test", seed=args.seed + 1)
    print(f"loaded {len(X_tr)} train / {len(X_te)} test images with the shared rule "
          f"(shape {X_tr.shape[1:]}, range {X_tr.min():.1f}-{X_tr.max():.1f}, "
          f"train cat/dog {np.bincount(y_tr).tolist()}, test cat/dog {np.bincount(y_te).tolist()})\n")
    run("Logistic regression (NumPy, one image at a time)", X_tr, y_tr, X_te, y_te, args.data, args.out,
        args.epochs, lr=0.01, batch=1, block_size=100, seed=args.seed)
    print()
    run("Logistic regression (NumPy, batches of 64)", X_tr, y_tr, X_te, y_te, args.data, args.out,
        args.epochs, lr=0.01, batch=64, block_size=100, seed=args.seed)


if __name__ == "__main__":
    main()
