"""
Lab 7: run both cats-vs-dogs models on the same data and settings, then build the comparison.

    python scripts/run_experiments.py --data data --epochs 10                  # both models
    python scripts/run_experiments.py --data data --epochs 10 --only torch     # just the GPU run
    python scripts/run_experiments.py --data data_small --epochs 2 --out results_check   # quick test

Steps: the NumPy CNN (src/part2_numpy_cnn.py), the PyTorch CNN (src/part3_torch.py), then
scripts/check_results.py and scripts/compare.py. Console output of
each step is saved to logs/run_<step>.log. A step that fails stops the run.
"""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def step(name, cmd, log_dir):
    print(f"\n=== {name}: {' '.join(cmd)}")
    log_path = log_dir / f"run_{name}.log"
    t0 = time.perf_counter()
    with open(log_path, "w") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in p.stdout:
            print(line, end="")
            log.write(line)
        p.wait()
    print(f"=== {name} finished in {time.perf_counter() - t0:.0f}s (exit {p.returncode}), log: {log_path}")
    return p.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results", help="results folder (results_check for tests)")
    ap.add_argument("--only", choices=["numpy", "torch"], help="run one model only")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="PyTorch device")
    args = ap.parse_args()

    log_dir = ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    py = sys.executable
    common = ["--data", args.data, "--epochs", str(args.epochs), "--seed", str(args.seed), "--out", args.out]
    steps = []
    if args.only in (None, "numpy"):
        steps.append(("numpy_cnn", [py, "src/part2_numpy_cnn.py", *common]))
    if args.only in (None, "torch"):
        torch_cmd = [py, "src/part3_torch.py", "--data-dir", args.data, "--epochs", str(args.epochs),
                     "--seed", str(args.seed), "--device", args.device]
        if args.out != "results":                    # part3_torch.py writes results/ by default
            torch_cmd += ["--output", os.path.join(args.out, "part3_torch.json")]
        steps.append(("torch_cnn", torch_cmd))
    steps.append(("check_results", None))          # file list is built after the models have run
    steps.append(("compare", [py, "scripts/compare.py", "--results", args.out]))

    for name, cmd in steps:
        if name == "check_results":
            cmd = [py, "scripts/check_results.py"] + sorted(
                os.path.join(args.out, f) for f in os.listdir(ROOT / args.out)
                if f.endswith(".json") and f not in {"summary.json", "data_verification.json"}
                and not f.startswith("part1_"))
        if step(name, cmd, log_dir) != 0:
            sys.exit(f"\nstopped: step '{name}' failed (see logs/run_{name}.log)")
    print("\nAll steps finished.")


if __name__ == "__main__":
    main()
