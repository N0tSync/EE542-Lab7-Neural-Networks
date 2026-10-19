"""
Quick tests for the shared tools: src/lab7_common.py, scripts/check_results.py, scripts/compare.py.
Uses made-up result files in a temporary folder (they never touch results/). Runs in a few seconds.

    python tests/test_results_pipeline.py      (from the repository root)
"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS)
import lab7_common as lc  # noqa: E402
import compare as mr  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def fake_data(root):
    """Tiny data/ folder: 3 images per class per split, plus stats.json."""
    rng = np.random.default_rng(0)
    for split in ("train", "test"):
        for name in lc.CLASSES:
            os.makedirs(os.path.join(root, split, name))
            for i in range(3):
                arr = rng.integers(0, 255, (50 + i, 80, 3), dtype=np.uint8)
                Image.fromarray(arr).save(os.path.join(root, split, name, f"{name}_{i}.jpg"))
    stats = {"counts": {"train/cat": 3, "train/dog": 3, "test/cat": 3, "test/dog": 3},
             "train_total": 6, "test_total": 6, "images_used": 12, "rows_read": 13, "bad_rows": 1,
             "test_frac": 0.5, "seed": 0, "split_fingerprint": "abc123abc123"}
    with open(os.path.join(root, "stats.json"), "w") as f:
        json.dump(stats, f)


def logged_run(model, device, data_root, img_per_s, acc, n_train=6, n_test=6, epochs=2):
    """Use RunLogger the way the model scripts do, with fixed timings."""
    log = lc.RunLogger(model, device, data_root=data_root, quiet=True,
                       config={"epochs": epochs, "seed": 0, "lr": 0.001})
    for e in range(epochs):
        log.epoch(e + 1, loss=0.3 - 0.05 * e, seconds=n_train / img_per_s, images=n_train)
    log.test_block(images=n_test, correct=round(acc * n_test), seconds=n_test / img_per_s)
    return log


def main():
    tmp = tempfile.mkdtemp(prefix="test_results_")
    try:
        data = os.path.join(tmp, "data")
        res = os.path.join(tmp, "results")
        fake_data(data)

        # 1. shared preprocessing rule
        X, y = lc.load_arrays(data, "train")
        check("1. load_arrays: 64x64 RGB, float32 in [0,1], labels cat=0 dog=1",
              X.shape == (6, 64, 64, 3) and X.dtype == np.float32 and 0 <= X.min() and X.max() <= 1
              and sorted(y.tolist()) == [0, 0, 0, 1, 1, 1], f"shape {X.shape}, labels {sorted(y.tolist())}")

        # 2. RunLogger writes a valid JSON + .log, totals add up
        r = logged_run("NumPy CNN", "CPU", data, img_per_s=100, acc=0.5).save(
            os.path.join(res, "part2_numpy_results.json"))
        ok = (not lc.validate_results(r) and r["n_test"] == 6 and abs(r["train_seconds"] - 0.12) < 1e-9
              and os.path.exists(os.path.join(res, "part2_numpy_results.log"))
              and r["data_split_fingerprint"] == "abc123abc123")
        check("2. RunLogger saves JSON + log with correct totals and fingerprint", ok,
              f"train {r['train_seconds']:.2f}s, n_test {r['n_test']}, problems {lc.validate_results(r)}")

        # 3. validator catches common mistakes
        bad = copy.deepcopy(r)
        bad["test_accuracy"] = 55.0                      # percent instead of 0-1
        del bad["n_train"]
        probs = lc.validate_results(bad)
        bad2 = copy.deepcopy(r)
        bad2["test_accuracy"] = 55.0
        check("3. validator catches a missing field and accuracy given as a percentage",
              any("n_train" in p for p in probs) and any("between 0 and 1" in p for p in lc.validate_results(bad2)),
              f"{len(probs)} + {len(lc.validate_results(bad2))} problems found")

        # 4. Timer with a sync function calls it
        calls = []
        with lc.Timer(sync=lambda: calls.append(1)) as t:
            sum(range(1000))
        check("4. Timer measures time and calls the GPU sync before and after", t.seconds > 0 and len(calls) == 2)

        # 5. compare: table, speed-ups, findings
        logged_run("PyTorch CatDogCNN", "cuda (Test GPU)", data, img_per_s=10000, acc=5 / 6).save(
            os.path.join(res, "part3_torch_cuda.json"))
        runs = mr.load_runs(res)
        base, rows, _ = mr.build(runs)
        found = mr.findings(runs, base)
        check("5. comparison: NumPy is the baseline, GPU speed-up 100x, findings match",
              base["model"] == "NumPy CNN" and rows[1]["Train speed-up"] == "100.0x"
              and any("100.0x faster" in f for f in found) and any("All 2 runs used the data split" in f for f in found),
              f"speed-up {rows[1]['Train speed-up']}")

        # 5b. Part III's own result format (src/part3_torch.py) is accepted by the tools
        part3 = {"model": "CatDogCNN", "device": "cuda", "device_name": "NVIDIA Test GPU",
                 "data_split_fingerprint": "abc123abc123", "n_train": 6, "n_test": 6, "epochs": 2,
                 "seed": 0, "image_size": [64, 64], "train_seconds": 0.002, "train_images_per_sec": 6000.0,
                 "test_seconds": 0.001, "test_images_per_sec": 6000.0, "test_accuracy": 1.0,
                 "epochs_log": [{"epoch": 1, "loss": 0.2, "seconds": 0.001, "train_accuracy": 0.5},
                                {"epoch": 2, "loss": 0.1, "seconds": 0.001, "train_accuracy": 1.0}],
                 "test_blocks": [{"block": 1, "n_images": 6, "accuracy": 1.0, "seconds": 0.001}]}
        res3 = os.path.join(tmp, "results_p3")
        os.makedirs(res3)
        shutil.copy(os.path.join(res, "part2_numpy_results.json"), res3)
        with open(os.path.join(res3, "part3_torch_cuda.json"), "w") as f:
            json.dump(part3, f)
        with open(os.path.join(res3, "part3_smoke_cuda.json"), "w") as f:
            json.dump(part3, f)
        runs3 = mr.load_runs(res3)
        check("5b. Part III format accepted (image_size list, n_images, device_name); smoke runs skipped",
              not lc.validate_results(part3) and len(runs3) == 2
              and runs3[1]["device"] == "cuda (NVIDIA Test GPU)" and runs3[1]["is_gpu"],
              f"problems {lc.validate_results(part3)}, runs {[r['file'] for r in runs3]}")

        # Dataset verification is metadata, not a model run.
        metadata = os.path.join(res, "data_verification.json")
        with open(metadata, "w") as f:
            json.dump({"passed": True, "images_verified": 12}, f)
        p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "check_results.py")],
                           capture_output=True, text=True, cwd=tmp)
        # The default command expects the conventional results/ directory.
        check("5c. data verification reports are excluded from model result checks",
              p.returncode == 0 and len(mr.load_runs(res)) == 2,
              p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr.strip())

        # 6. different splits are flagged by check_results.py
        other = os.path.join(tmp, "results_mixed")
        shutil.copytree(res, other)
        with open(os.path.join(other, "part3_torch_cuda.json")) as f:
            j = json.load(f)
        j["data_split_fingerprint"] = "different999"
        with open(os.path.join(other, "part3_torch_cuda.json"), "w") as f:
            json.dump(j, f)
        p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "check_results.py")] +
                           [os.path.join(other, n) for n in sorted(os.listdir(other)) if n.endswith(".json")],
                           capture_output=True, text=True)
        check("6. check_results.py fails when runs use different data splits",
              p.returncode == 1 and "different data splits" in p.stdout)

        # 7. compare.py writes all outputs
        p = subprocess.run([sys.executable, os.path.join(SCRIPTS, "compare.py"), "--results", res],
                           capture_output=True, text=True, cwd=tmp)
        outs = [n for n in ("comparison.md", "comparison.csv", "summary.json") if os.path.exists(os.path.join(res, n))]
        check("7. compare.py writes comparison.md, .csv and summary.json", p.returncode == 0 and len(outs) == 3,
              f"wrote {outs}")

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(RESULTS)
    print(f"\n{passed}/{len(RESULTS)} tests passed - " + ("ALL GOOD" if passed == len(RESULTS) else "SOMETHING FAILED"))
    sys.exit(0 if passed == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
