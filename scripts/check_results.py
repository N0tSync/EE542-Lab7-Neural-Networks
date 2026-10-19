"""
Check that result files follow the shared format in DATA.md (for the NumPy and PyTorch results).

Usage:
    python scripts/check_results.py                          # every results/*.json
    python scripts/check_results.py results/part2_numpy_results.json
"""
import glob
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # lab7_common, models

import lab7_common as lc


def main():
    paths = sys.argv[1:] or sorted(glob.glob("results/*.json"))
    paths = [p for p in paths if not p.endswith("summary.json")
             and not os.path.basename(p).startswith("part1_")     # Part I file is not a CNN run
             and "smoke" not in os.path.basename(p)]              # quick smoke runs
    if not paths:
        sys.exit("No result files found in results/. The model scripts save theirs there.")
    fingerprints = {}
    bad = 0
    for path in paths:
        try:
            with open(path) as f:
                r = json.load(f)
        except (OSError, ValueError) as e:
            print(f"[FAIL] {path}: cannot read ({e})")
            bad += 1
            continue
        problems = lc.validate_results(r)
        if problems:
            bad += 1
            print(f"[FAIL] {path}")
            for p in problems:
                print(f"         - {p}")
        else:
            fingerprints[path] = r["data_split_fingerprint"]
            print(f"[ OK ] {path}: {r['model']} on {r['device']} | {r['n_train']} train / {r['n_test']} test | "
                  f"acc {100*r['test_accuracy']:.2f}% | split {r['data_split_fingerprint']}")
    if len(set(fingerprints.values())) > 1:
        bad += 1
        print("[FAIL] the runs used different data splits:")
        for p, fp in fingerprints.items():
            print(f"         {fp}  {p}")
    print("\nAll result files OK." if not bad else f"\n{bad} problem(s) found.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
