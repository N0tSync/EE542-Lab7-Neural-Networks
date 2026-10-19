"""
Collect the NumPy and PyTorch result files and build the comparison for the report.

Reads results/*.json (written by src/part2_numpy_cnn.py and src/part3_torch.py, format in DATA.md), checks
them, and writes:
    results/comparison.md    accuracy / time / sample-count table + run configuration table
    results/comparison.csv   the same numbers for a spreadsheet
    results/comparison.png   throughput and accuracy chart
    results/summary.json     the numbers used in the report

Usage:
    python scripts/compare.py                       # all results/*.json
    python scripts/compare.py --results results_check
"""
import argparse
import csv
import glob
import json
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # lab7_common, models

import lab7_common as lc


def load_runs(results_dir):
    runs = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        name = os.path.basename(path)
        if name == "summary.json" or name.startswith("part1_") or "smoke" in name:
            continue                       # summary output, Part I file and quick smoke runs are skipped
        with open(path) as f:
            r = json.load(f)
        problems = lc.validate_results(r)
        if problems:
            print(f"skipping {path}:\n  - " + "\n  - ".join(problems))
            continue
        r["file"] = os.path.basename(path)
        if r.get("device_name") and r["device_name"] not in r["device"]:   # Part III: "cuda" + GPU name
            r["device"] = f"{r['device']} ({r['device_name']})"
        if isinstance(r["image_size"], list):
            r["image_size"] = r["image_size"][0]
        r["is_gpu"] = any(k in r["device"].lower() for k in ("cuda", "gpu", "mps"))
        r["is_numpy"] = "numpy" in r["model"].lower()
        runs.append(r)
    # NumPy first, then CPU, then GPU; slowest first, so the baseline is the slowest NumPy run
    runs.sort(key=lambda r: (not r["is_numpy"], r["is_gpu"], r["train_images_per_sec"]))
    return runs


def fmt_time(s):
    if s >= 3600:
        return f"{s/3600:.2f} h"
    if s >= 60:
        return f"{s/60:.1f} min"
    if s < 1:
        return f"{1000*s:.1f} ms"
    return f"{s:.2f} s"


def build(runs):
    base = next((r for r in runs if r["is_numpy"]), runs[0])
    rows = []
    for r in runs:
        rows.append({
            "Model": r["model"], "Device": r["device"],
            "Train images": r["n_train"], "Epochs": r["epochs"], "Test images": r["n_test"],
            "Train time": fmt_time(r["train_seconds"]),
            "Train img/s": f"{r['train_images_per_sec']:,.0f}",
            "Test time": fmt_time(r["test_seconds"]),
            "Test ms/img": f"{1000*r['test_seconds']/r['n_test']:.3f}",
            "Train speed-up": f"{r['train_images_per_sec']/base['train_images_per_sec']:,.1f}x",
            "Test speed-up": f"{r['test_images_per_sec']/base['test_images_per_sec']:,.1f}x",
            "Test accuracy": f"{100*r['test_accuracy']:.2f}%",
        })
    config_rows = [{
        "Model": r["model"], "Device": r["device"], "Machine": r.get("machine", ""),
        "Split fingerprint": r["data_split_fingerprint"], "Seed": r["seed"],
        "Image size": f"{r['image_size']}x{r['image_size']}",
        "Other settings": ", ".join(f"{k}={v}" for k, v in sorted((r.get("config") or {}).items())
                                    if k not in ("epochs", "seed")) or "-",
    } for r in runs]
    return base, rows, config_rows


def md_table(rows):
    cols = list(rows[0])
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(row[c]) for c in cols) + " |" for row in rows]
    return "\n".join(out)


def findings(runs, base):
    """Plain-language findings computed from the numbers, so the report's conclusions match the data."""
    lines = []
    fps = {r["data_split_fingerprint"] for r in runs}
    lines.append((f"All {len(runs)} runs used" if len(runs) > 1 else "The run used") +
                 f" the data split with fingerprint `{fps.pop()}`."
                 if len(fps) == 1 else
                 "WARNING: the runs used different data splits, so they are not directly comparable.")
    ep = {r["epochs"] for r in runs}
    if len(ep) > 1:
        lines.append(f"WARNING: the runs used different numbers of epochs ({sorted(ep)}).")
    nt = {r["n_test"] for r in runs}
    if len(nt) > 1:
        lines.append(f"Note: the runs were tested on different numbers of images ({sorted(nt)}).")
    for r in runs:
        if r is base:
            continue
        tr = r["train_images_per_sec"] / base["train_images_per_sec"]
        te = r["test_images_per_sec"] / base["test_images_per_sec"]
        d = 100 * (r["test_accuracy"] - base["test_accuracy"])
        lines.append(
            f"{r['model']} on {r['device']} trains {tr:,.1f}x {'faster' if tr >= 1 else 'slower'} "
            f"and tests {te:,.1f}x {'faster' if te >= 1 else 'slower'} per image than {base['model']}, "
            f"with test accuracy {100*r['test_accuracy']:.2f}% vs {100*base['test_accuracy']:.2f}% "
            f"({'+' if d >= 0 else ''}{d:.2f} percentage points).")
    for r in runs:
        if abs(r["test_accuracy"] - 0.5) < 0.03:
            lines.append(f"{r['model']} ({100*r['test_accuracy']:.2f}%) is close to chance level (50%).")
    return lines


def save_chart(runs, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed - chart skipped")
        return False
    import textwrap
    labels = ["\n".join(textwrap.wrap(f"{r['model']} - {r['device']}", 22)) for r in runs]
    colors = ["#2a6fdb" if r["is_gpu"] else "#9aa5b1" for r in runs]
    fig, ax = plt.subplots(1, 3, figsize=(max(12, 4.5 * len(runs)), 4.8))
    panels = [("Training throughput (images/s, log scale)", [r["train_images_per_sec"] for r in runs], True),
              ("Testing throughput (images/s, log scale)", [r["test_images_per_sec"] for r in runs], True),
              ("Test accuracy (%)", [100 * r["test_accuracy"] for r in runs], False)]
    for a, (title, vals, log) in zip(ax, panels):
        bars = a.bar(range(len(vals)), vals, color=colors, width=0.6)
        a.set_xticks(range(len(vals)), labels)
        if log:
            a.set_yscale("log")
            a.set_ylim(min(vals) / 3, max(vals) * 3)
        else:
            a.set_ylim(0, 100)
            a.axhline(50, color="#c0392b", lw=1, ls="--")
            a.text(len(vals) - 0.5, 51, "chance", color="#c0392b", ha="right", fontsize=8)
        for b, v in zip(bars, vals):
            a.text(b.get_x() + b.get_width() / 2, v, f"{v:,.1f}" if not log else f"{v:,.0f}",
                   ha="center", va="bottom", fontsize=8)
        a.set_title(title, fontsize=10)
        a.tick_params(axis="x", labelsize=8)
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    runs = load_runs(args.results)
    if not runs:
        sys.exit(f"No valid result files in {args.results}/ yet. The model scripts save theirs there "
                 "(see DATA.md section 5), then run this again.")
    base, rows, config_rows = build(runs)
    found = findings(runs, base)

    md = ["# NumPy vs. PyTorch comparison", "",
          "## Results", "", md_table(rows), "",
          f"Speed-ups are relative to {base['model']} ({base['device']}). "
          "Test time is per image over the whole test set.", "",
          "## Run configuration", "", md_table(config_rows), "",
          "## Findings", ""] + [f"- {x}" for x in found]
    with open(os.path.join(args.results, "comparison.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    with open(os.path.join(args.results, "comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    chart = save_chart(runs, os.path.join(args.results, "comparison.png"))
    summary = {"baseline": base["model"], "results_table": md_table(rows),
               "config_table": md_table(config_rows), "findings": found,
               "runs": [{k: r[k] for k in ("file", "model", "device", "n_train", "n_test", "epochs",
                                           "train_seconds", "test_seconds", "train_images_per_sec",
                                           "test_images_per_sec", "test_accuracy",
                                           "data_split_fingerprint")} for r in runs]}
    with open(os.path.join(args.results, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n".join(md))
    print(f"\nwrote {args.results}/comparison.md, comparison.csv, summary.json"
          + (", comparison.png" if chart else ""))


if __name__ == "__main__":
    main()
