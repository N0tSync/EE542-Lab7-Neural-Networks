"""
Lab 7 - shared code for Part II (A, NumPy) and Part III (C, PyTorch). Maintained by C (data and shared tools).

Everything here follows DATA.md, so both models load images the same way and write
results in the same format. Nothing here needs PyTorch unless you call torch_transform().

    import lab7_common as lc

    # --- loading (A, NumPy) ---
    X_train, y_train = lc.load_arrays("data", "train")          # float32 (N,64,64,3) in [0,1], labels 0/1
    X_test,  y_test  = lc.load_arrays("data", "test")

    # --- loading (C, PyTorch) ---
    from torchvision import datasets
    train_set = datasets.ImageFolder("data/train", transform=lc.torch_transform())

    # --- logging + timing (both models) ---
    log = lc.RunLogger("NumPy CNN", "CPU (Apple M2)", data_root="data",
                       config={"epochs": 10, "lr": 0.001, "batch_size": 1, "seed": 0})
    for epoch in range(10):
        with lc.Timer() as t:                          # PyTorch on GPU: lc.Timer(sync=torch.cuda.synchronize)
            loss = train_one_epoch()
        log.epoch(epoch + 1, loss=loss, seconds=t.seconds, images=len(X_train))
    for block in blocks:
        with lc.Timer() as t:
            correct = test_block(block)
        log.test_block(images=len(block), correct=correct, seconds=t.seconds)
    log.save("results/part2_numpy_results.json")   # also writes results/part2_numpy_results.log

Tools in scripts/ add src/ to sys.path to import this module.
"""
import json
import os
import platform
import time
from datetime import datetime

import numpy as np
from PIL import Image

IMG_SIZE = 64
CLASSES = ("cat", "dog")        # label 0 = cat, 1 = dog (alphabetical, same as torchvision ImageFolder)
THRESHOLD = 0.5                 # model output > 0.5 -> dog

REQUIRED_FIELDS = {
    "model": str, "device": str, "data_split_fingerprint": str,
    "n_train": int, "n_test": int, "epochs": int, "seed": int, "image_size": (int, list),
    "train_seconds": (int, float), "train_images_per_sec": (int, float),
    "test_seconds": (int, float), "test_images_per_sec": (int, float),
    "test_accuracy": (int, float), "epochs_log": list, "test_blocks": list,
}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def split_fingerprint(data_root):
    """Fingerprint written by prepare_data.py. Equal fingerprints = identical train/test split."""
    try:
        with open(os.path.join(data_root, "stats.json")) as f:
            return json.load(f).get("split_fingerprint", "unknown")
    except (OSError, ValueError):
        return "unknown"


def list_split(data_root, split):
    """[(path, label), ...] for data_root/split/{cat,dog}, sorted by file name."""
    items = []
    for label, name in enumerate(CLASSES):
        folder = os.path.join(data_root, split, name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"{folder} not found - run prepare_data.py first")
        for fname in sorted(os.listdir(folder)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                items.append((os.path.join(folder, fname), label))
    return items


def load_image(path):
    """The shared preprocessing rule: RGB, 64x64 bilinear, /255 -> float32 (64, 64, 3) in [0, 1]."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_image_uint8(path):
    """Same resize as load_image but kept as uint8 (4x less memory). Divide by 255 before use."""
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def load_arrays(data_root, split, max_images=0, seed=0, uint8=False):
    """Load a whole split as NumPy arrays, shuffled with a fixed seed. Optionally keep max_images.
    uint8=True keeps pixels as 0-255 to save memory (the full training set is ~920 MB as float32,
    ~230 MB as uint8); then apply the /255 step per image, e.g. X[i] / 255.0."""
    items = list_split(data_root, split)
    order = np.random.default_rng(seed).permutation(len(items))
    if max_images:
        order = order[:max_images]
    load = load_image_uint8 if uint8 else load_image
    X = np.stack([load(items[i][0]) for i in order])
    y = np.array([items[i][1] for i in order], dtype=np.int64)
    return X, y


def torch_transform():
    """The same preprocessing rule for torchvision (ImageFolder / DataLoader)."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),   # HxWxC uint8 -> CxHxW float in [0, 1]  (= /255)
    ])


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
class Timer:
    """with Timer() as t: ...  -> t.seconds.  For GPU code pass sync=torch.cuda.synchronize,
    so queued GPU work is finished before the clock is read (otherwise times come out too low)."""

    def __init__(self, sync=None):
        self.sync = sync
        self.seconds = 0.0

    def __enter__(self):
        if self.sync:
            self.sync()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.sync:
            self.sync()
        self.seconds = time.perf_counter() - self._t0
        return False


# ---------------------------------------------------------------------------
# Results logging (the format in DATA.md section 5)
# ---------------------------------------------------------------------------
class RunLogger:
    def __init__(self, model, device, data_root="data", config=None, quiet=False):
        self.config = dict(config or {})
        self.quiet = quiet
        self.lines = []
        self.r = {
            "model": model, "device": device,
            "data_root": data_root, "data_split_fingerprint": split_fingerprint(data_root),
            "image_size": IMG_SIZE,
            "epochs": int(self.config.get("epochs", 0)), "seed": int(self.config.get("seed", 0)),
            "config": self.config,
            "machine": f"{platform.system()} {platform.machine()} / Python {platform.python_version()}",
            "started": datetime.now().isoformat(timespec="seconds"),
            "epochs_log": [], "test_blocks": [],
        }
        self._say(f"run: {model} on {device} | data={data_root} "
                  f"(fingerprint {self.r['data_split_fingerprint']}) | config={self.config}")

    def _say(self, text):
        line = f"{datetime.now().strftime('%H:%M:%S')}  {text}"
        self.lines.append(line)
        if not self.quiet:
            print(line)

    def epoch(self, epoch, loss, seconds, images, train_accuracy=None):
        e = {"epoch": int(epoch), "loss": float(loss), "seconds": float(seconds),
             "images": int(images), "images_per_sec": images / seconds if seconds else 0.0}
        if train_accuracy is not None:
            e["train_accuracy"] = float(train_accuracy)
        self.r["epochs_log"].append(e)
        self._say(f"epoch {epoch:3d} | loss {loss:.4f} | {seconds:8.2f}s | {e['images_per_sec']:9.1f} img/s"
                  + (f" | train acc {100*train_accuracy:.2f}%" if train_accuracy is not None else ""))

    def test_block(self, images, correct, seconds, compute_seconds=None):
        b = {"block": len(self.r["test_blocks"]) + 1, "images": int(images), "correct": int(correct),
             "accuracy": correct / images, "seconds": float(seconds),
             "images_per_sec": images / seconds if seconds else 0.0}
        if compute_seconds is not None:
            b["compute_seconds"] = float(compute_seconds)
        self.r["test_blocks"].append(b)
        self._say(f"test block {b['block']:2d} | {images:5d} imgs | acc {100*b['accuracy']:6.2f}% | "
                  f"{seconds:7.3f}s | {b['images_per_sec']:9.1f} img/s")

    def finish(self, n_train=None, **extra):
        """Fill the totals from the epoch and block logs. n_train defaults to images in epoch 1."""
        ep, bl = self.r["epochs_log"], self.r["test_blocks"]
        if not self.r["epochs"]:
            self.r["epochs"] = len(ep)
        n_train = n_train if n_train is not None else (ep[0]["images"] if ep else 0)
        train_s = sum(e["seconds"] for e in ep)
        test_s = sum(b["seconds"] for b in bl)
        n_test = sum(b["images"] for b in bl)
        correct = sum(b["correct"] for b in bl)
        self.r.update({
            "n_train": int(n_train), "n_test": int(n_test),
            "train_seconds": train_s,
            "train_images_per_sec": sum(e["images"] for e in ep) / train_s if train_s else 0.0,
            "test_seconds": test_s,
            "test_images_per_sec": n_test / test_s if test_s else 0.0,
            "test_ms_per_image": 1000 * test_s / n_test if n_test else 0.0,
            "test_accuracy": correct / n_test if n_test else 0.0,
            "finished": datetime.now().isoformat(timespec="seconds"),
        })
        if any("compute_seconds" in b for b in bl):
            cs = sum(b.get("compute_seconds", 0.0) for b in bl)
            self.r["test_compute_seconds"] = cs
            self.r["test_compute_images_per_sec"] = n_test / cs if cs else 0.0
        self.r.update(extra)
        return self.r

    def save(self, path, **extra):
        r = self.finish(**extra)
        problems = validate_results(r)
        self._say(f"done | train {r['train_seconds']:.1f}s | test {r['test_seconds']:.2f}s | "
                  f"test accuracy {100*r['test_accuracy']:.2f}% | n_train {r['n_train']} n_test {r['n_test']}")
        for p in problems:
            self._say(f"WARNING: {p}")
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(r, f, indent=2)
        with open(os.path.splitext(path)[0] + ".log", "w") as f:
            f.write("\n".join(self.lines) + "\n")
        if not self.quiet:
            print(f"saved {path} and {os.path.splitext(path)[0]}.log")
        return r


def validate_results(r):
    """Return a list of problems with a results dict (empty list = OK)."""
    problems = []
    for key, typ in REQUIRED_FIELDS.items():
        if key not in r:
            problems.append(f"missing field '{key}'")
        elif not isinstance(r[key], typ) or isinstance(r[key], bool):
            problems.append(f"field '{key}' has type {type(r[key]).__name__}")
    if problems:
        return problems
    if not 0 <= r["test_accuracy"] <= 1:
        problems.append(f"test_accuracy {r['test_accuracy']} should be between 0 and 1 (not a percentage)")
    size = r["image_size"]
    if size not in (IMG_SIZE, [IMG_SIZE, IMG_SIZE]):      # 64 or [64, 64] (Part III writes the list)
        problems.append(f"image_size {size} differs from the shared rule ({IMG_SIZE})")
    if r["data_split_fingerprint"] == "unknown":
        problems.append("data_split_fingerprint unknown - was data/ built with prepare_data.py?")
    if r["n_test"] <= 0 or r["n_train"] <= 0:
        problems.append("n_train and n_test must be positive")
    if r["train_seconds"] <= 0 or r["test_seconds"] <= 0:
        problems.append("train_seconds and test_seconds must be positive")
    if r["epochs_log"] and len(r["epochs_log"]) != r["epochs"]:
        problems.append(f"epochs_log has {len(r['epochs_log'])} entries but epochs = {r['epochs']}")
    if r["test_blocks"]:
        n = sum(b.get("images", b.get("n_images", 0)) for b in r["test_blocks"])   # Part III: n_images
        if n != r["n_test"]:
            problems.append(f"test_blocks cover {n} images but n_test = {r['n_test']}")
    return problems
