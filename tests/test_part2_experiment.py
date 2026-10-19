"""Checks for bounded inference, uint8 training and the formal experiment CLI.

Run: python tests/test_part2_experiment.py
All image fixtures and CLI artifacts are created in temporary directories.
"""

from collections import Counter
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from data_utils import load_images, manifest_fingerprint, read_data_info  # noqa: E402
from part2_numpy_cnn import NumPyCNN, evaluate, parse_args, train  # noqa: E402


class PixelScores:
    """Read predetermined scores from pixels to test metrics independently of CNN weights."""

    def __init__(self):
        self.batch_sizes = []

    def forward(self, images):
        self.batch_sizes.append(len(images))
        return images[:, 0, 0, :1]


class EvaluationTests(unittest.TestCase):
    def test_tail_blocks_threshold_and_weighted_accuracy(self):
        values = np.array([0, 0.5, 0.9, 0.1, 0.7, 0.3, 0.8, 0.5, 0.6, 0.2, 0.9])
        labels = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1])
        images = np.zeros((len(values), 64, 64, 3))
        images[:, 0, 0, 0] = values
        model = PixelScores()
        result, scores = evaluate(model, images, labels, block_size=5, batch_size=4)
        self.assertEqual([b["n_images"] for b in result["test_blocks"]], [5, 5, 1])
        self.assertEqual([b["start_index"] for b in result["test_blocks"]], [0, 5, 10])
        self.assertEqual(model.batch_sizes, [4, 1, 4, 1, 1])
        self.assertEqual(result["test_correct"], 8)
        self.assertAlmostEqual(result["test_accuracy"], 8 / 11)
        self.assertEqual(result["test_predicted_dogs"], 5)
        self.assertEqual(result["test_predicted_cats"], 6)
        self.assertEqual(sum(map(sum, result["confusion_matrix"])), 11)
        self.assertAlmostEqual(result["test_mse"], float(np.mean((values - labels) ** 2)))
        self.assertGreaterEqual(result["test_seconds"], result["test_compute_seconds"])
        self.assertGreaterEqual(result["test_seconds"], sum(b["seconds"] for b in result["test_blocks"]))
        np.testing.assert_array_equal(scores, values)

    def test_uint8_evaluation_is_normalized_once(self):
        images = np.zeros((3, 64, 64, 3), dtype=np.uint8)
        images[:, 0, 0, 0] = [0, 128, 255]
        result, scores = evaluate(PixelScores(), images, [0, 1, 1], block_size=2, batch_size=1)
        np.testing.assert_allclose(scores, np.array([0, 128, 255]) / 255.0)
        self.assertEqual(result["test_accuracy"], 1.0)
        self.assertEqual(images.dtype, np.uint8)

    def test_uint8_sgd_matches_explicit_float_normalization(self):
        images = np.random.default_rng(1).integers(0, 256, (3, 64, 64, 3), dtype=np.uint8)
        first, second = NumPyCNN(3), NumPyCNN(3)
        for model in [first, second]:
            model.conv_filter[:] = np.abs(model.conv_filter)
            model.W_fc *= 0.1
            model.b_fc[:] = 0.5
        history = train(first, images, [0, 1, 0], epochs=2, lr=1e-4, seed=1, verbose=False)
        normalized = images.astype(float) / 255.0
        repeated = train(second, normalized, [0, 1, 0], epochs=2, lr=1e-4, seed=1, verbose=False)
        self.assertEqual(history, repeated)
        np.testing.assert_array_equal(first.W_fc, second.W_fc)
        np.testing.assert_array_equal(first.b_fc, second.b_fc)


class ExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="lab7-numpy-experiment-")
        cls.root = Path(cls.temp.name)
        cls.data = cls.root / "data"
        records = []
        rng = np.random.default_rng(1)
        for split in ["train", "test"]:
            for label, name in enumerate(["cat", "dog"]):
                folder = cls.data / split / name
                folder.mkdir(parents=True)
                for index in range(3):
                    filename = f"{split}_{name}_{index}.png"
                    pixels = (rng.uniform(0, 1, (64, 64, 3)) * 255).astype(np.uint8)
                    Image.fromarray(pixels).save(folder / filename)
                    records.append(dict(filename=filename, split=split, label=str(label),
                                        **{"class": name}, width=64, height=64))
        with (cls.data / "split_manifest.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        counts = dict(Counter(f"{r['split']}/{r['class']}" for r in records))
        stats = dict(dataset="generated_fixture", images_used=12, train_total=6, test_total=6,
                     counts=counts, split_fingerprint=manifest_fingerprint(records))
        (cls.data / "stats.json").write_text(json.dumps(stats))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_cli_writes_consistent_results_and_companion_files(self):
        output = self.root / "results"
        env = dict(os.environ, MPLCONFIGDIR=str(self.root / "matplotlib"),
                   PYTHONDONTWRITEBYTECODE="1")
        completed = subprocess.run(
            [sys.executable, str(ROOT / "src/part2_numpy_cnn.py"),
             "--data-dir", str(self.data), "--epochs", "2", "--seed", "3",
             "--max-train-per-class", "2", "--max-test-per-class", "2",
             "--test-block-size", "3", "--inference-batch-size", "2",
             "--output-dir", str(output)],
            capture_output=True, text=True, timeout=30, env=env, cwd=ROOT)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        result = json.loads((output / "part2_numpy_results.json").read_text())
        self.assertEqual((result["n_train"], result["n_test"]), (4, 4))
        self.assertEqual(result["run_type"], "subset_validation")
        self.assertEqual(result["device"], "cpu")
        self.assertTrue(result["device_name"])
        self.assertNotEqual(result["data_split_fingerprint"], result["source_data_split_fingerprint"])
        self.assertEqual(result["preloaded_image_bytes"], 8 * 64 * 64 * 3)
        self.assertEqual([b["n_images"] for b in result["test_blocks"]], [3, 1])
        self.assertEqual(sum(b["correct"] for b in result["test_blocks"]), result["test_correct"])
        self.assertEqual(len(result["epochs_log"]), 2)
        self.assertGreaterEqual(result["train_seconds"], sum(e["seconds"] for e in result["epochs_log"]))
        self.assertGreaterEqual(result["test_seconds"], result["test_compute_seconds"])
        self.assertAlmostEqual(result["train_images_per_sec"], 8 / result["train_seconds"])
        with (output / "part2_numpy_test_predictions.csv").open() as stream:
            predictions = list(csv.DictReader(stream))
        correct = sum(int(r["label"]) == int(r["prediction"]) for r in predictions)
        self.assertEqual(len(predictions), 4)
        self.assertEqual(result["test_accuracy"], correct / 4)
        with np.load(output / "part2_numpy_model.npz", allow_pickle=False) as checkpoint:
            original = NumPyCNN(3)
            np.testing.assert_array_equal(checkpoint["conv_filter"], original.conv_filter)
            self.assertEqual(checkpoint["W_fc"].shape, (961, 1))
            self.assertFalse(np.array_equal(checkpoint["W_fc"], original.W_fc))
        with Image.open(output / "part2_numpy_loss.png") as image:
            image.verify()
        with (output / "part2_numpy_epochs.csv").open() as stream:
            self.assertEqual(len(list(csv.DictReader(stream))), 2)
        self.assertIn("Results saved", (output / "part2_numpy_run.log").read_text())

    def test_loader_storage_and_missing_image_detection(self):
        _, records = read_data_info(self.data)
        pixels, labels = load_images(self.data, records, uint8=True)
        normalized, repeated_labels = load_images(self.data, records)
        self.assertEqual(pixels.dtype, np.uint8)
        self.assertEqual(normalized.dtype, np.float32)
        np.testing.assert_array_equal(normalized, pixels.astype(np.float32) / 255.0)
        np.testing.assert_array_equal(labels, repeated_labels)
        missing = self.data / "train/cat/train_cat_0.png"
        hidden = missing.with_suffix(".hidden")
        missing.rename(hidden)
        try:
            with self.assertRaisesRegex(ValueError, "manifest"):
                read_data_info(self.data)
        finally:
            hidden.rename(missing)

    def test_subset_defaults_do_not_target_formal_results(self):
        full = parse_args([])
        smoke = parse_args(["--max-train-per-class", "2"])
        self.assertEqual(full.output, Path("results/part2_numpy_results.json"))
        self.assertEqual(smoke.output, Path("results/part2_smoke/part2_numpy_results.json"))
        self.assertEqual(full.data_seed, full.seed)


if __name__ == "__main__":
    unittest.main(verbosity=2)
