"""Behavior checks for Part III, without downloading real cat/dog images.

Run: python -m unittest discover -s tests -p 'test_part3_torch.py' -v
The generated images are fixtures, not evidence of cat/dog classification accuracy.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image
import torch
from torch import nn
from torch.utils.data import TensorDataset

# Support direct execution and unittest discovery from the repository root.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import prepare_data
from src import part3_torch as part3


class Part3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix="lab7-part3-")
        cls.data_dir = Path(cls.temp.name) / "data"
        rows = []
        for label in (0, 1):
            for index in range(10):
                # Include grayscale sources to exercise RGB conversion in preparation.
                image = Image.new("L", (80 + index, 72), color=30 + label * 180 + index)
                buffer = io.BytesIO()
                image.save(buffer, format="PNG")
                rows.append(("fixture.parquet", label * 10 + index, buffer.getvalue(), label))
        with contextlib.redirect_stdout(io.StringIO()):
            prepare_data.build_dataset(iter(rows), str(cls.data_dir), seed=0)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_model_can_learn_and_updates_convolutions(self):
        part3.seed_everything(0)
        model = part3.CatDogCNN()
        images = torch.zeros(8, 3, 64, 64)
        images[4:, 0] = 1.0
        labels = torch.tensor([0.0] * 4 + [1.0] * 4).view(-1, 1)
        before = model.conv1.weight.detach().clone()
        outputs = model(images)
        self.assertEqual(tuple(outputs.shape), (8, 1))
        self.assertTrue((outputs > 0).all())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        for _ in range(40):
            optimizer.zero_grad()
            loss = nn.functional.mse_loss(model(images), labels)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            outputs = model(images)
        self.assertLess(nn.functional.mse_loss(outputs, labels).item(), 0.02)
        self.assertTrue(((outputs > 0.5) == labels.bool()).all())
        self.assertFalse(torch.equal(model.conv1.weight, before))

    def test_block_boundaries_threshold_and_weighted_accuracy(self):
        class FixedScores(nn.Module):
            def forward(self, images):
                return images[:, :1]

        scores = torch.tensor([0, 0.5, 0.9, 0.1, 0.7, 0.3, 0.8, 0.5, 0.6, 0.2, 0.9])
        labels = torch.tensor([0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1])
        result = part3.evaluate(FixedScores(), TensorDataset(scores[:, None], labels),
                                torch.device("cpu"), batch_size=4, block_size=5)
        self.assertEqual([b["n_images"] for b in result["test_blocks"]], [5, 5, 1])
        self.assertEqual([b["start_index"] for b in result["test_blocks"]], [0, 5, 10])
        self.assertEqual(result["test_correct"], 8)
        self.assertAlmostEqual(result["test_accuracy"], 8 / 11)
        self.assertEqual(result["test_predicted_dogs"], 5)
        self.assertGreaterEqual(result["test_seconds"], result["test_compute_seconds"])

    def test_loading_preprocessing_and_reproducible_subsets(self):
        train, test, stats = part3.load_image_datasets(self.data_dir)
        self.assertEqual((len(train), len(test)), (16, 4))
        self.assertEqual(stats["train_total"], len(train))
        subset, indices = part3.select_subset(train, 3, seed=7)
        self.assertEqual(indices, part3.select_subset(train, 3, seed=7)[1])
        self.assertEqual([sum(train.targets[i] == label for i in indices) for label in (0, 1)], [3, 3])
        loaded = part3.preload(subset, batch_size=4, num_workers=0)
        image = loaded.tensors[0][0]
        self.assertEqual(tuple(image.shape), (3, 64, 64))
        self.assertEqual(image.dtype, torch.float32)
        self.assertTrue(0 <= image.min() <= image.max() <= 1)
        source = Image.open(train.samples[indices[0]][0]).convert("RGB")
        expected = source.resize((64, 64), resample=Image.Resampling.BILINEAR)
        self.assertTrue(torch.equal(image, part3.transforms.ToTensor()(expected)))

    def test_missing_image_is_rejected(self):
        path = next((self.data_dir / "train" / "cat").glob("*.jpg"))
        hidden = path.with_suffix(".hidden")
        path.rename(hidden)
        try:
            with self.assertRaisesRegex(ValueError, "manifest"):
                part3.load_image_datasets(self.data_dir)
        finally:
            hidden.rename(path)

    def test_cpu_run_saves_results_plot_and_reloadable_checkpoint(self):
        output = Path(self.temp.name) / "results" / "fixture_cpu.json"
        args = argparse.Namespace(data_dir=self.data_dir, device="cpu", epochs=2, lr=0.001,
                                  batch_size=4, test_block_size=3, seed=0, num_workers=0,
                                  cpu_threads=2, max_train_per_class=3, max_test_per_class=2,
                                  output=output)
        with contextlib.redirect_stdout(io.StringIO()):
            part3.run(args)
        result = json.loads(output.read_text())
        self.assertEqual((result["n_train"], result["n_test"]), (6, 4))
        self.assertEqual(len(result["epochs_log"]), 2)
        self.assertEqual(sum(b["n_images"] for b in result["test_blocks"]), 4)
        self.assertNotEqual(result["data_split_fingerprint"], result["source_data_split_fingerprint"])
        checkpoint = torch.load(result["checkpoint"], map_location="cpu", weights_only=True)
        model = part3.CatDogCNN()
        model.load_state_dict(checkpoint["model_state_dict"])
        self.assertEqual(checkpoint["class_to_idx"], {"cat": 0, "dog": 1})
        with Image.open(output.parent / output.stem / "loss.png") as plot:
            plot.verify()


if __name__ == "__main__":
    unittest.main()
