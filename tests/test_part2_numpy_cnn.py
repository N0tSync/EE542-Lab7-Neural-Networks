"""Numerical checks for the Part II forward pass and FC-only gradients.

Run from the repository root: python tests/test_part2_numpy_cnn.py
"""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from part2_numpy_cnn import NumPyCNN, conv2d, maxpool, mse_grad, mse_loss, train  # noqa: E402


class TestNumPyCNN(unittest.TestCase):
    def setUp(self):
        self.images = np.random.default_rng(42).uniform(0, 1, size=(3, 64, 64, 3))
        self.labels = np.array([0, 1, 0])

    def active_model(self):
        # Keep these gradient checks away from ReLU's nondifferentiable zero.
        model = NumPyCNN(seed=3)
        model.conv_filter[:] = np.abs(model.conv_filter)
        model.W_fc *= 0.1
        model.b_fc[:] = 0.5
        return model

    def test_vectorized_convolution_and_pool_match_reference(self):
        rng = np.random.default_rng(10)
        image = rng.normal(size=(5, 6, 3))
        kernel = rng.normal(size=(3, 3, 3))
        expected = np.empty((3, 4))
        for i in range(3):
            for j in range(4):
                expected[i, j] = np.sum(image[i:i + 3, j:j + 3] * kernel)
        np.testing.assert_allclose(conv2d(image, kernel), expected, atol=1e-12)
        expected_pool = np.array([[expected[:2, :2].max(), expected[:2, 2:4].max()]])
        np.testing.assert_allclose(maxpool(expected), expected_pool)
        maps = np.stack([expected, expected + 10])
        np.testing.assert_allclose(maxpool(maps), np.stack([expected_pool, expected_pool + 10]))

    def test_forward_shapes_and_single_batch_agreement(self):
        model = NumPyCNN(seed=1)
        scores = model.forward(self.images).copy()
        self.assertEqual(model.conv_out.shape, (3, 62, 62))
        self.assertEqual(model.relu_out.shape, (3, 62, 62))
        self.assertEqual(model.pooled.shape, (3, 31, 31))
        self.assertEqual(model.flat.shape, (3, 961))
        self.assertEqual(model.W_fc.shape, (961, 1))
        self.assertEqual(scores.shape, (3, 1))
        individual = np.concatenate([model.forward(image) for image in self.images])
        np.testing.assert_allclose(scores, individual, atol=1e-12)

    def test_mse_values_and_label_shapes(self):
        predictions = np.array([[0.25], [0.75]])
        labels = np.array([[0.0], [1.0]])
        self.assertAlmostEqual(mse_loss(labels, predictions), 0.0625)
        np.testing.assert_allclose(mse_grad(labels, predictions), [[0.25], [-0.25]])
        # Reject shapes that would accidentally broadcast to an NxN error array.
        with self.assertRaises(ValueError):
            mse_loss(labels.ravel(), predictions)
        model = self.active_model()
        model.forward(self.images)
        self.assertEqual(model.loss(self.labels), model.loss(self.labels[:, None]))

    def test_fc_gradients_match_finite_differences(self):
        model = self.active_model()
        model.forward(self.images)
        self.assertTrue(np.all(model.z_fc > 0.1))
        dW, db = model.backward(self.labels)
        self.assertGreater(np.linalg.norm(dW), 0)
        epsilon = 1e-6

        def evaluate():
            model.forward(self.images)
            return model.loss(self.labels)

        for parameter, gradient, indices in [
            (model.W_fc, dW, [0, 17, 480, 800, 960]),
            (model.b_fc, db, [0]),
        ]:
            for index in indices:
                original = parameter.flat[index]
                parameter.flat[index] = original + epsilon
                plus = evaluate()
                parameter.flat[index] = original - epsilon
                minus = evaluate()
                parameter.flat[index] = original
                numerical = (plus - minus) / (2 * epsilon)
                np.testing.assert_allclose(gradient.flat[index], numerical, rtol=1e-5, atol=1e-8)

    def test_inactive_output_relu_has_zero_gradient(self):
        model = NumPyCNN(seed=0)
        model.W_fc[:] = 0
        for bias in [-0.5, 0.0]:
            with self.subTest(bias=bias):
                model.b_fc[:] = bias
                np.testing.assert_array_equal(model.forward(self.images), 0)
                dW, db = model.backward(self.labels)
                np.testing.assert_array_equal(dW, 0)
                np.testing.assert_array_equal(db, 0)

    def test_update_reduces_loss_and_preserves_filter(self):
        model = self.active_model()
        kernel_before = model.conv_filter.copy()
        weights_before = model.W_fc.copy()
        bias_before = model.b_fc.copy()
        model.forward(self.images)
        loss_before = model.loss(self.labels)
        model.step(model.backward(self.labels), lr=1e-4)
        model.forward(self.images)
        self.assertLess(model.loss(self.labels), loss_before)
        self.assertFalse(np.array_equal(weights_before, model.W_fc))
        self.assertFalse(np.array_equal(bias_before, model.b_fc))
        np.testing.assert_array_equal(kernel_before, model.conv_filter)

    def test_initialization_is_reproducible(self):
        first, second, different = NumPyCNN(2), NumPyCNN(2), NumPyCNN(3)
        np.testing.assert_array_equal(first.conv_filter, second.conv_filter)
        np.testing.assert_array_equal(first.W_fc, second.W_fc)
        self.assertFalse(np.array_equal(first.conv_filter, different.conv_filter))

    def test_training_is_reproducible_and_reduces_loss(self):
        first, second = self.active_model(), self.active_model()
        kernel = first.conv_filter.copy()
        first.forward(self.images)
        initial_loss = first.loss(self.labels)
        history = train(first, self.images, self.labels, epochs=5, lr=1e-4,
                        seed=2, verbose=False)
        repeated = train(second, self.images, self.labels[:, None], epochs=5, lr=1e-4,
                         seed=2, verbose=False)
        self.assertEqual(history, repeated)
        self.assertEqual(len(history), 5)
        np.testing.assert_array_equal(first.W_fc, second.W_fc)
        np.testing.assert_array_equal(first.conv_filter, kernel)
        first.forward(self.images)
        self.assertLess(first.loss(self.labels), initial_loss)

    def test_training_detects_inactive_positive_class(self):
        images = np.stack([np.ones((64, 64, 3)), np.zeros((64, 64, 3))])
        for active_negatives in [False, True]:
            with self.subTest(active_negatives=active_negatives):
                model = NumPyCNN(3)
                model.conv_filter[:] = 0.1
                model.W_fc[:] = 0.001 if active_negatives else 0.0
                model.b_fc[:] = -0.1
                with self.assertRaisesRegex(RuntimeError, "positive-class outputs were inactive"):
                    train(model, images, [0, 1], epochs=2, lr=1e-5, verbose=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
