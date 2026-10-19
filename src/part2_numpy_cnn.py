#!/usr/bin/env python3
"""EE 542 Lab 7, Part II: NumPy CNN forward pass and FC-only updates.

Architecture: 64x64x3 -> valid convolution -> ReLU -> 2x2 max pool
              -> 961 features -> fully connected -> ReLU.
The single 3x3x3 convolution filter is fixed; only W_fc and b_fc are trained.
Loss is ordinary MSE (without Part I's factor of 0.5).
An output ReLU has zero gradient at nonpositive logits; some initializations
can therefore produce all-zero scores and stall learning.

Run a synthetic forward/update demonstration: python src/part2_numpy_cnn.py
Real-image validation: python scripts/validate_part2_numpy.py
Formal experiments, block inference and timing come later.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def relu(x):
    return np.maximum(0.0, x)


def relu_derivative(x):
    """Use derivative 0 at the ReLU kink x = 0, as in the handout."""
    return (x > 0).astype(float)


def _mse_error(y_true, y_pred):
    targets = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    if targets.shape != predictions.shape or predictions.size == 0:
        raise ValueError("MSE targets and predictions must have the same nonempty shape")
    return predictions - targets


def mse_loss(y_true, y_pred):
    diff = _mse_error(y_true, y_pred)
    return float(np.mean(diff * diff))


def mse_grad(y_true, y_pred):
    diff = _mse_error(y_true, y_pred)
    return 2.0 * diff / diff.size


def conv2d(images, kernel):
    """Valid, stride-1 cross-correlation, summing all RGB channels.

    Input: (H, W, C) or (N, H, W, C); kernel: (KH, KW, C).
    Output: (H-KH+1, W-KW+1) or (N, H-KH+1, W-KW+1).
    A window view avoids copying every image patch into a separate array.
    """
    images = np.asarray(images, dtype=float)
    kernel = np.asarray(kernel, dtype=float)
    if images.ndim not in (3, 4) or kernel.ndim != 3:
        raise ValueError("Expected HWC/NHWC images and an HWC kernel")
    kh, kw, channels = kernel.shape
    if (min(kh, kw, channels) < 1 or images.shape[-1] != channels
            or images.shape[-3] < kh or images.shape[-2] < kw):
        raise ValueError("Kernel dimensions must fit the image and match its channels")
    windows = sliding_window_view(images, (kh, kw), axis=(-3, -2))
    # Window layout: (..., output_height, output_width, channels, kh, kw).
    return np.einsum("...hwcij,ijc->...hw", windows, kernel, optimize=False)


def maxpool(images, size=2):
    """Non-overlapping max pooling; discard incomplete bottom/right blocks."""
    images = np.asarray(images, dtype=float)
    if images.ndim not in (2, 3):
        raise ValueError("Expected HW/NHW feature maps")
    if not isinstance(size, (int, np.integer)) or size < 1:
        raise ValueError("Pool size must be a positive integer")
    h, w = images.shape[-2:]
    new_h, new_w = h // size, w // size
    if min(new_h, new_w) < 1:
        raise ValueError("Pool size must fit the feature map")
    cropped = images[..., :new_h * size, :new_w * size]
    blocks = cropped.reshape(*images.shape[:-2], new_h, size, new_w, size)
    return blocks.max(axis=(-3, -1))


class NumPyCNN:
    """One fixed convolution filter and one trainable output neuron.

    forward accepts (64, 64, 3) or (N, 64, 64, 3), returning (N, 1).
    Layer values are cached for inspection and for backward's FC gradients.
    Output is a nonnegative score; later classification uses score > 0.5.
    """

    input_shape = (64, 64, 3)
    n_features = 31 * 31

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.conv_filter = rng.normal(0.0, 0.1, size=(3, 3, 3))
        self.W_fc = rng.normal(0.0, 0.01, size=(self.n_features, 1))
        self.b_fc = np.zeros((1,))
        self.y_pred = None

    def forward(self, images):
        images = np.asarray(images, dtype=float)
        if images.ndim == 3:
            images = images[np.newaxis, ...]
        if images.ndim != 4 or images.shape[1:] != self.input_shape or len(images) == 0:
            raise ValueError("Expected a 64x64x3 image or a nonempty NHWC batch")
        self.conv_out = conv2d(images, self.conv_filter)
        self.relu_out = relu(self.conv_out)
        self.pooled = maxpool(self.relu_out)
        self.flat = self.pooled.reshape(len(images), self.n_features)
        self.z_fc = self.flat @ self.W_fc + self.b_fc
        self.y_pred = relu(self.z_fc)
        return self.y_pred

    def _targets(self, y):
        if self.y_pred is None:
            raise RuntimeError("Call forward before loss or backward")
        targets = np.asarray(y, dtype=float)
        if targets.ndim == 0:
            targets = targets.reshape(1, 1)
        elif targets.ndim == 1:
            targets = targets[:, np.newaxis]
        if targets.shape != self.y_pred.shape:
            raise ValueError("Expected one target per image, with shape (N,) or (N, 1)")
        return targets

    def loss(self, y):
        return mse_loss(self._targets(y), self.y_pred)

    def backward(self, y):
        grad_z = mse_grad(self._targets(y), self.y_pred) * relu_derivative(self.z_fc)
        dW_fc = self.flat.T @ grad_z
        db_fc = grad_z.sum(axis=0)
        # No convolution gradients: the feature extractor remains fixed.
        return dW_fc, db_fc

    def step(self, grads, lr=1e-3):
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("Learning rate must be finite and positive")
        dW_fc, db_fc = grads
        if dW_fc.shape != self.W_fc.shape or db_fc.shape != self.b_fc.shape:
            raise ValueError("Gradient shapes must match W_fc and b_fc")
        self.W_fc -= lr * dW_fc
        self.b_fc -= lr * db_fc


def train(model, images, labels, epochs=20, lr=1e-3, seed=0, verbose=True, log=None):
    """Shuffled per-image SGD, matching the handout's training scheme.

    Each epoch's loss and accuracy are online averages, measured immediately
    before each sample's update. Only the FC parameters are updated.
    """
    images = np.asarray(images, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if (images.ndim != 4 or images.shape[1:] != model.input_shape
            or len(images) == 0 or labels.shape != (len(images),)):
        raise ValueError("Expected NHWC images and one label per image")
    if not np.isfinite(images).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Images must be finite and labels must be 0 or 1")
    if not isinstance(epochs, (int, np.integer)) or epochs < 1:
        raise ValueError("Epochs must be a positive integer")
    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive")
    rng = np.random.default_rng(seed)
    positive_count = int(np.count_nonzero(labels == 1))
    history = []
    report = log if log is not None else print
    for epoch in range(1, epochs + 1):
        total_loss, correct, active_positive = 0.0, 0, 0
        for index in rng.permutation(len(images)):
            score = float(model.forward(images[index])[0, 0])
            loss = model.loss(labels[index])
            if not np.isfinite(loss):
                raise FloatingPointError(f"Non-finite MSE in epoch {epoch}, image {index}")
            total_loss += loss
            correct += int((score > 0.5) == labels[index])
            active_positive += int(labels[index] == 1 and model.z_fc[0, 0] > 0)
            model.step(model.backward(labels[index]), lr=lr)
        if positive_count and active_positive == 0:
            raise RuntimeError(
                f"All positive-class outputs were inactive in epoch {epoch}; "
                "the output ReLU blocks their gradients. Try a different model seed."
            )
        record = dict(epoch=epoch, loss=total_loss / len(images),
                      train_accuracy=correct / len(images),
                      active_positive_fraction=(active_positive / positive_count
                                                if positive_count else None))
        history.append(record)
        if verbose:
            report(f"epoch {epoch:3d}/{epochs} online_mse={record['loss']:.6f} "
                   f"online_accuracy={record['train_accuracy']:.2%}")
    return history


def main():
    """Show one forward pass and one gradient update on synthetic images."""
    images = np.random.default_rng(1).uniform(0.0, 1.0, size=(2, 64, 64, 3))
    labels = np.array([0, 1])
    # Seed 3 gives active output ReLUs for this demonstration's synthetic data.
    model = NumPyCNN(seed=3)
    original_filter = model.conv_filter.copy()
    model.forward(images)
    print("Synthetic demonstration: data seed=1, model seed=3")
    print("Synthetic images:", images.shape)
    print("Convolution:", model.conv_out.shape)
    print("Pooling:", model.pooled.shape)
    print("Flatten:", model.flat.shape)
    print("W_fc:", model.W_fc.shape, "b_fc:", model.b_fc.shape)
    print("Output scores:", model.y_pred.ravel())
    print(f"MSE before update: {model.loss(labels):.8f}")
    model.step(model.backward(labels), lr=1e-3)
    model.forward(images)
    print(f"MSE after update:  {model.loss(labels):.8f}")
    print("Convolution filter unchanged:", np.array_equal(original_filter, model.conv_filter))


if __name__ == "__main__":
    main()
