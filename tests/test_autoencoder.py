"""Tests for the client-side autoencoder (component 3)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from fl_ids.models.autoencoder import (
    Autoencoder,
    compute_anomaly_threshold,
    get_weights,
    reconstruction_error,
    reconstruction_error_histogram,
    set_weights,
    train_autoencoder,
)
from fl_ids.utils.config import AutoencoderConfig


def _ae_config(**overrides) -> AutoencoderConfig:
    defaults = dict(
        bottleneck_dim=4,
        hidden_dims=[16, 8],
        learning_rate=0.01,
        local_epochs=10,
        batch_size=32,
        anomaly_percentile=97,
        reconstruction_error_bins=20,
        reconstruction_error_range=(0.0, 5.0),
    )
    defaults.update(overrides)
    return AutoencoderConfig(**defaults)


def test_autoencoder_forward_preserves_shape():
    model = Autoencoder(input_dim=10, hidden_dims=[16, 8], bottleneck_dim=4)
    X = torch.randn(5, 10)
    out = model(X)
    assert out.shape == X.shape


def test_get_set_weights_round_trip():
    model_a = Autoencoder(input_dim=6, hidden_dims=[8], bottleneck_dim=3)
    model_b = Autoencoder(input_dim=6, hidden_dims=[8], bottleneck_dim=3)

    weights_a = get_weights(model_a)
    set_weights(model_b, weights_a)
    weights_b = get_weights(model_b)

    for wa, wb in zip(weights_a, weights_b):
        assert np.allclose(wa, wb)


def test_training_reduces_reconstruction_error():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(500, 10)).astype(np.float32)

    model = Autoencoder(input_dim=10, hidden_dims=[16, 8], bottleneck_dim=4)
    error_before = reconstruction_error(model, X).mean()

    train_autoencoder(model, X, _ae_config(local_epochs=20, batch_size=64), seed=0)
    error_after = reconstruction_error(model, X).mean()

    assert error_after < error_before


def test_train_autoencoder_returns_decreasing_epoch_losses():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, size=(300, 8)).astype(np.float32)
    model = Autoencoder(input_dim=8, hidden_dims=[16, 8], bottleneck_dim=4)

    losses = train_autoencoder(model, X, _ae_config(local_epochs=15, batch_size=32), seed=1)

    assert len(losses) == 15
    assert losses[-1] < losses[0]


def test_reconstruction_error_empty_input_returns_empty_array():
    model = Autoencoder(input_dim=5, hidden_dims=[8], bottleneck_dim=2)
    errors = reconstruction_error(model, np.empty((0, 5), dtype=np.float32))
    assert errors.shape == (0,)


def test_compute_anomaly_threshold_uses_percentile():
    errors = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
    threshold = compute_anomaly_threshold(errors, percentile=80)
    assert threshold == pytest.approx(np.percentile(errors, 80))


def test_compute_anomaly_threshold_empty_returns_inf():
    threshold = compute_anomaly_threshold(np.array([]), percentile=97)
    assert threshold == float("inf")


def test_reconstruction_error_histogram_fixed_bins_and_range():
    errors = np.array([0.5, 1.5, 1.5, 4.9, 10.0])  # 10.0 is out of range, should be excluded
    hist = reconstruction_error_histogram(errors, bins=5, value_range=(0.0, 5.0))
    assert hist.shape == (5,)
    assert hist.sum() == 4  # the out-of-range value is dropped by np.histogram's range
    assert hist.dtype == np.int64


def test_reconstruction_error_histogram_shape_independent_of_input_size():
    hist_small = reconstruction_error_histogram(np.array([1.0, 2.0]), bins=10, value_range=(0.0, 5.0))
    hist_large = reconstruction_error_histogram(np.full(1000, 2.0), bins=10, value_range=(0.0, 5.0))
    assert hist_small.shape == hist_large.shape == (10,)
