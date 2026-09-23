"""Tests for robust aggregation (component 6): trimmed mean + delta application."""

from __future__ import annotations

import numpy as np
import pytest

from fl_ids.robustness.aggregation import apply_delta, trimmed_mean_delta


def test_trimmed_mean_matches_plain_mean_with_zero_trim():
    deltas = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
    result = trimmed_mean_delta(deltas, trim_fraction=0.0)
    assert np.allclose(result, np.mean(deltas, axis=0))


def test_trimmed_mean_excludes_extreme_outlier():
    deltas = [np.array([1.0]), np.array([1.1]), np.array([0.9]), np.array([1.0]), np.array([1000.0])]
    # trim_fraction=0.2 with n=5 -> k=1, trims 1 from each end.
    result = trimmed_mean_delta(deltas, trim_fraction=0.2)
    assert result[0] < 2.0  # the 1000.0 outlier should be trimmed away


def test_trimmed_mean_falls_back_to_plain_mean_when_too_few_clients_to_trim():
    deltas = [np.array([1.0]), np.array([2.0])]
    result = trimmed_mean_delta(deltas, trim_fraction=0.4)  # k=0 for n=2, but guard for k*2>=n cases too
    assert np.allclose(result, [1.5])


def test_trimmed_mean_raises_on_empty_input():
    with pytest.raises(ValueError):
        trimmed_mean_delta([], trim_fraction=0.15)


def test_apply_delta_reshapes_and_adds():
    base_weights = [np.zeros((2, 2)), np.zeros(3)]
    delta_flat = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])

    new_weights = apply_delta(base_weights, delta_flat)

    assert np.allclose(new_weights[0], [[1.0, 2.0], [3.0, 4.0]])
    assert np.allclose(new_weights[1], [5.0, 6.0, 7.0])


def test_apply_delta_preserves_shapes():
    base_weights = [np.zeros((3, 4)), np.zeros((2,)), np.zeros((1, 1))]
    total_params = sum(w.size for w in base_weights)
    delta_flat = np.arange(total_params, dtype=np.float32)

    new_weights = apply_delta(base_weights, delta_flat)

    for base, new in zip(base_weights, new_weights):
        assert base.shape == new.shape
