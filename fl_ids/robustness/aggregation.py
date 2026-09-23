"""Robust aggregation (component 6).

Trimmed mean over the trust filter's surviving client deltas (component
5), applied to deltas rather than raw weights, then added back onto the
round's starting global weights.
"""

from __future__ import annotations

import numpy as np


def trimmed_mean_delta(deltas: list[np.ndarray], trim_fraction: float) -> np.ndarray:
    """Coordinate-wise trimmed mean across surviving clients' (flattened) deltas.

    Args:
        deltas: Surviving clients' norm-clipped weight deltas (see
            `fl_ids.robustness.trust_filter.filter_client_deltas`).
        trim_fraction: Fraction trimmed from *each* end of the sorted
            values, per coordinate (config default 0.15: trims the lowest
            and highest 15% before averaging).

    Returns:
        The trimmed-mean delta vector.

    Raises:
        ValueError: If `deltas` is empty.
    """
    if not deltas:
        raise ValueError("Cannot compute a trimmed mean over zero surviving client deltas")

    stacked = np.stack(deltas)  # (n_clients, n_params)
    n = stacked.shape[0]
    k = int(np.floor(trim_fraction * n))
    if 2 * k >= n:
        # Too few surviving clients to trim without discarding everyone --
        # fall back to a plain mean. Deciding whether "too few clients
        # survived" should block the round entirely is the strategy's call
        # (component 7), not this function's.
        return stacked.mean(axis=0)

    sorted_stacked = np.sort(stacked, axis=0)
    return sorted_stacked[k : n - k].mean(axis=0)


def apply_delta(base_weights: list[np.ndarray], delta_flat: np.ndarray) -> list[np.ndarray]:
    """Reshape a flattened aggregated delta back into per-layer arrays and add it to the base weights.

    Args:
        base_weights: The round's starting global weights, per layer.
        delta_flat: A single flattened delta vector matching the total
            parameter count of `base_weights`.

    Returns:
        New per-layer weights: `base_weights[i] + delta_flat_reshaped[i]`.
    """
    new_weights = []
    offset = 0
    for w in base_weights:
        size = w.size
        delta_slice = delta_flat[offset : offset + size].reshape(w.shape)
        new_weights.append(w + delta_slice)
        offset += size
    return new_weights
