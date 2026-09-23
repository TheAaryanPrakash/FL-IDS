"""Client data file I/O for real multi-process Flower runs.

Real client processes (`python -m fl_ids.fl.client`) don't share memory
with whatever generated the federated dataset, so each client's slice
needs to be handed off via a file. Used by orchestration (Phase A) and by
the Phase 3 multi-process integration test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

CLIENT_DATA_KEYS = ("X", "X_raw", "y", "X_val_benign", "y_val_benign", "X_test", "X_test_raw", "y_test")


def save_client_data(path: str | Path, data: dict[str, np.ndarray]) -> None:
    """Save one client's data dict (component 1's per-client output shape) to `.npz`."""
    np.savez(path, **{key: data[key] for key in CLIENT_DATA_KEYS})


def load_client_data(path: str | Path) -> dict[str, np.ndarray]:
    """Load a client data dict previously written by `save_client_data`."""
    with np.load(path) as npz:
        return {key: npz[key] for key in CLIENT_DATA_KEYS}
