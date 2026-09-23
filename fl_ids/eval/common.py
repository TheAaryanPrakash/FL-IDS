"""Shared setup for Phase 6's evaluation sweeps (poisoning resistance, ablation).

Both sweeps need the same ingredients — a server-held calibration set, a
bootstrap boosting model, per-client federated data, and a held-out test
set — built once and reused across every sweep point/ablation variant so
results are comparable (same data, same seed, per CLAUDE.md's ablation
requirement).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fl_ids.data.pipeline import load_and_encode, partition_and_normalize_clients
from fl_ids.models.autoencoder import Autoencoder, compute_anomaly_threshold, reconstruction_error, set_weights
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSetup:
    client_data: dict[int, dict[str, np.ndarray]]
    boosting_model: BoostingClassifier
    class_names: list[str]
    benign_class: int
    input_dim: int
    X_test_raw: np.ndarray
    X_test_norm: np.ndarray
    y_test: np.ndarray


def build_evaluation_setup(
    config: Config,
    real_csv_path: str | Path,
    seed: int,
    pool_subsample_size: int = 60_000,
) -> EvaluationSetup:
    """Build the shared calibration/federated/test data every sweep point reuses.

    Uses real Edge-IIoTset data (subsampled after the calibration split
    purely to bound sweep runtime across many repeated FL training runs —
    still real traffic, real feature values, real per-client skew, per
    the same pattern established in Phase 5).

    Args:
        config: Full project config (data split fractions, boosting
            hyperparameters, cascade confidence threshold).
        real_csv_path: Path to `DNN-EdgeIIoT-dataset.csv`.
        seed: Random seed, for reproducible splitting/partitioning.
        pool_subsample_size: Caps the federated-client pool size.

    Returns:
        An `EvaluationSetup` with the calibration-trained boosting model,
        per-client federated data, and a held-out test set.
    """
    X, y, label_encoder, feature_names, benign_class = load_and_encode(real_csv_path)
    X_calib, X_pool, y_calib, y_pool = train_test_split(
        X, y, train_size=config.boosting.calibration_fraction, random_state=seed, stratify=y
    )

    if len(X_pool) > pool_subsample_size:
        X_pool, _, y_pool, _ = train_test_split(
            X_pool, y_pool, train_size=pool_subsample_size, random_state=seed, stratify=y_pool
        )

    boosting_model = BoostingClassifier(
        config.boosting,
        num_classes=len(label_encoder.classes_),
        benign_class=benign_class,
        seed=seed,
        confidence_threshold=config.cascade.confidence_threshold,
    )
    boosting_model.train(X_calib, y_calib)

    # A held-out test set, disjoint from both calibration and the
    # federated pool (drawn from the same pool split before per-client
    # partitioning, mirroring build_server_and_federated_dataset's
    # disjointness-by-construction).
    X_pool, X_test_raw, y_pool, y_test = train_test_split(
        X_pool, y_pool, test_size=0.15, random_state=seed, stratify=y_pool
    )

    client_data = partition_and_normalize_clients(X_pool, y_pool, benign_class, config.data, seed)

    # Normalize the held-out test set the same way every client normalizes
    # its own data (its own scaler) -- for cascade/autoencoder evaluation,
    # any one client's scaler is as good as another's for this purpose, so
    # we use client 0's, applied consistently across the whole test set.
    ref_scaler = StandardScaler().fit(client_data[0]["X_raw"])
    X_test_norm = ref_scaler.transform(X_test_raw).astype(np.float32)

    logger.info(
        "Evaluation setup: %d calibration, %d federated pool (subsampled), %d test rows",
        len(X_calib),
        len(X_pool),
        len(X_test_raw),
    )

    return EvaluationSetup(
        client_data=client_data,
        boosting_model=boosting_model,
        class_names=list(label_encoder.classes_),
        benign_class=benign_class,
        input_dim=X.shape[1],
        X_test_raw=X_test_raw.astype(np.float32),
        X_test_norm=X_test_norm,
        y_test=y_test,
    )


def calibrate_threshold_from_final_weights(
    final_weights: list[np.ndarray],
    setup: EvaluationSetup,
    config: Config,
) -> tuple[Autoencoder, float]:
    """Load simulation output weights into a fresh Autoencoder and calibrate its threshold.

    Args:
        final_weights: The simulation's final global autoencoder weights.
        setup: The shared `EvaluationSetup` (for input_dim and client val_benign data).
        config: Full project config (autoencoder architecture, anomaly percentile).

    Returns:
        (autoencoder, anomaly_threshold), ready for cascade/standalone evaluation.
    """
    autoencoder = Autoencoder(setup.input_dim, config.autoencoder.hidden_dims, config.autoencoder.bottleneck_dim)
    set_weights(autoencoder, final_weights)

    val_benign = np.concatenate(
        [data["X_val_benign"] for data in setup.client_data.values() if len(data["X_val_benign"]) > 0]
    )
    errors = reconstruction_error(autoencoder, val_benign)
    threshold = compute_anomaly_threshold(errors, config.autoencoder.anomaly_percentile)
    return autoencoder, threshold


def select_malicious_clients(client_ids: list[int], fraction: float, seed: int) -> set[int]:
    """Deterministically select a fraction of client IDs as malicious.

    Args:
        client_ids: All client IDs in the federation.
        fraction: Fraction to mark malicious, in [0, 1].
        seed: Random seed, for reproducible selection.

    Returns:
        The selected malicious client IDs.
    """
    rng = np.random.default_rng(seed)
    n_malicious = round(fraction * len(client_ids))
    return set(rng.choice(client_ids, size=n_malicious, replace=False).tolist())
