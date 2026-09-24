"""Shared setup for Phase 6's evaluation sweeps (poisoning resistance, ablation, zero-day).

Every sweep needs the same ingredients — a server-held calibration set, a
bootstrap boosting model, per-client federated data, and a held-out test
set — built once and reused across every sweep point/ablation variant so
results are comparable (same data, same seed, per CLAUDE.md's ablation
requirement).

**Evaluation is per client, like deployment.** Each client normalizes its
own traffic with a scaler fit on its own training rows (component 1) and
calibrates its own anomaly threshold on its own benign validation slice
(component 3). The test set is therefore the union of the clients' own
held-out test slices, each row normalized with its client's scaler and
scored against that client's threshold — never one client's scaler
applied to everyone, or a threshold pooled across differently-normalized
data (which is what an earlier version did, mixing the two scales).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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
    """Everything a sweep point needs, shared across variants.

    `X_test_raw`/`X_test_norm`/`y_test`/`test_client_ids` are the clients'
    own test slices concatenated, row-aligned: row i came from client
    `test_client_ids[i]` and was normalized with that client's scaler.
    """

    client_data: dict[int, dict[str, np.ndarray]]
    boosting_model: BoostingClassifier
    class_names: list[str]
    benign_class: int
    input_dim: int
    X_test_raw: np.ndarray
    X_test_norm: np.ndarray
    y_test: np.ndarray
    test_client_ids: np.ndarray
    # Each client's scaler, so rows from outside its data (the zero-day
    # experiment's held-out attack rows) are normalized the way that
    # client would normalize them.
    client_scalers: dict[int, StandardScaler] = field(default_factory=dict)
    # Classes removed from the calibration set and the federated pool
    # before training (the zero-day experiment's "never seen" attack types).
    excluded_classes: frozenset[int] = frozenset()


def concat_client_test_slices(
    client_data: dict[int, dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate every client's own test slice into one row-aligned test set.

    Args:
        client_data: Component 1's per-client output.

    Returns:
        (X_test_raw, X_test_norm, y_test, test_client_ids).
    """
    ids = sorted(client_data)
    X_raw = np.concatenate([client_data[cid]["X_test_raw"] for cid in ids]).astype(np.float32)
    X_norm = np.concatenate([client_data[cid]["X_test"] for cid in ids]).astype(np.float32)
    y = np.concatenate([client_data[cid]["y_test"] for cid in ids])
    client_ids = np.concatenate([np.full(len(client_data[cid]["y_test"]), cid) for cid in ids])
    return X_raw, X_norm, y, client_ids


def build_evaluation_setup(
    config: Config,
    real_csv_path: str | Path,
    seed: int,
    pool_subsample_size: int = 60_000,
) -> EvaluationSetup:
    """Load the real dataset and build the shared evaluation setup from it.

    Thin wrapper over `build_evaluation_setup_from_arrays` for callers that
    only need one setup; callers building several (e.g. one per held-out
    attack class) should call `load_and_encode` once and reuse the arrays.

    Args:
        config: Full project config.
        real_csv_path: Path to `DNN-EdgeIIoT-dataset.csv`.
        seed: Random seed, for reproducible splitting/partitioning.
        pool_subsample_size: Caps the federated-client pool size.

    Returns:
        An `EvaluationSetup` (see `build_evaluation_setup_from_arrays`).
    """
    X, y, label_encoder, _feature_names, benign_class = load_and_encode(real_csv_path)
    return build_evaluation_setup_from_arrays(
        X, y, list(label_encoder.classes_), benign_class, config, seed, pool_subsample_size
    )


def build_evaluation_setup_from_arrays(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    benign_class: int,
    config: Config,
    seed: int,
    pool_subsample_size: int = 60_000,
    excluded_classes: frozenset[int] = frozenset(),
) -> EvaluationSetup:
    """Build the shared calibration/federated/test data every sweep point reuses.

    Uses real Edge-IIoTset data (subsampled after the calibration split
    purely to bound sweep runtime across many repeated FL training runs —
    still real traffic, real feature values, real per-client skew, per
    the same pattern established in Phase 5).

    `excluded_classes` supports the zero-day experiment: those classes are
    dropped from the calibration set (so boosting never learns them) and
    from the federated pool before partitioning (so no client ever holds
    them — not in training data, not in its scaler's statistics, not in
    its test slice).

    Args:
        X: Full raw-scale feature matrix (`load_and_encode` output).
        y: Integer class labels.
        class_names: Class names in class-index order.
        benign_class: Integer class index corresponding to "Normal".
        config: Full project config (data split fractions, boosting
            hyperparameters, cascade confidence threshold).
        seed: Random seed, for reproducible splitting/partitioning.
        pool_subsample_size: Caps the federated-client pool size.
        excluded_classes: Class indices to withhold from all training data.

    Returns:
        An `EvaluationSetup` with the calibration-trained boosting model,
        per-client federated data, and the clients' concatenated test slices.

    Raises:
        ValueError: If `excluded_classes` contains the benign class.
    """
    if benign_class in excluded_classes:
        raise ValueError("The benign class can't be excluded: every stage is calibrated on benign traffic")

    X_calib, X_pool, y_calib, y_pool = train_test_split(
        X, y, train_size=config.boosting.calibration_fraction, random_state=seed, stratify=y
    )

    if len(X_pool) > pool_subsample_size:
        X_pool, _, y_pool, _ = train_test_split(
            X_pool, y_pool, train_size=pool_subsample_size, random_state=seed, stratify=y_pool
        )

    if excluded_classes:
        excluded = np.array(sorted(excluded_classes))
        calib_keep = ~np.isin(y_calib, excluded)
        pool_keep = ~np.isin(y_pool, excluded)
        X_calib, y_calib = X_calib[calib_keep], y_calib[calib_keep]
        X_pool, y_pool = X_pool[pool_keep], y_pool[pool_keep]
        logger.info(
            "Excluded classes %s from training: removed %d calibration rows, %d federated-pool rows",
            [class_names[c] for c in excluded], int((~calib_keep).sum()), int((~pool_keep).sum()),
        )

    # num_classes stays the full taxonomy even when classes are excluded, so
    # class indices mean the same thing across every setup; LightGBM simply
    # assigns ~0 probability to a class with no training rows.
    boosting_model = BoostingClassifier(
        config.boosting,
        num_classes=len(class_names),
        benign_class=benign_class,
        seed=seed,
        confidence_threshold=config.cascade.confidence_threshold,
    )
    boosting_model.train(X_calib, y_calib)

    client_data = partition_and_normalize_clients(X_pool, y_pool, benign_class, config.data, seed)
    # Refit on each client's raw training rows: the exact scaler
    # partition_and_normalize_clients fit and applied (StandardScaler is
    # deterministic), kept for normalizing rows from outside the client.
    client_scalers = {
        cid: StandardScaler().fit(data["X_raw"])
        for cid, data in client_data.items()
        if config.data.normalize_per_client and len(data["X_raw"]) > 0
    }
    X_test_raw, X_test_norm, y_test, test_client_ids = concat_client_test_slices(client_data)

    logger.info(
        "Evaluation setup: %d calibration, %d federated pool (subsampled), %d test rows across %d clients",
        len(X_calib),
        len(X_pool),
        len(y_test),
        len(client_data),
    )

    return EvaluationSetup(
        client_data=client_data,
        boosting_model=boosting_model,
        class_names=class_names,
        benign_class=benign_class,
        input_dim=X.shape[1],
        X_test_raw=X_test_raw,
        X_test_norm=X_test_norm,
        y_test=y_test,
        test_client_ids=test_client_ids,
        client_scalers=client_scalers,
        excluded_classes=frozenset(excluded_classes),
    )


def calibrate_client_thresholds(
    final_weights: list[np.ndarray],
    setup: EvaluationSetup,
    config: Config,
) -> tuple[Autoencoder, dict[int, float]]:
    """Load final global weights and calibrate each client's own anomaly threshold.

    Mirrors what each client does after every FL round (component 3,
    `AutoencoderClient.evaluate`): the configured percentile of
    reconstruction error on its own benign validation slice. A client with
    no benign validation rows gets an infinite threshold, exactly as it
    does during training — so it never flags anything, and that cost shows
    up in the metrics rather than being papered over by a pooled fallback.

    Args:
        final_weights: The simulation's final global autoencoder weights.
        setup: The shared `EvaluationSetup`.
        config: Full project config (autoencoder architecture, anomaly percentile).

    Returns:
        (autoencoder, {client_id: threshold}).
    """
    autoencoder = Autoencoder(setup.input_dim, config.autoencoder.hidden_dims, config.autoencoder.bottleneck_dim)
    set_weights(autoencoder, final_weights)

    thresholds = {}
    for cid, data in setup.client_data.items():
        if len(data["X_val_benign"]) == 0:
            thresholds[cid] = float("inf")
            continue
        errors = reconstruction_error(autoencoder, data["X_val_benign"])
        thresholds[cid] = compute_anomaly_threshold(errors, config.autoencoder.anomaly_percentile)

    uncalibrated = [cid for cid, t in thresholds.items() if not np.isfinite(t)]
    if uncalibrated:
        affected = int(np.isin(setup.test_client_ids, uncalibrated).sum())
        logger.warning(
            "Clients %s have no benign validation rows (infinite threshold); %d test rows can't be flagged "
            "by the autoencoder", uncalibrated, affected,
        )
    return autoencoder, thresholds


def per_row_thresholds(thresholds: dict[int, float], client_ids: np.ndarray) -> np.ndarray:
    """Expand `{client_id: threshold}` to one threshold per row.

    Args:
        thresholds: `calibrate_client_thresholds` output.
        client_ids: Which client each row belongs to.

    Returns:
        Float array aligned with `client_ids`.
    """
    return np.array([thresholds[cid] for cid in client_ids], dtype=np.float64)


def assign_rows_to_clients(
    setup: EvaluationSetup, X_raw: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Spread rows from outside the federation across clients, normalizing each with its client's scaler.

    Used for the zero-day experiment's held-out attack rows, which belong
    to no client: each row is scored as if it appeared in one client's
    traffic, with clients assigned uniformly at random so a novel attack
    shows up everywhere rather than only at one client.

    Args:
        setup: The evaluation setup (its clients and `client_scalers`).
        X_raw: Raw-scale rows to assign.
        seed: Random seed for the assignment.

    Returns:
        (X_norm, client_ids), row-aligned with `X_raw`.
    """
    client_ids_available = np.array(sorted(setup.client_data))
    client_ids = np.random.default_rng(seed).choice(client_ids_available, size=len(X_raw))
    # Clients without a scaler (normalize_per_client off, or no training
    # rows) see raw features, as they do during training.
    X_norm = X_raw.astype(np.float32, copy=True)
    for cid, scaler in setup.client_scalers.items():
        rows = client_ids == cid
        if rows.any():
            X_norm[rows] = scaler.transform(X_raw[rows])
    return X_norm, client_ids


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
