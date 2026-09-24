"""Data pipeline (component 1).

Loads the pre-selected `DNN-EdgeIIoT-dataset.csv` (61 engineered features —
the dataset authors already reduced these from the raw 1176; this module
does not redo that feature engineering), drops ID/timestamp/raw-payload
columns, one-hot encodes the remaining nominal categoricals, partitions
samples across simulated clients non-IID via a Dirichlet(alpha) label-skew
split, and — per client, never globally — carves out a held-out benign
validation slice (for autoencoder reconstruction-error thresholding) and a
held-out test slice (for evaluation), then normalizes each client's splits
using a scaler fit only on that client's own training data.

The column drop list and one-hot categorical column list below are taken
verbatim from the dataset authors' own preprocessing recipe (see
`data/raw/Readme.txt`, "Step 4" and "Step 5"), not independently re-derived —
CLAUDE.md is explicit that this dataset's feature engineering shouldn't be
redone. `Attack_label` is additionally excluded from the feature matrix
(beyond the authors' recipe) because it is fully derivable from the
`Attack_type` target used here (0 iff Attack_type == "Normal"), so keeping
it as a feature would leak the label.

One deliberate departure from the authors' recipe: the "field doesn't
apply" placeholder in the categorical columns is spelled "0" in some
source capture files and "0.0" in others, and the spelling follows the
label -- in the mqtt.* and dns.qry.name.len columns every Normal row says
"0" and every attack row "0.0", so the one-hot column `mqtt.topic_0.0`
alone separates benign from attack with AUROC 1.0. That's an export
artifact, not network behavior, and any model trained on it learns the
file format. `canonicalize_placeholders` collapses both spellings to one
before encoding (see its docstring).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from fl_ids.utils.config import DataConfig

logger = logging.getLogger(__name__)

# Verbatim from the Edge-IIoTset authors' Readme.txt "Step 4": ID/timestamp/
# raw-payload columns that don't generalize across hosts or testbeds.
DEFAULT_DROP_COLUMNS: list[str] = [
    "frame.time",
    "ip.src_host",
    "ip.dst_host",
    "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4",
    "http.file_data",
    "http.request.full_uri",
    "icmp.transmit_timestamp",
    "http.request.uri.query",
    "tcp.options",
    "tcp.payload",
    "tcp.srcport",
    "tcp.dstport",
    "udp.port",
    "mqtt.msg",
]

# Verbatim from the Readme's "Step 5": nominal columns one-hot encoded via
# `encode_text_dummy` (equivalent to pandas `get_dummies`).
CATEGORICAL_COLUMNS: list[str] = [
    "http.request.method",
    "http.referer",
    "http.request.version",
    "dns.qry.name.len",
    "mqtt.conack.flags",
    "mqtt.protoname",
    "mqtt.topic",
]

# Spellings of the categorical columns' "field doesn't apply" placeholder,
# and the one they're all rewritten to. Only these two occur in the dataset.
PLACEHOLDER_SPELLINGS: tuple[str, ...] = ("0", "0.0")
CANONICAL_PLACEHOLDER = "0"

TARGET_COLUMN = "Attack_type"
LEAKY_LABEL_COLUMNS: list[str] = ["Attack_label"]
BENIGN_LABEL = "Normal"


def load_raw_csv(
    csv_path: str | Path, drop_columns: list[str] | None = None
) -> pd.DataFrame:
    """Load the DNN-EdgeIIoT-dataset.csv, excluding dropped columns at parse time.

    Excluding ID/timestamp/raw-payload columns via `usecols` (rather than
    loading then dropping) avoids ever materializing the highest-cardinality
    string columns (e.g. `tcp.payload`, `ip.src_host`) in memory.

    Args:
        csv_path: Path to the CSV file.
        drop_columns: Columns to exclude; defaults to `DEFAULT_DROP_COLUMNS`.

    Returns:
        The loaded DataFrame, with dropped and leaky-label columns absent.
    """
    drop_columns = drop_columns if drop_columns is not None else DEFAULT_DROP_COLUMNS
    excluded = set(drop_columns) | set(LEAKY_LABEL_COLUMNS)
    all_columns = pd.read_csv(csv_path, nrows=0).columns.tolist()
    usecols = [c for c in all_columns if c not in excluded]
    logger.info(
        "Loading %d/%d columns from %s (excluded %d)",
        len(usecols),
        len(all_columns),
        csv_path,
        len(all_columns) - len(usecols),
    )
    return pd.read_csv(csv_path, usecols=usecols, low_memory=False)


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Drop NaN rows and exact-duplicate rows, matching the authors' Step 4.

    Args:
        df: Raw (post-column-drop) DataFrame.

    Returns:
        A cleaned DataFrame with no NaNs and no duplicate rows, index reset.
    """
    before = len(df)
    df = df.dropna(axis=0, how="any")
    after_na = len(df)
    df = df.drop_duplicates(subset=None, keep="first")
    after_dup = len(df)
    logger.info(
        "Cleaned dataframe: %d -> %d rows (dropped %d NaN, %d duplicate)",
        before,
        after_dup,
        before - after_na,
        after_na - after_dup,
    )
    return df.reset_index(drop=True)


def canonicalize_placeholders(
    df: pd.DataFrame, categorical_columns: list[str] | None = None
) -> pd.DataFrame:
    """Rewrite every placeholder spelling in the categorical columns to one canonical value.

    The authors' CSV writes the "field doesn't apply" placeholder as "0" or
    "0.0" depending on which per-attack/per-device capture file a row came
    from, so the spelling encodes the label: in mqtt.topic,
    mqtt.conack.flags, mqtt.protoname and dns.qry.name.len every Normal row
    uses "0" and every attack row "0.0" (no overlap across 2.2M rows).
    Left alone, one-hot encoding turns that into a perfect benign-vs-attack
    feature. Live traffic has no such spelling (tshark reports an absent
    field as empty), so a model leaning on it would also fail in Phase B.

    Must run before `clean_dataframe`, so rows that differed only in
    placeholder spelling are deduplicated together.

    Args:
        df: Raw (post-column-drop) DataFrame.
        categorical_columns: Columns to canonicalize; defaults to
            `CATEGORICAL_COLUMNS`.

    Returns:
        A copy of `df` with every `PLACEHOLDER_SPELLINGS` value in those
        columns replaced by `CANONICAL_PLACEHOLDER`.
    """
    categorical_columns = (
        categorical_columns if categorical_columns is not None else CATEGORICAL_COLUMNS
    )
    df = df.copy()
    for col in categorical_columns:
        if col not in df.columns:
            continue
        as_str = df[col].astype(str)
        is_placeholder = as_str.isin(PLACEHOLDER_SPELLINGS)
        # Only rows that actually hold a value get rewritten; NaN stays NaN
        # for clean_dataframe to drop.
        is_placeholder &= df[col].notna()
        df[col] = df[col].astype(object).where(~is_placeholder, CANONICAL_PLACEHOLDER)
        logger.debug("Canonicalized %d placeholder values in %s", int(is_placeholder.sum()), col)
    return df


def one_hot_encode_categoricals(
    df: pd.DataFrame, categorical_columns: list[str] | None = None
) -> pd.DataFrame:
    """One-hot encode nominal categorical columns, matching the authors' Step 5.

    Encoding is done globally (on the full dataset, before per-client
    partitioning) so every client ends up with an identically-shaped,
    aligned feature space — unlike per-client normalization, a fixed
    category-to-column vocabulary carries no cross-client distributional
    information, so doing this globally does not leak the kind of
    information per-client normalization is meant to protect.

    Args:
        df: DataFrame after cleaning.
        categorical_columns: Columns to one-hot encode; defaults to
            `CATEGORICAL_COLUMNS`.

    Returns:
        DataFrame with categorical columns replaced by one-hot indicator
        columns.
    """
    categorical_columns = (
        categorical_columns if categorical_columns is not None else CATEGORICAL_COLUMNS
    )
    present = [c for c in categorical_columns if c in df.columns]
    return pd.get_dummies(df, columns=present)


def dirichlet_partition(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int
) -> dict[int, np.ndarray]:
    """Partition sample indices across clients via a Dirichlet(alpha) label skew.

    For each class, draws a per-client proportion vector from
    Dirichlet(alpha, ..., alpha) and splits that class's sample indices
    accordingly — the standard non-IID federated partition recipe (Hsu,
    Qi & Brown, 2019). Small alpha (e.g. 0.1) yields a highly skewed,
    non-IID split; large alpha approaches a uniform (IID-like) split.

    Args:
        labels: Integer-encoded class label for every sample.
        num_clients: Number of simulated clients to split across.
        alpha: Dirichlet concentration parameter.
        seed: Random seed, for reproducibility.

    Returns:
        {client_id: array of sample indices assigned to that client}.
    """
    rng = np.random.default_rng(seed)
    client_indices: dict[int, list[int]] = {i: [] for i in range(num_clients)}
    for c in np.unique(labels):
        idx_c = np.where(labels == c)[0]
        rng.shuffle(idx_c)
        proportions = rng.dirichlet(alpha * np.ones(num_clients))
        split_points = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
        for client_id, split in enumerate(np.split(idx_c, split_points)):
            client_indices[client_id].extend(split.tolist())
    return {cid: np.array(idx, dtype=np.int64) for cid, idx in client_indices.items()}


def _split_client_data(
    client_idx: np.ndarray,
    y: np.ndarray,
    benign_class: int,
    val_benign_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Carve one client's indices into (train, val_benign, test) index arrays."""
    rng = np.random.default_rng(seed)
    idx = client_idx.copy()
    rng.shuffle(idx)

    n_test = int(len(idx) * test_fraction)
    test_idx, remaining = idx[:n_test], idx[n_test:]

    benign_mask = y[remaining] == benign_class
    benign_remaining = remaining[benign_mask]
    non_benign_remaining = remaining[~benign_mask]

    n_val_benign = int(len(benign_remaining) * val_benign_fraction)
    val_benign_idx = benign_remaining[:n_val_benign]
    train_benign_idx = benign_remaining[n_val_benign:]

    train_idx = np.concatenate([train_benign_idx, non_benign_remaining])
    return train_idx, val_benign_idx, test_idx


def partition_and_normalize_clients(
    X: np.ndarray,
    y: np.ndarray,
    benign_class: int,
    config: DataConfig,
    seed: int,
) -> dict[int, dict[str, np.ndarray]]:
    """Partition (X, y) non-IID across clients, then per-client split and normalize.

    Shared by `build_federated_dataset` (real Edge-IIoTset CSV) and the
    synthetic federated dataset builder used for Phase 3's FL-loop tests
    (`fl_ids.data.synthetic.make_synthetic_federated_dataset`) — both need
    the identical Dirichlet-partition -> per-client held-out-slice ->
    per-client-normalize pipeline, just starting from different sources of
    (X, y).

    **Both raw and per-client-normalized features are returned for the
    train/test splits** (`"X"`/`"X_raw"`, `"X_test"`/`"X_test_raw"`) —
    see `fl_ids.models.boosting`'s module docstring for why: the boosting
    classifier (component 2) must always see raw features, never
    per-client-normalized ones. Under real non-IID skew, a client's local
    mean/std can differ from the population's by orders of magnitude
    (e.g. a client that's 94% benign traffic ends up with per-feature
    statistics dominated by benign traffic's characteristic value range),
    which silently destroys a shared model's decision-tree thresholds if
    it's ever evaluated against per-client-normalized input. Phase 5's
    real-data validation caught this directly: the same held-out rows
    scored ~0.94 accuracy in raw features vs. ~0.06 through a
    heavily-skewed client's own scaler. `X_val_benign` doesn't need a raw
    counterpart — component 3 never runs it through boosting, only
    through the autoencoder, which does want normalized input.

    Args:
        X: Full feature matrix, shape (n_samples, n_features).
        y: Integer-encoded class labels, shape (n_samples,).
        benign_class: Integer class index corresponding to the benign class.
        config: Data configuration (num_clients, dirichlet_alpha, split
            fractions, normalize_per_client).
        seed: Global random seed, for reproducible partitioning/splitting.

    Returns:
        `{client_id: {"X", "X_raw", "y", "X_val_benign", "y_val_benign",
        "X_test", "X_test_raw", "y_test"}}`, all values `np.ndarray`.
    """
    client_index_map = dirichlet_partition(y, config.num_clients, config.dirichlet_alpha, seed)

    client_data: dict[int, dict[str, np.ndarray]] = {}
    for client_id, client_idx in client_index_map.items():
        if len(client_idx) == 0:
            logger.warning("Client %d received zero samples from the Dirichlet split", client_id)

        train_idx, val_benign_idx, test_idx = _split_client_data(
            client_idx,
            y,
            benign_class,
            config.val_benign_fraction,
            config.test_fraction,
            seed=seed + client_id,
        )
        if len(val_benign_idx) == 0:
            logger.warning(
                "Client %d has zero benign validation samples (heavy Dirichlet skew "
                "or too few benign samples in its shard)",
                client_id,
            )

        X_train_raw, X_val_benign_raw, X_test_raw = X[train_idx], X[val_benign_idx], X[test_idx]

        if config.normalize_per_client and len(train_idx) > 0:
            scaler = StandardScaler().fit(X_train_raw)
            X_train_norm = scaler.transform(X_train_raw)
            X_val_benign_norm = scaler.transform(X_val_benign_raw) if len(val_benign_idx) else X_val_benign_raw
            X_test_norm = scaler.transform(X_test_raw) if len(test_idx) else X_test_raw
        else:
            X_train_norm, X_val_benign_norm, X_test_norm = X_train_raw, X_val_benign_raw, X_test_raw

        client_data[client_id] = {
            "X": X_train_norm.astype(np.float32),
            "X_raw": X_train_raw.astype(np.float32),
            "y": y[train_idx],
            "X_val_benign": X_val_benign_norm.astype(np.float32),
            "y_val_benign": y[val_benign_idx],
            "X_test": X_test_norm.astype(np.float32),
            "X_test_raw": X_test_raw.astype(np.float32),
            "y_test": y[test_idx],
        }

    return client_data


def load_and_encode(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray, LabelEncoder, list[str], int]:
    """Load, clean, and encode the raw CSV into a feature matrix and integer labels.

    Shared by `build_federated_dataset` and
    `build_server_and_federated_dataset` — both need the identical
    load/clean/encode steps, just splitting the resulting (X, y)
    differently afterward.

    Args:
        csv_path: Path to `DNN-EdgeIIoT-dataset.csv`.

    Returns:
        (X, y, label_encoder, feature_names, benign_class).

    Raises:
        AssertionError: If NaNs remain in the feature matrix after cleaning
            (would indicate a bug in `clean_dataframe`).
    """
    df = load_raw_csv(csv_path)
    df = canonicalize_placeholders(df)
    df = clean_dataframe(df)
    df = one_hot_encode_categoricals(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN].values)
    benign_class = int(label_encoder.transform([BENIGN_LABEL])[0])

    feature_df = df.drop(columns=[TARGET_COLUMN])
    feature_names = feature_df.columns.tolist()
    X = feature_df.to_numpy(dtype=np.float32)

    assert not np.isnan(X).any(), "NaNs present in feature matrix after cleaning"

    return X, y, label_encoder, feature_names, benign_class


def build_federated_dataset(
    csv_path: str | Path,
    config: DataConfig,
    seed: int,
) -> tuple[dict[int, dict[str, np.ndarray]], LabelEncoder, list[str]]:
    """Build the full per-client federated dataset (component 1's main entrypoint).

    Args:
        csv_path: Path to `DNN-EdgeIIoT-dataset.csv`.
        config: Data configuration (num_clients, dirichlet_alpha, split
            fractions, normalize_per_client).
        seed: Global random seed, for reproducible partitioning/splitting.

    Returns:
        A tuple of:
        - `{client_id: {"X", "y", "X_val_benign", "y_val_benign", "X_test",
          "y_test"}}`, all values `np.ndarray`.
        - The fitted `LabelEncoder` mapping `Attack_type` strings to the
          integer classes used in `y`.
        - The list of feature column names, in `X` column order.
    """
    X, y, label_encoder, feature_names, benign_class = load_and_encode(csv_path)
    client_data = partition_and_normalize_clients(X, y, benign_class, config, seed)
    return client_data, label_encoder, feature_names


def build_server_and_federated_dataset(
    csv_path: str | Path,
    config: DataConfig,
    calibration_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, np.ndarray]], LabelEncoder, list[str]]:
    """Build a server-held calibration set plus the per-client federated dataset.

    Per component 2's label-source design decision (see
    `fl_ids.models.boosting`'s module docstring), the boosting classifier's
    training labels come from a small, independently server-held
    calibration set — never from client data. That means this calibration
    slice must be carved out of the full dataset *before* client
    partitioning and kept disjoint from every client's data, not
    reconstructed after the fact from what clients happen to hold.

    Args:
        csv_path: Path to `DNN-EdgeIIoT-dataset.csv`.
        config: Data configuration (num_clients, dirichlet_alpha, split
            fractions, normalize_per_client) — applied to the *remainder*
            after the calibration slice is removed.
        calibration_fraction: Fraction of the full cleaned dataset held out
            server-side (stratified by class), matching
            `BoostingConfig.calibration_fraction`.
        seed: Global random seed, for reproducible splitting/partitioning.

    Returns:
        (X_calibration, y_calibration, client_data, label_encoder, feature_names).
        `X_calibration` is raw-scale (not normalized) — the caller
        standardizes it however the boosting model expects (see
        `fl_ids.models.boosting`'s feature-scale design decision).
    """
    X, y, label_encoder, feature_names, benign_class = load_and_encode(csv_path)

    X_calib, X_pool, y_calib, y_pool = train_test_split(
        X, y, train_size=calibration_fraction, random_state=seed, stratify=y
    )

    client_data = partition_and_normalize_clients(X_pool, y_pool, benign_class, config, seed)
    return X_calib, y_calib, client_data, label_encoder, feature_names
