"""Tests for the data pipeline (component 1).

Fast unit tests use small synthetic DataFrames/CSVs to exercise the
column-dropping, cleaning, encoding, and partitioning mechanics in
isolation. `test_build_federated_dataset_real_data` is the Phase 1
milestone check: it loads the actual `DNN-EdgeIIoT-dataset.csv` and
confirms the full pipeline produces correctly-shaped, non-IID,
NaN-free client data — it's skipped (not failed) if the dataset file
isn't present, so the rest of the suite stays runnable without it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fl_ids.data.pipeline import (
    canonicalize_placeholders,
    load_and_encode,
    CATEGORICAL_COLUMNS,
    DEFAULT_DROP_COLUMNS,
    TARGET_COLUMN,
    build_federated_dataset,
    build_server_and_federated_dataset,
    clean_dataframe,
    dirichlet_partition,
    load_raw_csv,
    one_hot_encode_categoricals,
)
from fl_ids.utils.config import DataConfig

REAL_DATASET_PATH = Path("data/raw/DNN-EdgeIIoT-dataset.csv")


def _base_data_config(**overrides) -> DataConfig:
    defaults = dict(
        dnn_csv_path=str(REAL_DATASET_PATH),
        pcap_dir="data/raw/pcaps",
        num_clients=5,
        dirichlet_alpha=0.1,
        val_benign_fraction=0.2,
        test_fraction=0.2,
        normalize_per_client=True,
    )
    defaults.update(overrides)
    return DataConfig(**defaults)


def _pairwise_total_variation(distributions: np.ndarray) -> float:
    """Average pairwise total-variation distance between row-wise distributions."""
    n = distributions.shape[0]
    dists = []
    for i in range(n):
        for j in range(i + 1, n):
            dists.append(0.5 * np.abs(distributions[i] - distributions[j]).sum())
    return float(np.mean(dists)) if dists else 0.0


def _make_synthetic_dataframe(n_per_class: int = 300, seed: int = 0) -> pd.DataFrame:
    """Build a small DataFrame mimicking the real schema closely enough to
    exercise column dropping, categorical encoding, and class/feature
    correlation (so non-IID label skew implies non-IID feature skew, as in
    the real dataset).
    """
    rng = np.random.default_rng(seed)
    classes = ["Normal", "DDoS_UDP", "SQL_injection", "Password", "XSS"]

    rows = []
    for class_idx, cls in enumerate(classes):
        for _ in range(n_per_class):
            rows.append(
                {
                    # Columns that must be dropped:
                    "frame.time": f"time_{rng.integers(0, 1_000_000)}",
                    "ip.src_host": f"10.0.0.{rng.integers(1, 254)}",
                    "ip.dst_host": f"10.0.0.{rng.integers(1, 254)}",
                    "tcp.payload": f"payload_{rng.integers(0, 1_000_000)}",
                    # Categorical columns to be one-hot encoded:
                    "http.request.method": rng.choice(["GET", "POST", "0"]),
                    "http.referer": rng.choice(["0", "ref_a"]),
                    "http.request.version": rng.choice(["HTTP/1.1", "0"]),
                    "dns.qry.name.len": rng.choice(["0", "12"]),
                    "mqtt.conack.flags": rng.choice(["0", "0x00000000"]),
                    "mqtt.protoname": rng.choice(["0", "MQTT"]),
                    "mqtt.topic": rng.choice(["0", "topic_a"]),
                    # Numeric features, correlated with class so label skew
                    # implies a detectable feature-distribution skew:
                    "feature_a": class_idx * 10.0 + rng.normal(0, 1.0),
                    "feature_b": rng.normal(0, 1.0),
                    # Labels:
                    "Attack_label": 0 if cls == "Normal" else 1,
                    TARGET_COLUMN: cls,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_csv(tmp_path) -> Path:
    df = _make_synthetic_dataframe()
    path = tmp_path / "synthetic_edgeiiot.csv"
    df.to_csv(path, index=False)
    return path


def test_load_raw_csv_excludes_drop_and_leaky_columns(synthetic_csv):
    df = load_raw_csv(synthetic_csv)
    for col in DEFAULT_DROP_COLUMNS:
        assert col not in df.columns
    assert "Attack_label" not in df.columns
    assert TARGET_COLUMN in df.columns
    assert "feature_a" in df.columns


def test_clean_dataframe_drops_nan_and_duplicate_rows():
    df = pd.DataFrame(
        {
            "a": [1.0, 1.0, np.nan, 2.0],
            "b": [1.0, 1.0, 3.0, 2.0],
            TARGET_COLUMN: ["Normal", "Normal", "DDoS_UDP", "Normal"],
        }
    )
    cleaned = clean_dataframe(df)
    # Row 2 (NaN) dropped; rows 0 and 1 are exact duplicates -> one dropped.
    assert len(cleaned) == 2
    assert not cleaned.isna().any().any()


def test_one_hot_encode_categoricals_replaces_columns_with_indicators():
    df = pd.DataFrame(
        {
            "http.request.method": ["GET", "POST", "GET"],
            "numeric_col": [1.0, 2.0, 3.0],
        }
    )
    encoded = one_hot_encode_categoricals(df, categorical_columns=["http.request.method"])
    assert "http.request.method" not in encoded.columns
    assert "http.request.method_GET" in encoded.columns
    assert "http.request.method_POST" in encoded.columns
    assert "numeric_col" in encoded.columns


def test_canonicalize_placeholders_collapses_spellings_before_encoding():
    df = pd.DataFrame(
        {
            "mqtt.topic": ["0", "0.0", "Temperature_and_Humidity", None],
            "tcp.len": [0.0, 1.0, 2.0, 3.0],
        }
    )
    encoded = one_hot_encode_categoricals(canonicalize_placeholders(df))

    assert "mqtt.topic_0.0" not in encoded.columns
    assert encoded["mqtt.topic_0"].tolist() == [1, 1, 0, 0]
    assert encoded["tcp.len"].tolist() == [0.0, 1.0, 2.0, 3.0]  # non-categorical columns untouched


def test_dirichlet_partition_is_a_complete_nonoverlapping_split():
    labels = np.array([0] * 100 + [1] * 100 + [2] * 100)
    partition = dirichlet_partition(labels, num_clients=4, alpha=0.3, seed=42)

    assert set(partition.keys()) == {0, 1, 2, 3}
    all_idx = np.concatenate(list(partition.values()))
    assert len(all_idx) == len(labels)
    assert len(set(all_idx.tolist())) == len(labels)  # no overlap


def test_dirichlet_partition_low_alpha_more_skewed_than_high_alpha():
    labels = np.array([0] * 1000 + [1] * 1000 + [2] * 1000 + [3] * 1000)
    num_clients = 6

    def client_label_distributions(alpha: float, seed: int) -> np.ndarray:
        partition = dirichlet_partition(labels, num_clients, alpha, seed)
        dists = []
        for cid in range(num_clients):
            idx = partition[cid]
            counts = np.bincount(labels[idx], minlength=4).astype(float)
            dists.append(counts / counts.sum() if counts.sum() > 0 else counts)
        return np.array(dists)

    low_alpha_tvd = _pairwise_total_variation(client_label_distributions(0.1, seed=1))
    high_alpha_tvd = _pairwise_total_variation(client_label_distributions(100.0, seed=1))

    assert low_alpha_tvd > 0.2, "low-alpha (non-IID) split should be strongly skewed"
    assert low_alpha_tvd > high_alpha_tvd * 2, (
        "low-alpha split should be substantially more skewed than a near-uniform "
        "high-alpha split"
    )


def test_build_federated_dataset_synthetic_end_to_end(synthetic_csv):
    config = _base_data_config(dnn_csv_path=str(synthetic_csv), num_clients=5, dirichlet_alpha=0.1)
    client_data, label_encoder, feature_names = build_federated_dataset(
        synthetic_csv, config, seed=42
    )

    assert set(client_data.keys()) == set(range(5))
    benign_class = label_encoder.transform(["Normal"])[0]

    for cid, data in client_data.items():
        for key in ("X", "y", "X_val_benign", "y_val_benign", "X_test", "y_test"):
            assert key in data
        assert not np.isnan(data["X"]).any(), f"client {cid} train X has NaNs"
        assert not np.isnan(data["X_val_benign"]).any(), f"client {cid} val_benign has NaNs"
        assert not np.isnan(data["X_test"]).any(), f"client {cid} test X has NaNs"
        assert data["X"].shape[0] == data["y"].shape[0]
        assert data["X"].shape[1] == len(feature_names)
        if data["y_val_benign"].shape[0] > 0:
            assert np.all(data["y_val_benign"] == benign_class)


def test_build_federated_dataset_per_client_feature_distributions_differ(synthetic_csv):
    """Milestone check: confirms the non-IID split is real, not accidentally
    uniform, by showing per-client feature distributions actually differ.
    """
    # normalize_per_client=False here so we're comparing raw feature values —
    # per-client normalization deliberately re-centers every client to
    # mean 0/std 1, which would mask the very skew this test checks for.
    low_alpha_config = _base_data_config(
        dnn_csv_path=str(synthetic_csv), num_clients=5, dirichlet_alpha=0.05,
        normalize_per_client=False,
    )
    high_alpha_config = _base_data_config(
        dnn_csv_path=str(synthetic_csv), num_clients=5, dirichlet_alpha=100.0,
        normalize_per_client=False,
    )

    low_alpha_data, _, feature_names = build_federated_dataset(synthetic_csv, low_alpha_config, seed=7)
    high_alpha_data, _, _ = build_federated_dataset(synthetic_csv, high_alpha_config, seed=7)

    feature_a_idx = feature_names.index("feature_a")

    def per_client_means(client_data):
        means = []
        for data in client_data.values():
            combined = np.concatenate([data["X"], data["X_val_benign"], data["X_test"]])
            if len(combined) > 0:
                means.append(combined[:, feature_a_idx].mean())
        return np.array(means)

    low_alpha_spread = per_client_means(low_alpha_data).std()
    high_alpha_spread = per_client_means(high_alpha_data).std()

    assert low_alpha_spread > high_alpha_spread * 2, (
        "strongly non-IID (low alpha) partition should show far more spread in "
        "per-client feature_a means than a near-uniform (high alpha) partition"
    )


def test_build_federated_dataset_normalization_is_per_client_not_global(synthetic_csv):
    config = _base_data_config(dnn_csv_path=str(synthetic_csv), num_clients=5, dirichlet_alpha=0.05)
    client_data, _, _ = build_federated_dataset(synthetic_csv, config, seed=3)

    for cid, data in client_data.items():
        if data["X"].shape[0] < 5:
            continue  # too few samples for a meaningful mean/std check
        assert np.allclose(data["X"].mean(axis=0), 0.0, atol=1e-4), (
            f"client {cid} train features should be ~zero-mean under per-client "
            "normalization"
        )
        assert np.allclose(data["X"].std(axis=0), 1.0, atol=1e-2), (
            f"client {cid} train features should be ~unit-std under per-client "
            "normalization"
        )


@pytest.mark.skipif(not REAL_DATASET_PATH.exists(), reason="real dataset not present")
def test_build_federated_dataset_real_data():
    """Phase 1 milestone: loading the real dataset produces correctly-shaped,
    non-IID-partitioned client data with no NaNs, confirmed non-uniform.
    """
    config = _base_data_config(num_clients=10, dirichlet_alpha=0.3)
    client_data, label_encoder, feature_names = build_federated_dataset(
        REAL_DATASET_PATH, config, seed=42
    )

    assert set(client_data.keys()) == set(range(10))
    assert len(feature_names) > 0
    assert "Normal" in label_encoder.classes_
    benign_class = label_encoder.transform(["Normal"])[0]

    total_rows = 0
    label_dists = []
    for cid, data in client_data.items():
        for key in ("X", "X_val_benign", "X_test"):
            assert not np.isnan(data[key]).any(), f"client {cid} {key} has NaNs"
        total_rows += data["X"].shape[0] + data["X_val_benign"].shape[0] + data["X_test"].shape[0]
        if data["y_val_benign"].shape[0] > 0:
            assert np.all(data["y_val_benign"] == benign_class)

        counts = np.bincount(data["y"], minlength=len(label_encoder.classes_)).astype(float)
        label_dists.append(counts / counts.sum() if counts.sum() > 0 else counts)

    assert total_rows > 1_000_000  # sanity: most of the ~2.2M rows survived cleaning

    tvd = _pairwise_total_variation(np.array(label_dists))
    assert tvd > 0.05, "real-data client split should show a real non-IID label skew"


@pytest.mark.skipif(not REAL_DATASET_PATH.exists(), reason="real dataset not present")
def test_build_server_and_federated_dataset_calibration_disjoint_from_clients():
    """Phase 5: the server-held calibration set (component 2's label
    source) must be carved out before client partitioning and stay
    disjoint from every client's data — not reconstructed after the
    fact from data clients already hold.

    Client data is per-client *normalized* while the calibration set is
    raw-scale, so a direct row-value comparison between the two would be
    meaningless (identical underlying rows look numerically different
    after different scalers). Instead this checks: (a) the calibration
    split is exactly reproducible from the same (X, y, seed) via a
    fresh, independent `train_test_split` call — proving disjointness by
    construction, backed by sklearn's own tested guarantee that
    `train_test_split` returns non-overlapping index partitions — and
    (b) row counts are conserved (no rows lost or double-counted between
    the calibration set and the client pool).
    """
    from sklearn.model_selection import train_test_split

    from fl_ids.data.pipeline import load_and_encode

    config = _base_data_config(num_clients=5, dirichlet_alpha=0.3)
    X_calib, y_calib, client_data, label_encoder, feature_names = build_server_and_federated_dataset(
        REAL_DATASET_PATH, config, calibration_fraction=0.05, seed=42
    )

    assert X_calib.shape[0] > 0
    assert X_calib.shape[1] == len(feature_names)
    assert not np.isnan(X_calib).any()

    X_full, y_full, _, _, _ = load_and_encode(REAL_DATASET_PATH)
    expected_X_calib, expected_X_pool, expected_y_calib, _ = train_test_split(
        X_full, y_full, train_size=0.05, random_state=42, stratify=y_full
    )
    assert np.array_equal(X_calib, expected_X_calib)
    assert np.array_equal(y_calib, expected_y_calib)

    total_client_rows = sum(
        data["X"].shape[0] + data["X_val_benign"].shape[0] + data["X_test"].shape[0]
        for data in client_data.values()
    )
    assert total_client_rows == expected_X_pool.shape[0]
    # Calibration set (~5%) should be much smaller than what's left for clients.
    assert X_calib.shape[0] < total_client_rows * 0.1


@pytest.mark.skipif(not REAL_DATASET_PATH.exists(), reason="real dataset not present")
def test_no_single_feature_separates_benign_from_attack_on_real_data():
    """Guards against export artifacts leaking the label into the features.

    The source CSVs spell the categorical placeholder "0" for Normal rows
    and "0.0" for attack rows in several columns; before canonicalization
    the one-hot column `mqtt.topic_0.0` alone scored AUROC 1.0000. After
    it, the strongest single feature (tcp.flags.ack) scores ~0.71. Real
    traffic features overlap; a near-perfect single feature means a
    recording artifact, not behavior.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import train_test_split

    X, y, _, feature_names, benign_class = load_and_encode(REAL_DATASET_PATH)
    assert not [f for f in feature_names if f.endswith("_0.0")]

    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=100_000, random_state=0, stratify=y)
    is_attack = (y_sample != benign_class).astype(int)
    too_separable = {}
    for j, name in enumerate(feature_names):
        column = X_sample[:, j]
        if column.std() == 0:
            continue
        auroc = roc_auc_score(is_attack, column)
        auroc = max(auroc, 1.0 - auroc)
        if auroc > 0.9:
            too_separable[name] = round(auroc, 4)
    assert not too_separable, f"features that alone separate benign from attack: {too_separable}"
