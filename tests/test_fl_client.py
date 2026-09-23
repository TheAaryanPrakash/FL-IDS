"""Tests for the Flower client (component 4), no networking involved.

`test_fit_trains_autoencoder_only_on_boosting_filtered_subset` is one of
the two required Phase 3 milestone checks: it proves the client's
`fit()` trains on exactly the boosting-filtered "normal" subset, not raw
unfiltered local data.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.fl.client import AutoencoderClient
from fl_ids.models.autoencoder import get_weights
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import AutoencoderConfig, BoostingConfig, Config, CascadeConfig, DataConfig, EvaluationConfig, FLConfig, LoggingConfig, RobustnessConfig, SDNConfig

BENIGN_CLASS = 0


def _boosting_config(**overrides) -> BoostingConfig:
    defaults = dict(
        label_source="server_held_calibration_set",
        calibration_fraction=0.05,
        num_boost_round=50,
        learning_rate=0.1,
        num_leaves=31,
        broadcast_every_n_rounds=1,
        update_every_n_rounds=3,
    )
    defaults.update(overrides)
    return BoostingConfig(**defaults)


def _full_config(boosting_config: BoostingConfig, autoencoder_config: AutoencoderConfig) -> Config:
    return Config(
        seed=42,
        data=DataConfig(
            dnn_csv_path="unused.csv",
            pcap_dir="unused",
            num_clients=3,
            dirichlet_alpha=0.3,
            val_benign_fraction=0.2,
            test_fraction=0.2,
            normalize_per_client=True,
        ),
        boosting=boosting_config,
        autoencoder=autoencoder_config,
        cascade=CascadeConfig(confidence_threshold=0.7, anomaly_confidence_clip=(0.0, 1.0)),
        fl=FLConfig(num_rounds=5, clients_per_round=3, local_epochs=5, strategy="fedavg"),
        robustness=RobustnessConfig(
            trim_fraction=0.15, norm_clip_multiplier=2.0, mad_outlier_threshold=3.0, trust_ema_alpha=0.3
        ),
        sdn=SDNConfig(
            block_confidence_threshold=0.85,
            rate_limit_confidence_threshold=0.5,
            controller_host="127.0.0.1",
            controller_port=6653,
            bridge_api_host="127.0.0.1",
            bridge_api_port=8080,
        ),
        evaluation=EvaluationConfig(poisoning_fractions=[0.0], output_dir="results"),
        logging=LoggingConfig(level="INFO", log_dir="logs", log_file="fl_ids.log"),
    )


def _autoencoder_config(**overrides) -> AutoencoderConfig:
    defaults = dict(
        bottleneck_dim=4,
        hidden_dims=[16, 8],
        learning_rate=0.01,
        local_epochs=2,
        batch_size=32,
        anomaly_percentile=97,
        reconstruction_error_bins=20,
        reconstruction_error_range=(0.0, 5.0),
    )
    defaults.update(overrides)
    return AutoencoderConfig(**defaults)


def _make_client_and_boosting_model(seed: int = 10):
    X, y, class_names = make_synthetic_attack_dataset(n_samples=3000, n_features=12, seed=seed)
    X_train, X_client_raw, y_train, y_client = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Boosting always trains/predicts on raw features (see
    # fl_ids.models.boosting's module docstring for why per-client
    # normalization must never reach it).
    boosting_config = _boosting_config()
    autoencoder_config = _autoencoder_config()
    config = _full_config(boosting_config, autoencoder_config)

    # Ground-truth model used below to compute `expected_mask` -- built
    # with the same confidence_threshold (config.cascade's, the single
    # source of truth) that fit() will use internally, so the two can't
    # silently drift apart.
    boosting_model = BoostingClassifier(
        boosting_config, len(class_names), BENIGN_CLASS, seed=seed,
        confidence_threshold=config.cascade.confidence_threshold,
    )
    boosting_model.train(X_train, y_train)

    # The autoencoder trains on a *different* (normalized) representation
    # of the same rows -- mirrors component 1's per-client scaler, fit
    # only on this client's own data.
    client_scaler = StandardScaler().fit(X_client_raw)
    X_client_norm = client_scaler.transform(X_client_raw).astype(np.float32)

    val_benign_raw = X_client_raw[y_client == BENIGN_CLASS][:50]
    val_benign = client_scaler.transform(val_benign_raw).astype(np.float32)
    client = AutoencoderClient(
        client_id=0,
        X_train=X_client_norm,
        X_train_raw=X_client_raw,
        X_val_benign=val_benign,
        config=config,
        input_dim=X.shape[1],
    )
    return client, boosting_model, config


def _fit_config(boosting_model: BoostingClassifier, num_classes: int) -> dict:
    return {
        "boosting_model_bytes": boosting_model.to_bytes(),
        "num_classes": num_classes,
        "benign_class": BENIGN_CLASS,
        "server_round": 1,
    }


def test_fit_trains_autoencoder_only_on_boosting_filtered_subset():
    """Phase 3 milestone: confirms fit() trains the autoencoder on exactly
    the boosting-filtered subset of local data, not raw unfiltered data.
    """
    client, boosting_model, config = _make_client_and_boosting_model()
    fit_config = _fit_config(boosting_model, boosting_model.num_classes)

    # The filter mask is computed on raw features, but the autoencoder
    # must train on the *normalized* counterpart of exactly those rows.
    expected_mask = boosting_model.passes_to_autoencoder(client.X_train_raw)
    expected_X_filtered = client.X_train[expected_mask]

    # Sanity: the filter must actually exclude *something*, or this test
    # would pass vacuously even if fit() ignored filtering entirely.
    assert 0 < expected_mask.sum() < len(expected_mask)

    initial_weights = get_weights(client.model)

    from fl_ids.models.autoencoder import train_autoencoder as original_train

    captured = {}

    def spy_train_autoencoder(model, X, cfg, seed):
        captured["X"] = X.copy()
        return original_train(model, X, cfg, seed)

    with patch("fl_ids.fl.client.train_autoencoder", side_effect=spy_train_autoencoder):
        client.fit(initial_weights, fit_config)

    assert "X" in captured, "train_autoencoder should have been called"
    assert captured["X"].shape == expected_X_filtered.shape
    assert np.allclose(np.sort(captured["X"], axis=0), np.sort(expected_X_filtered, axis=0))
    # And explicitly not the same as the raw unfiltered local data (proves
    # filtering actually happened, not a no-op pass-through).
    assert captured["X"].shape[0] != client.X_train_raw.shape[0]


def test_fit_reports_filtered_fraction_and_num_examples_consistently():
    client, boosting_model, _ = _make_client_and_boosting_model(seed=11)
    fit_config = _fit_config(boosting_model, boosting_model.num_classes)
    initial_weights = get_weights(client.model)

    _, num_examples, metrics = client.fit(initial_weights, fit_config)

    expected_mask = boosting_model.passes_to_autoencoder(client.X_train_raw)
    assert num_examples == int(expected_mask.sum())
    assert metrics["filtered_fraction"] == client.last_filtered_fraction
    assert metrics["filtered_fraction"] == expected_mask.mean()


def test_fit_returns_updated_weights_with_matching_shapes():
    client, boosting_model, _ = _make_client_and_boosting_model(seed=12)
    fit_config = _fit_config(boosting_model, boosting_model.num_classes)
    initial_weights = get_weights(client.model)

    updated_weights, _, _ = client.fit(initial_weights, fit_config)

    assert len(updated_weights) == len(initial_weights)
    for before, after in zip(initial_weights, updated_weights):
        assert before.shape == after.shape


def test_evaluate_recomputes_threshold_and_returns_mean_error():
    client, boosting_model, _ = _make_client_and_boosting_model(seed=13)
    fit_config = _fit_config(boosting_model, boosting_model.num_classes)
    initial_weights = get_weights(client.model)

    updated_weights, _, _ = client.fit(initial_weights, fit_config)
    loss, num_examples, metrics = client.evaluate(updated_weights, {})

    assert num_examples == len(client.X_val_benign)
    assert loss >= 0.0
    assert metrics["anomaly_threshold"] >= 0.0


def test_evaluate_with_no_benign_validation_data_is_safe():
    client, boosting_model, _ = _make_client_and_boosting_model(seed=14)
    client.X_val_benign = np.empty((0, client.X_train_raw.shape[1]), dtype=np.float32)
    weights = get_weights(client.model)

    loss, num_examples, metrics = client.evaluate(weights, {})

    assert loss == 0.0
    assert num_examples == 0
    assert metrics["anomaly_threshold"] == float("inf")
