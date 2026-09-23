"""Tests for the in-process FL simulation driver (used by Phase 6's sweeps)."""

from __future__ import annotations

from fl_ids.data.synthetic import make_synthetic_federated_dataset
from fl_ids.eval.simulation import run_simulated_fl_training
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import (
    AutoencoderConfig,
    BoostingConfig,
    CascadeConfig,
    Config,
    DataConfig,
    EvaluationConfig,
    FLConfig,
    LoggingConfig,
    RobustnessConfig,
    SDNConfig,
)


def _config(num_clients: int) -> Config:
    return Config(
        seed=42,
        data=DataConfig(
            dnn_csv_path="unused.csv", pcap_dir="unused", num_clients=num_clients, dirichlet_alpha=0.3,
            val_benign_fraction=0.2, test_fraction=0.1, normalize_per_client=True,
        ),
        boosting=BoostingConfig(
            label_source="server_held_calibration_set", calibration_fraction=0.1,
            num_boost_round=50, learning_rate=0.1, num_leaves=31,
            broadcast_every_n_rounds=1, update_every_n_rounds=3,
        ),
        autoencoder=AutoencoderConfig(
            bottleneck_dim=4, hidden_dims=[16, 8], learning_rate=0.02, local_epochs=3, batch_size=32,
            anomaly_percentile=97, reconstruction_error_bins=20, reconstruction_error_range=(0.0, 5.0),
        ),
        cascade=CascadeConfig(confidence_threshold=0.6, anomaly_confidence_clip=(0.0, 1.0)),
        fl=FLConfig(num_rounds=5, clients_per_round=num_clients, local_epochs=3, strategy="custom_trust_filtered"),
        robustness=RobustnessConfig(
            trim_fraction=0.15, norm_clip_multiplier=2.0, mad_outlier_threshold=3.0, trust_ema_alpha=0.3
        ),
        sdn=SDNConfig(
            block_confidence_threshold=0.85, rate_limit_confidence_threshold=0.5,
            controller_host="127.0.0.1", controller_port=6653,
            bridge_api_host="127.0.0.1", bridge_api_port=8080,
        ),
        evaluation=EvaluationConfig(poisoning_fractions=[0.0], output_dir="results"),
        logging=LoggingConfig(level="WARNING", log_dir="logs", log_file="test.log"),
    )


def _build_boosting_and_clients(seed: int, num_clients: int):
    config = _config(num_clients)
    client_data, class_names = make_synthetic_federated_dataset(
        config.data, seed=seed, n_samples=6000, n_features=12
    )
    boosting_config = BoostingConfig(
        label_source="server_held_calibration_set", calibration_fraction=0.1,
        num_boost_round=50, learning_rate=0.1, num_leaves=31,
        broadcast_every_n_rounds=1, update_every_n_rounds=3,
    )
    X_all = client_data[0]["X_raw"]
    y_all = client_data[0]["y"]
    boosting_model = BoostingClassifier(
        boosting_config, len(class_names), benign_class=0, seed=seed,
        confidence_threshold=config.cascade.confidence_threshold,
    )
    boosting_model.train(X_all, y_all)
    return config, client_data, class_names, boosting_model


def test_simulation_trust_filtered_runs_and_reduces_loss():
    config, client_data, class_names, boosting_model = _build_boosting_and_clients(seed=1, num_clients=4)

    result = run_simulated_fl_training(
        client_data, boosting_model, len(class_names), benign_class=0, config=config,
        num_rounds=5, aggregation="trust_filtered", seed=1,
    )

    assert len(result.rounds) == 5
    assert result.rounds[-1].mean_val_loss < result.rounds[0].mean_val_loss
    assert result.total_communication_bytes > 0
    assert 1 <= result.rounds_to_convergence <= 5
    assert result.rounds[0].trust_scores is not None


def test_simulation_fedavg_mode_has_no_trust_scores():
    config, client_data, class_names, boosting_model = _build_boosting_and_clients(seed=2, num_clients=4)

    result = run_simulated_fl_training(
        client_data, boosting_model, len(class_names), benign_class=0, config=config,
        num_rounds=3, aggregation="fedavg", seed=2,
    )

    assert len(result.rounds) == 3
    assert all(r.trust_scores is None for r in result.rounds)


def test_simulation_with_malicious_clients_still_completes():
    config, client_data, class_names, boosting_model = _build_boosting_and_clients(seed=3, num_clients=5)

    result = run_simulated_fl_training(
        client_data, boosting_model, len(class_names), benign_class=0, config=config,
        num_rounds=4, aggregation="trust_filtered", malicious_client_ids={0}, seed=3,
    )

    assert len(result.rounds) == 4
    # Attacker's trust should end up lower than at least one honest client's.
    final_trust = result.rounds[-1].trust_scores
    assert final_trust[0] < max(v for cid, v in final_trust.items() if cid != 0)


def test_simulation_use_boosting_filter_false_trains_on_all_data():
    config, client_data, class_names, boosting_model = _build_boosting_and_clients(seed=4, num_clients=3)

    result = run_simulated_fl_training(
        client_data, boosting_model, len(class_names), benign_class=0, config=config,
        num_rounds=2, aggregation="trust_filtered", use_boosting_filter=False, seed=4,
    )
    assert len(result.rounds) == 2
