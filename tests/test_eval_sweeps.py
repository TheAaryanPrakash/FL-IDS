"""Fast tests for the poisoning-resistance sweep and ablation table (component 12).

Uses tiny synthetic-scale settings (few clients, few rounds) so these
run quickly in CI — the actual milestone deliverable (real CSV/plot from
real data) is produced by running fl_ids.eval.poisoning_sweep /
fl_ids.eval.ablation as scripts, checked separately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fl_ids.data.pipeline import partition_and_normalize_clients
from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.eval.ablation import run_ablation
from fl_ids.eval.common import EvaluationSetup
from fl_ids.eval.poisoning_sweep import plot_poisoning_sweep, run_poisoning_sweep
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


def _config(num_clients: int, poisoning_fractions: list[float]) -> Config:
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
            bottleneck_dim=4, hidden_dims=[16, 8], learning_rate=0.02, local_epochs=2, batch_size=32,
            anomaly_percentile=97, reconstruction_error_bins=20, reconstruction_error_range=(0.0, 5.0),
        ),
        cascade=CascadeConfig(confidence_threshold=0.6, anomaly_confidence_clip=(0.0, 1.0)),
        fl=FLConfig(num_rounds=3, clients_per_round=num_clients, local_epochs=2, strategy="custom_trust_filtered"),
        robustness=RobustnessConfig(
            trim_fraction=0.15, norm_clip_multiplier=2.0, mad_outlier_threshold=3.0, trust_ema_alpha=0.3
        ),
        sdn=SDNConfig(
            block_confidence_threshold=0.85, rate_limit_confidence_threshold=0.5,
            controller_host="127.0.0.1", controller_port=6653,
            bridge_api_host="127.0.0.1", bridge_api_port=8080,
        ),
        evaluation=EvaluationConfig(poisoning_fractions=poisoning_fractions, output_dir="results"),
        logging=LoggingConfig(level="WARNING", log_dir="logs", log_file="test.log"),
    )


def _build_setup(seed: int, num_clients: int, poisoning_fractions: list[float]) -> tuple[Config, EvaluationSetup]:
    config = _config(num_clients, poisoning_fractions)
    X, y, class_names = make_synthetic_attack_dataset(n_samples=8000, n_features=12, seed=seed)

    from sklearn.model_selection import train_test_split

    X_calib, X_rest, y_calib, y_rest = train_test_split(X, y, train_size=0.1, random_state=seed, stratify=y)
    X_pool, X_test, y_pool, y_test = train_test_split(X_rest, y_rest, test_size=0.2, random_state=seed, stratify=y_rest)

    boosting_model = BoostingClassifier(
        config.boosting, len(class_names), benign_class=0, seed=seed,
        confidence_threshold=config.cascade.confidence_threshold,
    )
    boosting_model.train(X_calib, y_calib)

    client_data = partition_and_normalize_clients(X_pool, y_pool, benign_class=0, config=config.data, seed=seed)

    setup = EvaluationSetup(
        client_data=client_data,
        boosting_model=boosting_model,
        class_names=class_names,
        benign_class=0,
        input_dim=X.shape[1],
        X_test_raw=X_test,
        X_test_norm=X_test,  # synthetic data is already roughly standardized; fine for this fast test
        y_test=y_test,
    )
    return config, setup


def test_poisoning_sweep_runs_and_produces_expected_columns():
    config, setup = _build_setup(seed=1, num_clients=5, poisoning_fractions=[0.0, 0.4])
    df = run_poisoning_sweep(setup, config, num_rounds=3, seed=1)

    assert len(df) == 2
    expected_cols = {
        "malicious_fraction", "accuracy", "weighted_f1", "false_positive_rate",
        "rounds_to_convergence", "total_communication_bytes", "final_val_loss",
        "autoencoder_alone_attack_recall", "malicious_client_survival_rate",
    }
    assert expected_cols.issubset(df.columns)
    assert (df["accuracy"] >= 0.0).all() and (df["accuracy"] <= 1.0).all()


def test_poisoning_sweep_plot_saves_a_real_file(tmp_path):
    config, setup = _build_setup(seed=2, num_clients=5, poisoning_fractions=[0.0, 0.3])
    df = run_poisoning_sweep(setup, config, num_rounds=2, seed=2)

    output_path = tmp_path / "sweep.png"
    plot_poisoning_sweep(df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_ablation_runs_and_produces_five_variants():
    config, setup = _build_setup(seed=3, num_clients=5, poisoning_fractions=[0.3])
    df = run_ablation(setup, config, num_rounds=3, poisoning_fraction=0.3, seed=3)

    assert set(df["variant"]) == {
        "full_pipeline", "plain_fedavg", "trimmed_mean_only", "autoencoder_only", "boosting_only",
    }
    assert (df["accuracy"] >= 0.0).all() and (df["accuracy"] <= 1.0).all()
    # boosting_only does no FL training -- no communication cost.
    assert df.loc[df["variant"] == "boosting_only", "total_communication_bytes"].iloc[0] == 0
