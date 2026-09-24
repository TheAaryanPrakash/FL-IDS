"""Fast tests for the poisoning-resistance sweep and ablation table (component 12).

Uses tiny synthetic-scale settings (few clients, few rounds) so these
run quickly in CI — the actual milestone deliverable (real CSV/plot from
real data) is produced by running fl_ids.eval.poisoning_sweep /
fl_ids.eval.ablation as scripts, checked separately.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.eval.ablation import TABLE_COLUMNS, run_ablation
from fl_ids.eval.common import EvaluationSetup, build_evaluation_setup_from_arrays
from fl_ids.eval.metrics import detection_metrics
from fl_ids.eval.poisoning_sweep import SWEEP_COLUMNS, plot_poisoning_sweep, run_poisoning_sweep
from fl_ids.eval.poisoning_sweep import add_zero_day_column as add_sweep_zero_day
from fl_ids.eval.variants import ABLATION_VARIANTS
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
    setup = build_evaluation_setup_from_arrays(X, y, class_names, 0, config, seed)
    return config, setup


def test_poisoning_sweep_scores_every_fraction_with_shared_detection_metrics():
    config, setup = _build_setup(seed=1, num_clients=5, poisoning_fractions=[0.0, 0.4])
    df = run_poisoning_sweep(setup, config, num_rounds=3, seed=1)

    assert list(df.columns) == SWEEP_COLUMNS
    assert list(df["malicious_fraction"]) == [0.0, 0.4]
    for col in ("attack_macro_recall", "benign_fpr", "autoencoder_alone_attack_macro_recall"):
        assert df[col].between(0.0, 1.0).all(), col
    assert np.isnan(df.loc[0, "malicious_client_survival_rate"])  # no attackers at 0%
    assert df.loc[1, "malicious_client_survival_rate"] >= 0.0
    assert df["zero_day_macro_recall"].isna().all()  # filled only by add_zero_day_column


def test_poisoning_sweep_zero_day_column_is_filled(synthetic_for_zero_day):
    X, y, class_names = synthetic_for_zero_day
    config, setup = _build_setup(seed=1, num_clients=5, poisoning_fractions=[0.0, 0.4])
    config.evaluation.zero_day_holdout_classes = ["Uploading"]
    config.evaluation.zero_day_max_holdout_rows = 100
    df = run_poisoning_sweep(setup, config, num_rounds=2, seed=1)

    df, detail = add_sweep_zero_day(df, X, y, class_names, 0, config, num_rounds=2, seed=1)

    assert df["zero_day_macro_recall"].between(0.0, 1.0).all()
    assert len(detail) == 2 and set(detail["holdout_class"]) == {"Uploading"}


def test_poisoning_sweep_plot_saves_a_real_file(tmp_path):
    config, setup = _build_setup(seed=2, num_clients=5, poisoning_fractions=[0.0, 0.3])
    df = run_poisoning_sweep(setup, config, num_rounds=2, seed=2)

    output_path = tmp_path / "sweep.png"
    plot_poisoning_sweep(df, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_ablation_scores_all_five_variants_the_same_way():
    config, setup = _build_setup(seed=3, num_clients=5, poisoning_fractions=[0.3])
    table, per_class = run_ablation(setup, config, num_rounds=3, poisoning_fraction=0.3, seed=3)

    assert list(table.columns) == TABLE_COLUMNS
    assert list(table["variant"]) == [v.name for v in ABLATION_VARIANTS]
    assert table["attack_macro_recall"].between(0.0, 1.0).all()
    assert table["benign_fpr"].between(0.0, 1.0).all()
    boosting_only = table.set_index("variant").loc["boosting_only"]
    # No FL training -- no communication cost, no autoencoder to report on.
    assert boosting_only["total_communication_bytes"] == 0
    assert np.isnan(boosting_only["autoencoder_alone_attack_macro_recall"])
    # One recall per (variant, attack type present in the test set).
    n_attack_types = len(set(setup.y_test) - {setup.benign_class})
    assert len(per_class) == len(ABLATION_VARIANTS) * n_attack_types


def test_boosting_only_row_uses_the_cascade_confidence_rule_not_argmax():
    config, setup = _build_setup(seed=3, num_clients=5, poisoning_fractions=[0.3])
    table, _ = run_ablation(setup, config, num_rounds=2, poisoning_fraction=0.3, seed=3)

    stage1 = setup.boosting_model.predict_cascade_stage1(setup.X_test_raw)
    expected = detection_metrics(stage1.is_confident_attack, setup.y_test, setup.benign_class, setup.class_names)
    row = table.set_index("variant").loc["boosting_only"]
    assert row["attack_macro_recall"] == pytest.approx(expected["attack_macro_recall"])
    assert row["benign_fpr"] == pytest.approx(expected["benign_fpr"])


@pytest.fixture(scope="module")
def synthetic_for_zero_day():
    return make_synthetic_attack_dataset(n_samples=8000, n_features=12, seed=1)
