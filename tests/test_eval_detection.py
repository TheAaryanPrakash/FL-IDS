"""Tests for the shared detection scoring behind component 12's comparisons.

Covers the pieces every ablation/sweep/zero-day row goes through:
per-client thresholds and normalization, `flag_attacks`/`detection_metrics`,
per-row thresholds in the cascade rule, and convergence with divergence
detection.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.eval.common import (
    assign_rows_to_clients,
    build_evaluation_setup_from_arrays,
    calibrate_client_thresholds,
    per_row_thresholds,
)
from fl_ids.eval.metrics import detection_metrics, flag_attacks
from fl_ids.eval.simulation import SimulationResult, SimulationRoundRecord
from fl_ids.models.autoencoder import Autoencoder, compute_anomaly_threshold, get_weights, reconstruction_error
from fl_ids.models.cascade import cascade_predict
from tests.test_eval_sweeps import _config

BENIGN = 0


@pytest.fixture(scope="module")
def setup_and_config():
    config = _config(num_clients=4, poisoning_fractions=[0.0])
    X, y, class_names = make_synthetic_attack_dataset(n_samples=6000, n_features=12, seed=5)
    return build_evaluation_setup_from_arrays(X, y, class_names, BENIGN, config, seed=5), config


@pytest.fixture(scope="module")
def trained_autoencoder(setup_and_config):
    setup, config = setup_and_config
    model = Autoencoder(setup.input_dim, config.autoencoder.hidden_dims, config.autoencoder.bottleneck_dim)
    return calibrate_client_thresholds(get_weights(model), setup, config)


def test_test_set_is_each_clients_own_slice_normalized_by_its_own_scaler(setup_and_config):
    setup, _ = setup_and_config
    for cid, data in setup.client_data.items():
        rows = setup.test_client_ids == cid
        np.testing.assert_array_equal(setup.y_test[rows], data["y_test"])
        np.testing.assert_allclose(setup.X_test_norm[rows], data["X_test"], rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(
            setup.client_scalers[cid].transform(setup.X_test_raw[rows]), data["X_test"], rtol=1e-4, atol=1e-4
        )


def test_each_client_gets_its_own_threshold_from_its_own_benign_slice(setup_and_config, trained_autoencoder):
    setup, config = setup_and_config
    autoencoder, thresholds = trained_autoencoder
    for cid, data in setup.client_data.items():
        if len(data["X_val_benign"]) == 0:
            assert thresholds[cid] == float("inf")
        else:
            expected = compute_anomaly_threshold(
                reconstruction_error(autoencoder, data["X_val_benign"]), config.autoencoder.anomaly_percentile
            )
            assert thresholds[cid] == pytest.approx(expected)
    row_thresholds = per_row_thresholds(thresholds, setup.test_client_ids)
    assert row_thresholds[0] == thresholds[setup.test_client_ids[0]]


def test_rows_from_outside_the_federation_use_their_assigned_clients_scaler(setup_and_config):
    setup, _ = setup_and_config
    X_raw = setup.X_test_raw[:200]
    X_norm, client_ids = assign_rows_to_clients(setup, X_raw, seed=0)
    assert len(set(client_ids)) > 1
    for cid in set(client_ids):
        rows = client_ids == cid
        np.testing.assert_allclose(X_norm[rows], setup.client_scalers[cid].transform(X_raw[rows]), rtol=1e-5)


def test_cascade_rule_accepts_per_row_thresholds(setup_and_config, trained_autoencoder):
    setup, config = setup_and_config
    autoencoder, _ = trained_autoencoder
    X_raw, X_norm = setup.X_test_raw[:300], setup.X_test_norm[:300]
    args = (setup.boosting_model, autoencoder)
    rest = (X_raw, X_norm, setup.class_names, config.cascade)

    scalar = cascade_predict(*args, 0.5, *rest)
    constant_array = cascade_predict(*args, np.full(300, 0.5), *rest)
    np.testing.assert_array_equal(scalar.predicted_label, constant_array.predicted_label)
    np.testing.assert_allclose(scalar.confidence, constant_array.confidence)

    # An infinite per-row threshold (uncalibrated client) never yields "anomalous".
    thresholds = np.full(300, 1e-9)
    thresholds[:150] = np.inf
    mixed = cascade_predict(*args, thresholds, *rest)
    assert not (mixed.predicted_label[:150] == "anomalous").any()


def test_flag_attacks_modes(setup_and_config, trained_autoencoder):
    setup, config = setup_and_config
    autoencoder, thresholds = trained_autoencoder
    row_thresholds = per_row_thresholds(thresholds, setup.test_client_ids)
    common = (setup.boosting_model, autoencoder, row_thresholds, setup.X_test_raw, setup.X_test_norm,
              setup.class_names, config.cascade)

    boosting = flag_attacks("boosting_only", *common)
    cascade = flag_attacks("cascade", *common)
    autoencoder_only = flag_attacks("autoencoder_only", *common)

    np.testing.assert_array_equal(boosting, setup.boosting_model.predict_cascade_stage1(setup.X_test_raw).is_confident_attack)
    # The backstop only adds flags on what boosting let through.
    assert (cascade >= boosting).all()
    np.testing.assert_array_equal(autoencoder_only, reconstruction_error(autoencoder, setup.X_test_norm) > row_thresholds)
    with pytest.raises(ValueError):
        flag_attacks("nope", *common)


def test_detection_metrics_macro_recall_weights_every_attack_type_equally():
    names = ["Normal", "Common", "Rare"]
    y = np.array([0] * 10 + [1] * 90 + [2] * 10)
    flags = np.zeros(len(y), dtype=bool)
    flags[10:100] = True  # every Common row, no Rare rows
    flags[0] = True  # one benign false positive

    m = detection_metrics(flags, y, benign_class=0, class_names=names)

    assert m["per_class_recall"] == {"Common": 1.0, "Rare": 0.0}
    assert m["attack_macro_recall"] == pytest.approx(0.5)
    assert m["attack_micro_recall"] == pytest.approx(0.9)
    assert m["benign_fpr"] == pytest.approx(0.1)
    precision, recall = 90 / 91, 0.9
    assert m["detection_f1"] == pytest.approx(2 * precision * recall / (precision + recall))


def _result(losses: list[float]) -> SimulationResult:
    rounds = [SimulationRoundRecord(round=i + 1, mean_val_loss=loss, communication_bytes=0) for i, loss in enumerate(losses)]
    return SimulationResult(final_weights=[], rounds=rounds)


@pytest.mark.parametrize(
    ("losses", "expected"),
    [
        ([1.0, 0.5, 0.3, 0.29, 0.3], 3.0),  # settles in round 3 and stays
        ([0.5, 0.3, 0.6, 0.3, 0.31], 4.0),  # leaves the band once, so it counts from after that
        ([0.5], 1.0),
    ],
)
def test_rounds_to_convergence(losses, expected):
    assert _result(losses).rounds_to_convergence(tolerance=0.1) == expected


@pytest.mark.parametrize("losses", [[0.7, 5.0, 1e14, 5e14], [0.3, 0.2, float("nan")], [0.2, 0.1, 0.5]])
def test_diverged_or_unfinished_runs_do_not_count_as_converged(losses):
    assert np.isnan(_result(losses).rounds_to_convergence(tolerance=0.1))
