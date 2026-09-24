"""Tests for the zero-day (leave-one-attack-class-out) experiment (component 12).

Synthetic data, few clients/rounds, so these run in seconds. The real
deliverable (results/zero_day_holdout.{csv,png} on Edge-IIoTset) comes
from running fl_ids.eval.zero_day as a script.
"""

from __future__ import annotations

import numpy as np
import pytest

from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.eval.common import build_evaluation_setup_from_arrays
from fl_ids.eval.zero_day import plot_zero_day, resolve_holdout_classes, run_zero_day_experiment
from tests.test_eval_sweeps import _config

BENIGN = 0


@pytest.fixture(scope="module")
def synthetic():
    X, y, class_names = make_synthetic_attack_dataset(n_samples=8000, n_features=12, seed=0)
    return X, y, class_names


@pytest.fixture(scope="module")
def zero_day_results(synthetic):
    X, y, class_names = synthetic
    config = _config(num_clients=5, poisoning_fractions=[0.0])
    config.evaluation.zero_day_holdout_classes = ["DDoS_TCP", "Uploading"]
    config.evaluation.zero_day_max_holdout_rows = 150
    return run_zero_day_experiment(X, y, class_names, BENIGN, config, num_rounds=3, seed=0)


def test_excluded_class_never_reaches_any_training_data(synthetic):
    X, y, class_names = synthetic
    config = _config(num_clients=5, poisoning_fractions=[0.0])
    holdout = class_names.index("DDoS_TCP")

    setup = build_evaluation_setup_from_arrays(
        X, y, class_names, BENIGN, config, seed=0, excluded_classes=frozenset({holdout})
    )

    for data in setup.client_data.values():
        assert holdout not in data["y"]
        assert holdout not in data["y_test"]
    # Boosting never saw a single row of it, so it can't assign it real probability.
    holdout_rows = X[y == holdout]
    assert setup.boosting_model.predict_proba(holdout_rows)[:, holdout].max() < 1e-3


def test_exclusion_leaves_the_shared_test_set_unchanged(synthetic):
    X, y, class_names = synthetic
    config = _config(num_clients=5, poisoning_fractions=[0.0])

    baseline = build_evaluation_setup_from_arrays(X, y, class_names, BENIGN, config, seed=0)
    excluded = build_evaluation_setup_from_arrays(
        X, y, class_names, BENIGN, config, seed=0, excluded_classes=frozenset({class_names.index("XSS")})
    )

    np.testing.assert_array_equal(baseline.X_test_raw, excluded.X_test_raw)
    np.testing.assert_array_equal(baseline.y_test, excluded.y_test)


def test_benign_class_cannot_be_excluded(synthetic):
    X, y, class_names = synthetic
    config = _config(num_clients=5, poisoning_fractions=[0.0])
    with pytest.raises(ValueError):
        build_evaluation_setup_from_arrays(X, y, class_names, BENIGN, config, seed=0, excluded_classes=frozenset({BENIGN}))


def test_resolve_holdout_classes():
    names = ["Normal", "A", "B"]
    assert resolve_holdout_classes([], names, benign_class=0) == [1, 2]
    assert resolve_holdout_classes(["B"], names, benign_class=0) == [2]
    with pytest.raises(ValueError):
        resolve_holdout_classes(["Normal"], names, benign_class=0)
    with pytest.raises(ValueError):
        resolve_holdout_classes(["nope"], names, benign_class=0)


def test_cascade_detection_decomposes_into_boosting_plus_autoencoder(zero_day_results):
    # The autoencoder only ever sees what boosting let through, so the
    # cascade's detections are exactly boosting's plus the autoencoder's
    # additions -- nothing double-counted, nothing lost.
    np.testing.assert_allclose(
        zero_day_results["cascade_detection_rate"],
        zero_day_results["boosting_only_detection_rate"] + zero_day_results["autoencoder_added_detection_rate"],
    )


def test_autoencoder_backstop_catches_unseen_attacks_boosting_misses(zero_day_results):
    for _, row in zero_day_results.iterrows():
        assert row["autoencoder_added_detection_rate"] > 0.05, row["holdout_class"]
        assert row["cascade_detection_rate"] > row["boosting_only_detection_rate"], row["holdout_class"]
        # Reconstruction error genuinely ranks unseen attack rows above benign ones.
        assert row["autoencoder_auroc_vs_benign"] > 0.6, row["holdout_class"]
        # Boosting can't name an attack type it was never trained on.
        assert row["boosting_top_misattributed_label"] != row["holdout_class"]


def test_zero_day_reports_the_false_positive_cost(zero_day_results):
    # The backstop's price: it can only add flags, never remove them.
    assert (zero_day_results["cascade_benign_fpr"] >= zero_day_results["boosting_only_benign_fpr"]).all()
    assert (zero_day_results["cascade_known_attack_recall"] >= zero_day_results["boosting_only_known_attack_recall"]).all()


def test_zero_day_plot_saves_a_real_file(zero_day_results, tmp_path):
    output_path = tmp_path / "zero_day.png"
    plot_zero_day(zero_day_results, output_path)
    assert output_path.exists() and output_path.stat().st_size > 0
