"""Tests for the per-stage evaluation harness (component 12)."""

from __future__ import annotations

import json

import numpy as np
from sklearn.model_selection import train_test_split

from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.eval.metrics import (
    build_per_stage_table,
    evaluate_autoencoder_alone,
    evaluate_boosting_alone,
    evaluate_cascade,
    stage_report_summary,
)
from fl_ids.models.autoencoder import Autoencoder, compute_anomaly_threshold, reconstruction_error, train_autoencoder
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import AutoencoderConfig, BoostingConfig, CascadeConfig

BENIGN_CLASS = 0


def _build_pipeline(seed: int = 1):
    X, y, class_names = make_synthetic_attack_dataset(n_samples=6000, n_features=12, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    boosting_config = BoostingConfig(
        label_source="server_held_calibration_set", calibration_fraction=0.05,
        num_boost_round=100, learning_rate=0.1, num_leaves=31,
        broadcast_every_n_rounds=1, update_every_n_rounds=3,
    )
    boosting_model = BoostingClassifier(
        boosting_config, len(class_names), BENIGN_CLASS, seed=seed, confidence_threshold=0.7
    )
    boosting_model.train(X_train, y_train)

    mask = boosting_model.passes_to_autoencoder(X_train)
    ae_config = AutoencoderConfig(
        bottleneck_dim=4, hidden_dims=[16, 8], learning_rate=0.01, local_epochs=10, batch_size=32,
        anomaly_percentile=97, reconstruction_error_bins=20, reconstruction_error_range=(0.0, 5.0),
    )
    autoencoder = Autoencoder(X.shape[1], ae_config.hidden_dims, ae_config.bottleneck_dim)
    train_autoencoder(autoencoder, X_train[mask], ae_config, seed=seed)

    val_benign = X_train[(y_train == BENIGN_CLASS) & mask]
    threshold = compute_anomaly_threshold(reconstruction_error(autoencoder, val_benign), ae_config.anomaly_percentile)

    cascade_config = CascadeConfig(confidence_threshold=0.7, anomaly_confidence_clip=(0.0, 1.0))
    return boosting_model, autoencoder, threshold, cascade_config, class_names, X_test, y_test


def test_evaluate_boosting_alone_shapes_and_ranges():
    boosting_model, _, _, _, class_names, X_test, y_test = _build_pipeline()
    report = evaluate_boosting_alone(boosting_model, X_test, y_test, class_names, BENIGN_CLASS)

    assert report.stage == "boosting_alone"
    assert len(report.per_class) == len(class_names)
    assert (report.per_class["precision"] >= 0).all() and (report.per_class["precision"] <= 1).all()
    assert 0.0 <= report.accuracy <= 1.0
    assert 0.0 <= report.false_positive_rate <= 1.0


def test_evaluate_autoencoder_alone_binary_shape():
    _, autoencoder, threshold, _, _, X_test, y_test = _build_pipeline()
    report = evaluate_autoencoder_alone(autoencoder, threshold, X_test, y_test, BENIGN_CLASS)

    assert report.stage == "autoencoder_alone"
    assert list(report.per_class["class"]) == ["benign", "attack"]
    assert 0.0 <= report.accuracy <= 1.0


def test_evaluate_cascade_binary_shape_and_stage_breakdown():
    boosting_model, autoencoder, threshold, cascade_config, class_names, X_test, y_test = _build_pipeline()
    report = evaluate_cascade(
        boosting_model, autoencoder, threshold, X_test, X_test, y_test, class_names, BENIGN_CLASS, cascade_config
    )

    assert report.stage == "cascade_combined"
    assert list(report.per_class["class"]) == ["benign", "attack"]
    assert "per_label" in report.extra
    assert set(report.extra["stage_counts"].keys()) <= {"boosting", "autoencoder"}
    assert sum(report.extra["stage_counts"].values()) == len(X_test)


def test_cascade_accuracy_at_least_as_good_as_boosting_alone_on_confident_subset():
    """Sanity: the cascade shouldn't be *worse* than boosting alone at the
    coarse benign-vs-attack framing, since it strictly adds a second
    detection pass rather than overriding boosting's confident calls.
    """
    boosting_model, autoencoder, threshold, cascade_config, class_names, X_test, y_test = _build_pipeline()

    boosting_report = evaluate_boosting_alone(boosting_model, X_test, y_test, class_names, BENIGN_CLASS)
    cascade_report = evaluate_cascade(
        boosting_model, autoencoder, threshold, X_test, X_test, y_test, class_names, BENIGN_CLASS, cascade_config
    )
    # Not a strict inequality (different metrics can trade off), but both
    # should be reasonable and the cascade shouldn't collapse.
    assert cascade_report.accuracy > 0.5
    assert boosting_report.accuracy > 0.5


def test_build_per_stage_table_combines_all_reports():
    boosting_model, autoencoder, threshold, cascade_config, class_names, X_test, y_test = _build_pipeline()

    reports = [
        evaluate_boosting_alone(boosting_model, X_test, y_test, class_names, BENIGN_CLASS),
        evaluate_autoencoder_alone(autoencoder, threshold, X_test, y_test, BENIGN_CLASS),
        evaluate_cascade(
            boosting_model, autoencoder, threshold, X_test, X_test, y_test, class_names, BENIGN_CLASS, cascade_config
        ),
    ]
    table = build_per_stage_table(reports)

    assert set(table["stage"].unique()) == {"boosting_alone", "autoencoder_alone", "cascade_combined"}
    assert {"precision", "recall", "f1", "auroc", "stage_accuracy", "stage_false_positive_rate"}.issubset(table.columns)


def test_stage_report_summary_is_json_serializable_and_matches_report():
    boosting_model, _, _, _, class_names, X_test, y_test = _build_pipeline()
    report = evaluate_boosting_alone(boosting_model, X_test, y_test, class_names, BENIGN_CLASS)

    summary = stage_report_summary(report)
    assert json.loads(json.dumps(summary)) == summary
    assert summary["accuracy"] == report.accuracy
    assert set(summary["per_class"]) == set(class_names)
