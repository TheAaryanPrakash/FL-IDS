"""Tests for the boosting classifier (component 2), Phase 2.

Covers: cold-start training on a small server-held calibration slice with
metrics computed on a held-out synthetic test slice (the Phase 2
milestone), the shared cascade decision-rule helper, and the
serialize/deserialize distribution mechanism used to broadcast the model
to clients — verified in isolation, without any real Flower networking.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import BoostingConfig

BENIGN_CLASS = 0  # "Normal" is index 0 in make_synthetic_attack_dataset's default classes


def _boosting_config(**overrides) -> BoostingConfig:
    defaults = dict(
        label_source="server_held_calibration_set",
        calibration_fraction=0.05,
        confidence_threshold=0.7,
        num_boost_round=100,
        learning_rate=0.1,
        num_leaves=31,
        broadcast_every_n_rounds=1,
        update_every_n_rounds=3,
    )
    defaults.update(overrides)
    return BoostingConfig(**defaults)


def test_boosting_classifier_trains_and_predicts_valid_probabilities():
    X, y, class_names = make_synthetic_attack_dataset(n_samples=2000, seed=1)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

    clf = BoostingClassifier(_boosting_config(), num_classes=len(class_names), benign_class=BENIGN_CLASS, seed=1)
    assert not clf.is_trained
    clf.train(X_train, y_train)
    assert clf.is_trained

    proba = clf.predict_proba(X_test)
    assert proba.shape == (len(X_test), len(class_names))
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_predict_proba_before_training_raises():
    clf = BoostingClassifier(_boosting_config(), num_classes=5, benign_class=BENIGN_CLASS, seed=1)
    with pytest.raises(RuntimeError):
        clf.predict_proba(np.zeros((3, 4)))


def test_cold_start_bootstrap_metrics_on_held_out_slice():
    """Phase 2 milestone: trains on a small server-held calibration slice
    (simulating cold start, per component 2), evaluates real
    precision/recall/F1 on a much larger held-out synthetic slice, and
    asserts the model is reasonably decent — not just that it runs.
    """
    config = _boosting_config()
    X, y, class_names = make_synthetic_attack_dataset(n_samples=30_000, seed=2)

    # calibration_fraction simulates the small server-held labeled set the
    # model is bootstrapped on, before any FL round has happened.
    X_calib, X_held_out, y_calib, y_held_out = train_test_split(
        X, y, train_size=config.calibration_fraction, random_state=2, stratify=y
    )

    clf = BoostingClassifier(config, num_classes=len(class_names), benign_class=BENIGN_CLASS, seed=2)
    clf.train(X_calib, y_calib)

    y_pred = np.argmax(clf.predict_proba(X_held_out), axis=1)
    accuracy = (y_pred == y_held_out).mean()
    weighted_f1 = f1_score(y_held_out, y_pred, average="weighted", zero_division=0)
    per_class_precision, per_class_recall, _, _ = precision_recall_fscore_support(
        y_held_out, y_pred, average=None, zero_division=0
    )

    assert len(X_calib) < len(X_held_out) * 0.1, "calibration slice should be small relative to held-out data"
    # Weighted F1/accuracy (dominated by the majority "Normal" class, as in
    # the real dataset) is the fair "reasonably decent" bar here — unlike
    # macro F1, it doesn't weight synthetic classes with only ~20-90
    # calibration examples equally against classes with 800+.
    assert weighted_f1 > 0.65, f"cold-start bootstrap model should be reasonably decent, got weighted F1={weighted_f1:.3f}"
    assert accuracy > 0.65, f"cold-start bootstrap model should be reasonably decent, got accuracy={accuracy:.3f}"
    # Every class should show *some* real signal from the small calibration
    # set — none should be completely unlearned.
    assert np.all(per_class_precision > 0.0), "every class should have some non-zero precision"
    assert np.all(per_class_recall > 0.0), "every class should have some non-zero recall"


def test_cascade_stage1_confidence_threshold_controls_fallthrough():
    X, y, class_names = make_synthetic_attack_dataset(n_samples=4000, seed=3)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=3, stratify=y)

    # Well-separated synthetic classes -> the model should be confident on
    # correctly-classified attack samples.
    high_threshold_config = _boosting_config(confidence_threshold=0.999)
    low_threshold_config = _boosting_config(confidence_threshold=0.01)

    clf_high = BoostingClassifier(high_threshold_config, len(class_names), BENIGN_CLASS, seed=3)
    clf_high.train(X_train, y_train)
    result_high = clf_high.predict_cascade_stage1(X_test)

    clf_low = BoostingClassifier(low_threshold_config, len(class_names), BENIGN_CLASS, seed=3)
    clf_low._booster = clf_high._booster  # identical underlying model, different threshold only
    result_low = clf_low.predict_cascade_stage1(X_test)

    # Benign predictions never count as a confident attack, regardless of threshold.
    benign_predicted = result_high.predicted_class == BENIGN_CLASS
    assert not np.any(result_high.is_confident_attack[benign_predicted])
    assert not np.any(result_low.is_confident_attack[benign_predicted])

    # A near-impossible threshold should confidently-flag far fewer samples
    # than a near-zero threshold.
    assert result_high.is_confident_attack.sum() < result_low.is_confident_attack.sum()

    # Every non-benign prediction is confident under the near-zero threshold.
    non_benign_predicted = result_low.predicted_class != BENIGN_CLASS
    assert np.all(result_low.is_confident_attack[non_benign_predicted])


def test_passes_to_autoencoder_is_complement_of_confident_attack():
    X, y, class_names = make_synthetic_attack_dataset(n_samples=2000, seed=4)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=4, stratify=y)

    clf = BoostingClassifier(_boosting_config(), len(class_names), BENIGN_CLASS, seed=4)
    clf.train(X_train, y_train)

    stage1 = clf.predict_cascade_stage1(X_test)
    passes = clf.passes_to_autoencoder(X_test)
    assert np.array_equal(passes, ~stage1.is_confident_attack)


def test_serialization_round_trip_matches_original_predictions():
    """Distribution mechanism check: the server->client broadcast payload
    (serialized model bytes) must be reconstructable into a classifier
    that predicts identically to the original — testable without any
    real Flower networking.
    """
    X, y, class_names = make_synthetic_attack_dataset(n_samples=2000, seed=5)
    X_train, X_test, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=5, stratify=y)

    config = _boosting_config()
    server_clf = BoostingClassifier(config, len(class_names), BENIGN_CLASS, seed=5)
    server_clf.train(X_train, y_train)

    payload = server_clf.to_bytes()
    assert isinstance(payload, bytes)
    assert len(payload) > 0

    client_clf = BoostingClassifier.from_bytes(
        payload, config, num_classes=len(class_names), benign_class=BENIGN_CLASS, seed=5
    )
    assert client_clf.is_trained

    server_proba = server_clf.predict_proba(X_test)
    client_proba = client_clf.predict_proba(X_test)
    assert np.allclose(server_proba, client_proba, atol=1e-6)

    server_stage1 = server_clf.predict_cascade_stage1(X_test)
    client_stage1 = client_clf.predict_cascade_stage1(X_test)
    assert np.array_equal(server_stage1.is_confident_attack, client_stage1.is_confident_attack)


def test_to_bytes_before_training_raises():
    clf = BoostingClassifier(_boosting_config(), num_classes=5, benign_class=BENIGN_CLASS, seed=1)
    with pytest.raises(RuntimeError):
        clf.to_bytes()
