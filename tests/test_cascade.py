"""Tests for the full cascade decision rule (boosting + autoencoder combined)."""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split

from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.models.autoencoder import Autoencoder, compute_anomaly_threshold, reconstruction_error, train_autoencoder
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.models.cascade import ANOMALOUS_LABEL, BENIGN_LABEL, cascade_predict
from fl_ids.utils.config import AutoencoderConfig, BoostingConfig, CascadeConfig

BENIGN_CLASS = 0


def _boosting_config() -> BoostingConfig:
    return BoostingConfig(
        label_source="server_held_calibration_set",
        calibration_fraction=0.05,
        num_boost_round=100,
        learning_rate=0.1,
        num_leaves=31,
        broadcast_every_n_rounds=1,
        update_every_n_rounds=3,
    )


def _autoencoder_config() -> AutoencoderConfig:
    return AutoencoderConfig(
        bottleneck_dim=4,
        hidden_dims=[16, 8],
        learning_rate=0.01,
        local_epochs=10,
        batch_size=32,
        anomaly_percentile=97,
        reconstruction_error_bins=20,
        reconstruction_error_range=(0.0, 5.0),
    )


def _build_pipeline(seed: int = 1):
    X, y, class_names = make_synthetic_attack_dataset(n_samples=5000, n_features=12, seed=seed)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)

    boosting_config = _boosting_config()
    boosting_model = BoostingClassifier(
        boosting_config, len(class_names), BENIGN_CLASS, seed=seed, confidence_threshold=0.7
    )
    boosting_model.train(X_train, y_train)

    # Autoencoder trained only on boosting-filtered "normal" training data.
    mask = boosting_model.passes_to_autoencoder(X_train)
    ae_config = _autoencoder_config()
    autoencoder = Autoencoder(X.shape[1], ae_config.hidden_dims, ae_config.bottleneck_dim)
    train_autoencoder(autoencoder, X_train[mask], ae_config, seed=seed)

    val_benign = X_train[(y_train == BENIGN_CLASS) & mask]
    threshold = compute_anomaly_threshold(
        reconstruction_error(autoencoder, val_benign), ae_config.anomaly_percentile
    )

    cascade_config = CascadeConfig(confidence_threshold=0.7, anomaly_confidence_clip=(0.0, 1.0))
    return boosting_model, autoencoder, threshold, cascade_config, class_names, X_test, y_test


def test_cascade_confident_boosting_predictions_skip_autoencoder():
    boosting_model, autoencoder, threshold, cascade_config, class_names, X_test, y_test = _build_pipeline()

    output = cascade_predict(
        boosting_model, autoencoder, threshold, X_test, X_test, class_names, cascade_config
    )

    stage1 = boosting_model.predict_cascade_stage1(X_test)
    for i in np.where(stage1.is_confident_attack)[0]:
        assert output.stage[i] == "boosting"
        assert output.predicted_label[i] == class_names[stage1.predicted_class[i]]
        assert output.confidence[i] == stage1.confidence[i]


def test_cascade_fallthrough_uses_autoencoder_and_labels_benign_or_anomalous():
    boosting_model, autoencoder, threshold, cascade_config, class_names, X_test, y_test = _build_pipeline()

    output = cascade_predict(
        boosting_model, autoencoder, threshold, X_test, X_test, class_names, cascade_config
    )

    stage1 = boosting_model.predict_cascade_stage1(X_test)
    fallthrough_idx = np.where(~stage1.is_confident_attack)[0]
    assert len(fallthrough_idx) > 0, "test setup should produce some fallthrough samples"

    for i in fallthrough_idx:
        assert output.stage[i] == "autoencoder"
        assert output.predicted_label[i] in (BENIGN_LABEL, ANOMALOUS_LABEL)
        assert 0.0 <= output.confidence[i] <= 1.0


def test_cascade_never_outputs_specific_attack_type_from_autoencoder_stage():
    """Per spec: the autoencoder stage deliberately never names a specific
    attack type -- only "anomalous" (a zero-day signal) or "benign".
    """
    boosting_model, autoencoder, threshold, cascade_config, class_names, X_test, y_test = _build_pipeline()
    output = cascade_predict(
        boosting_model, autoencoder, threshold, X_test, X_test, class_names, cascade_config
    )
    autoencoder_stage_labels = set(output.predicted_label[output.stage == "autoencoder"])
    assert autoencoder_stage_labels <= {BENIGN_LABEL, ANOMALOUS_LABEL}


def test_cascade_confidence_always_in_unit_range():
    boosting_model, autoencoder, threshold, cascade_config, class_names, X_test, y_test = _build_pipeline()
    output = cascade_predict(
        boosting_model, autoencoder, threshold, X_test, X_test, class_names, cascade_config
    )
    assert np.all(output.confidence >= 0.0)
    assert np.all(output.confidence <= 1.0)


def test_cascade_handles_infinite_threshold_safely():
    """No benign validation data -> threshold=inf -> nothing should ever be
    flagged anomalous (can't safely calibrate without real data).
    """
    boosting_model, autoencoder, _, cascade_config, class_names, X_test, y_test = _build_pipeline()
    output = cascade_predict(
        boosting_model, autoencoder, float("inf"), X_test, X_test, class_names, cascade_config
    )
    autoencoder_stage = output.stage == "autoencoder"
    assert np.all(output.predicted_label[autoencoder_stage] == BENIGN_LABEL)
