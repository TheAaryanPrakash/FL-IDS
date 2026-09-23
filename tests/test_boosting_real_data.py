"""Phase 5: re-validates Phase 2's boosting bootstrap milestone on real data.

Trains on the real server-held calibration set (component 2's label
source, carved out by `build_server_and_federated_dataset`) and
evaluates on the real per-client held-out test slices — the same
cold-start shape as Phase 2's synthetic-data test, just with real
Edge-IIoTset traffic. Skipped (not failed) if the dataset file isn't
present.

Boosting always trains/predicts on *raw* features (see
`fl_ids.models.boosting`'s module docstring) — this test evaluates
against `X_test_raw`, never per-client-normalized `X_test`. An earlier
version of this test evaluated against normalized `X_test` and saw
accuracy collapse to ~0.50 purely from the scale mismatch (see the
module docstring's account of Phase 5's real-data finding), not because
the model was actually bad.

Uses the real-data-tuned hyperparameters from configs/config.yaml
(num_leaves=127, num_boost_round=300, learning_rate=0.1) rather than
Phase 2's synthetic-tuned defaults (31/200/0.05) — see the config
file's comment: with 95 real, unevenly-important features, num_leaves=31
didn't give LightGBM enough per-round tree capacity to reliably split on
every class's discriminating feature once training was made properly
deterministic (see fl_ids.models.boosting's train()).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_recall_fscore_support

from fl_ids.data.pipeline import build_server_and_federated_dataset
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import BoostingConfig, DataConfig

REAL_DATASET_PATH = Path("data/raw/DNN-EdgeIIoT-dataset.csv")


@pytest.mark.skipif(not REAL_DATASET_PATH.exists(), reason="real dataset not present")
def test_cold_start_bootstrap_metrics_on_real_data():
    data_config = DataConfig(
        dnn_csv_path=str(REAL_DATASET_PATH),
        pcap_dir="unused",
        num_clients=8,
        dirichlet_alpha=0.3,
        val_benign_fraction=0.15,
        test_fraction=0.15,
        normalize_per_client=True,
    )
    boosting_config = BoostingConfig(
        label_source="server_held_calibration_set",
        calibration_fraction=0.05,
        num_boost_round=300,
        learning_rate=0.1,
        num_leaves=127,
        broadcast_every_n_rounds=1,
        update_every_n_rounds=3,
    )

    X_calib, y_calib, client_data, label_encoder, feature_names = build_server_and_federated_dataset(
        REAL_DATASET_PATH, data_config, calibration_fraction=boosting_config.calibration_fraction, seed=42
    )
    benign_class = int(label_encoder.transform(["Normal"])[0])

    boosting_model = BoostingClassifier(
        boosting_config, num_classes=len(label_encoder.classes_), benign_class=benign_class, seed=42,
        confidence_threshold=0.7,
    )
    boosting_model.train(X_calib, y_calib)

    X_test = np.concatenate([data["X_test_raw"] for data in client_data.values()])
    y_test = np.concatenate([data["y_test"] for data in client_data.values()])
    assert len(X_test) > 10_000, "should have a substantial real held-out test set"

    y_pred = np.argmax(boosting_model.predict_proba(X_test), axis=1)
    accuracy = (y_pred == y_test).mean()
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    per_class_precision, per_class_recall, _, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, zero_division=0, labels=range(len(label_encoder.classes_))
    )

    print(f"Real-data cold-start: accuracy={accuracy:.3f}, weighted F1={weighted_f1:.3f}")
    for cls_idx, cls_name in enumerate(label_encoder.classes_):
        print(f"  {cls_name}: precision={per_class_precision[cls_idx]:.2f} recall={per_class_recall[cls_idx]:.2f} support={support[cls_idx]}")

    # With real-data-tuned hyperparameters, this comfortably clears the
    # same "reasonably decent" bar Phase 2 set on synthetic data -- raised
    # here to match what's actually and reliably achieved (~0.90 acc /
    # ~0.91 weighted F1), not left at the older, looser synthetic bar.
    assert weighted_f1 > 0.85, f"cold-start bootstrap model should be reasonably decent on real data, got weighted F1={weighted_f1:.3f}"
    assert accuracy > 0.85, f"cold-start bootstrap model should be reasonably decent on real data, got accuracy={accuracy:.3f}"

    # Classes with zero support in the held-out test set, or so few real
    # calibration examples (well under 1000 overall, meaning tens of
    # examples in a 5% calibration slice) that no cold-start model could
    # reasonably learn them, are excluded from the "every class learns
    # something" check -- Fingerprinting (136 total) and MITM (49 total)
    # in the real dataset are genuinely at that floor.
    learnable = support > 1000
    assert np.all(per_class_precision[learnable] > 0.0), (
        f"every class with a meaningful number of real examples should have some non-zero "
        f"precision, got precision={per_class_precision} support={support}"
    )


@pytest.mark.skipif(not REAL_DATASET_PATH.exists(), reason="real dataset not present")
def test_boosting_distribution_mechanism_on_real_data():
    """Re-validates Phase 2's serialize/deserialize round-trip on real data."""
    data_config = DataConfig(
        dnn_csv_path=str(REAL_DATASET_PATH),
        pcap_dir="unused",
        num_clients=5,
        dirichlet_alpha=0.3,
        val_benign_fraction=0.15,
        test_fraction=0.15,
        normalize_per_client=True,
    )
    boosting_config = BoostingConfig(
        label_source="server_held_calibration_set",
        calibration_fraction=0.05,
        num_boost_round=300,
        learning_rate=0.1,
        num_leaves=127,
        broadcast_every_n_rounds=1,
        update_every_n_rounds=3,
    )

    X_calib, y_calib, client_data, label_encoder, _ = build_server_and_federated_dataset(
        REAL_DATASET_PATH, data_config, calibration_fraction=boosting_config.calibration_fraction, seed=7
    )
    benign_class = int(label_encoder.transform(["Normal"])[0])

    server_model = BoostingClassifier(
        boosting_config, num_classes=len(label_encoder.classes_), benign_class=benign_class, seed=7,
        confidence_threshold=0.7,
    )
    server_model.train(X_calib, y_calib)

    payload = server_model.to_bytes()
    client_model = BoostingClassifier.from_bytes(
        payload, boosting_config, num_classes=len(label_encoder.classes_), benign_class=benign_class, seed=7,
        confidence_threshold=0.7,
    )

    X_sample = client_data[0]["X_test_raw"][:500]
    assert np.allclose(server_model.predict_proba(X_sample), client_model.predict_proba(X_sample), atol=1e-6)
