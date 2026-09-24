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

Uses the production hyperparameters from configs/config.yaml, not a
test-local copy, so these tests check the model the pipeline actually
ships. `test_boosting_is_stable_under_small_calibration_changes` guards
against the training divergence found while building the zero-day
experiment (see fl_ids.models.boosting's train()): before leaf
regularization, removing a random 1% of calibration rows moved benign
false-positive rate between 0.1% and 16%.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from fl_ids.data.pipeline import build_server_and_federated_dataset, load_and_encode
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import DataConfig
from tests.real_data_boosting import PRODUCTION_CONFIG

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
    boosting_config = PRODUCTION_CONFIG.boosting

    X_calib, y_calib, client_data, label_encoder, feature_names = build_server_and_federated_dataset(
        REAL_DATASET_PATH, data_config, calibration_fraction=boosting_config.calibration_fraction, seed=42
    )
    benign_class = int(label_encoder.transform(["Normal"])[0])

    boosting_model = BoostingClassifier(
        boosting_config, num_classes=len(label_encoder.classes_), benign_class=benign_class, seed=42,
        confidence_threshold=PRODUCTION_CONFIG.cascade.confidence_threshold,
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

    # The same "reasonably decent" bar Phase 2 set on synthetic data, raised
    # to match what the production config reliably achieves on real data.
    assert weighted_f1 > 0.95, f"cold-start bootstrap model should be reasonably decent on real data, got weighted F1={weighted_f1:.3f}"
    assert accuracy > 0.95, f"cold-start bootstrap model should be reasonably decent on real data, got accuracy={accuracy:.3f}"

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
    boosting_config = PRODUCTION_CONFIG.boosting

    X_calib, y_calib, client_data, label_encoder, _ = build_server_and_federated_dataset(
        REAL_DATASET_PATH, data_config, calibration_fraction=boosting_config.calibration_fraction, seed=7
    )
    benign_class = int(label_encoder.transform(["Normal"])[0])

    server_model = BoostingClassifier(
        boosting_config, num_classes=len(label_encoder.classes_), benign_class=benign_class, seed=7,
        confidence_threshold=PRODUCTION_CONFIG.cascade.confidence_threshold,
    )
    server_model.train(X_calib, y_calib)

    payload = server_model.to_bytes()
    client_model = BoostingClassifier.from_bytes(
        payload, boosting_config, num_classes=len(label_encoder.classes_), benign_class=benign_class, seed=7,
        confidence_threshold=PRODUCTION_CONFIG.cascade.confidence_threshold,
    )

    X_sample = client_data[0]["X_test_raw"][:500]
    assert np.allclose(server_model.predict_proba(X_sample), client_model.predict_proba(X_sample), atol=1e-6)


@pytest.mark.skipif(not REAL_DATASET_PATH.exists(), reason="real dataset not present")
def test_boosting_is_stable_under_small_calibration_changes():
    """Removing a random 1% of calibration rows must not change what the model does.

    Before leaf regularization, training diverged: held-out logloss peaked
    around iteration 16 and then blew up, so the final model depended on
    exactly which rows it saw. Dropping 1% of rows moved benign
    false-positive rate between 0.1% and 16% and accuracy between 0.58 and
    0.98. Every evaluation number downstream inherits that, so this checks
    the spread directly rather than one lucky configuration.
    """
    X, y, label_encoder, _, benign_class = load_and_encode(REAL_DATASET_PATH)
    config = PRODUCTION_CONFIG
    X_calib, X_rest, y_calib, y_rest = train_test_split(
        X, y, train_size=config.boosting.calibration_fraction, random_state=42, stratify=y
    )
    X_eval, _, y_eval, _ = train_test_split(X_rest, y_rest, train_size=20_000, random_state=0, stratify=y_rest)
    is_benign = y_eval == benign_class

    accuracies, benign_fprs = [], []
    for removal_seed in (None, 1, 2):
        keep = np.ones(len(y_calib), dtype=bool)
        if removal_seed is not None:
            drop = np.random.default_rng(removal_seed).choice(len(y_calib), len(y_calib) // 100, replace=False)
            keep[drop] = False
        model = BoostingClassifier(
            config.boosting, num_classes=len(label_encoder.classes_), benign_class=benign_class, seed=42,
            confidence_threshold=config.cascade.confidence_threshold,
        )
        model.train(X_calib[keep], y_calib[keep])
        accuracies.append(float((np.argmax(model.predict_proba(X_eval), axis=1) == y_eval).mean()))
        benign_fprs.append(float(model.predict_cascade_stage1(X_eval[is_benign]).is_confident_attack.mean()))

    print(f"accuracies={accuracies} benign_fprs={benign_fprs}")
    assert max(accuracies) - min(accuracies) < 0.01, f"accuracy unstable across 1% row removals: {accuracies}"
    assert max(benign_fprs) < 0.01, f"benign false-positive rate too high or unstable: {benign_fprs}"
