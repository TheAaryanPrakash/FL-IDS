"""The full cascade decision rule (CLAUDE.md's "Cascade decision rule"),
combining boosting (component 2) and the autoencoder (component 3) into
one final per-sample classification.

This is the single implementation of the combined rule — component 12's
evaluation, component 13's dashboard, and the SDN bridge (component 9)
all call this rather than re-deriving it independently:

- Boosting produces a predicted class + that class's probability. If the
  predicted class is a known attack type **and** that probability is >=
  `CascadeConfig.confidence_threshold`, the final output is that attack
  type, with `confidence` = boosting's probability. The autoencoder is
  not consulted.
- Otherwise — boosting predicts "normal", or predicts an attack type
  without enough confidence — the sample passes to the autoencoder. If
  its reconstruction error exceeds the calibrated benign threshold, the
  final output is `"anomalous"` (a zero-day signal, deliberately not a
  specific attack type), with `confidence` = `min(1.0, error / threshold
  - 1.0)` clamped to `CascadeConfig.anomaly_confidence_clip`. If
  reconstruction error is within the threshold, the final output is
  `"benign"`.

`predict_cascade_stage1`/`passes_to_autoencoder` (component 2) already
implement the boosting half in isolation, for the client-side traffic
filter (component 3/4) that doesn't need the autoencoder half at all.
This module builds the *complete* rule on top of that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from fl_ids.models.autoencoder import Autoencoder, reconstruction_error
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import CascadeConfig

ANOMALOUS_LABEL = "anomalous"
BENIGN_LABEL = "benign"


@dataclass
class CascadeOutput:
    """Per-sample final cascade output.

    Attributes:
        predicted_label: A known attack-type class name (from boosting),
            or `"anomalous"`, or `"benign"`.
        confidence: In `[0, 1]`.
        stage: `"boosting"` or `"autoencoder"` — which stage made the call.
    """

    predicted_label: np.ndarray  # dtype=object, array of str
    confidence: np.ndarray
    stage: np.ndarray  # dtype=object, array of str


def cascade_predict(
    boosting_model: BoostingClassifier,
    autoencoder: Autoencoder,
    anomaly_threshold: float | np.ndarray,
    X_raw: np.ndarray,
    X_normalized: np.ndarray,
    class_names: list[str],
    cascade_config: CascadeConfig,
) -> CascadeOutput:
    """Apply the full cascade decision rule to a batch of samples.

    Args:
        boosting_model: Trained boosting classifier (cascade stage 1).
        autoencoder: Trained autoencoder (cascade stage 2).
        anomaly_threshold: The calibrated benign reconstruction-error
            threshold (see `fl_ids.models.autoencoder.compute_anomaly_threshold`),
            either one value for every sample or one per sample -- e.g. when
            each row comes from a different client, scored against that
            client's own threshold.
        X_raw: Raw-scale features (what boosting consumes), shape
            (n_samples, n_features).
        X_normalized: Per-client-normalized features (what the
            autoencoder consumes) for the *same* rows, same shape as `X_raw`.
        class_names: Attack-type class names in boosting's class-index order.
        cascade_config: `confidence_threshold`, `anomaly_confidence_clip`.

    Returns:
        A `CascadeOutput` with one entry per input sample.
    """
    stage1 = boosting_model.predict_cascade_stage1(X_raw)
    n = len(X_raw)

    predicted_label = np.empty(n, dtype=object)
    confidence = np.zeros(n, dtype=np.float64)
    stage = np.empty(n, dtype=object)

    confident_mask = stage1.is_confident_attack
    for idx in np.where(confident_mask)[0]:
        predicted_label[idx] = class_names[stage1.predicted_class[idx]]
    confidence[confident_mask] = stage1.confidence[confident_mask]
    stage[confident_mask] = "boosting"

    fallthrough_mask = ~confident_mask
    if fallthrough_mask.any():
        errors = reconstruction_error(autoencoder, X_normalized[fallthrough_mask])
        thresholds = np.broadcast_to(np.asarray(anomaly_threshold, dtype=np.float64), (n,))[fallthrough_mask]
        is_anomalous = errors > thresholds

        clip_low, clip_high = cascade_config.anomaly_confidence_clip
        ratio = np.divide(errors, thresholds, out=np.full(errors.shape, np.inf), where=thresholds > 0)
        anomaly_confidence = np.clip(ratio - 1.0, clip_low, clip_high)
        benign_confidence = 1.0 - np.clip(np.where(thresholds > 0, ratio, 0.0), 0.0, 1.0)

        fallthrough_indices = np.where(fallthrough_mask)[0]
        for local_i, global_i in enumerate(fallthrough_indices):
            if is_anomalous[local_i]:
                predicted_label[global_i] = ANOMALOUS_LABEL
                confidence[global_i] = anomaly_confidence[local_i]
            else:
                predicted_label[global_i] = BENIGN_LABEL
                confidence[global_i] = benign_confidence[local_i]
            stage[global_i] = "autoencoder"

    return CascadeOutput(predicted_label, confidence, stage)
