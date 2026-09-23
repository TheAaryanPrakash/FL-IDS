"""Evaluation harness (component 12) — per-stage metrics.

Computes precision/recall/F1/AUROC per class plus false-positive rate on
a held-out test set, for each cascade stage *separately* as well as
combined — boosting alone, autoencoder alone, the full cascade — so each
stage's actual contribution is visible, not just a final number.

AUROC is reported for the boosting stage (one-vs-rest, from its
predicted probabilities) and the autoencoder stage (binary, from
reconstruction error as the anomaly score) — both have a natural
continuous score to rank by. The full cascade's hard, two-stage decision
doesn't have one coherent continuous score to build a ROC curve from
(boosting's probability and the autoencoder's reconstruction error live
on different scales for different subsets of samples), so cascade-level
results report accuracy/precision/recall/F1/FPR only — a deliberate
scope decision, not an oversight.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from fl_ids.models.autoencoder import Autoencoder, reconstruction_error
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.models.cascade import BENIGN_LABEL, cascade_predict
from fl_ids.utils.config import CascadeConfig


@dataclass
class StageReport:
    """One stage's evaluation results, ready to flatten into a table row per class."""

    stage: str
    per_class: pd.DataFrame  # columns: class, precision, recall, f1, auroc, support
    accuracy: float
    macro_f1: float
    weighted_f1: float
    false_positive_rate: float
    extra: dict = field(default_factory=dict)


def _false_positive_rate(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> float:
    """Fraction of true-benign (0) samples predicted as attack (1)."""
    benign_mask = y_true_binary == 0
    if not benign_mask.any():
        return 0.0
    return float((y_pred_binary[benign_mask] == 1).mean())


def evaluate_boosting_alone(
    boosting_model: BoostingClassifier,
    X_raw: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    benign_class: int,
) -> StageReport:
    """Evaluate the boosting classifier on its own, ignoring the cascade rule.

    Args:
        boosting_model: Trained boosting classifier.
        X_raw: Raw-scale features.
        y_true: True integer class labels.
        class_names: Class names in class-index order.
        benign_class: Integer class index corresponding to "Normal".

    Returns:
        A `StageReport` with per-class precision/recall/F1/AUROC.
    """
    proba = boosting_model.predict_proba(X_raw)
    y_pred = np.argmax(proba, axis=1)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0, labels=range(len(class_names))
    )
    auroc = []
    for c in range(len(class_names)):
        y_true_binary = (y_true == c).astype(int)
        if y_true_binary.sum() == 0 or y_true_binary.sum() == len(y_true_binary):
            auroc.append(float("nan"))
        else:
            auroc.append(roc_auc_score(y_true_binary, proba[:, c]))

    per_class = pd.DataFrame(
        {
            "class": class_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
            "support": support,
        }
    )

    y_true_binary = (y_true != benign_class).astype(int)
    y_pred_binary = (y_pred != benign_class).astype(int)
    _, _, weighted_f1_arr, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    _, _, macro_f1_arr, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)

    return StageReport(
        stage="boosting_alone",
        per_class=per_class,
        accuracy=float((y_pred == y_true).mean()),
        macro_f1=float(macro_f1_arr),
        weighted_f1=float(weighted_f1_arr),
        false_positive_rate=_false_positive_rate(y_true_binary, y_pred_binary),
    )


def evaluate_autoencoder_alone(
    autoencoder: Autoencoder,
    anomaly_threshold: float,
    X_normalized: np.ndarray,
    y_true: np.ndarray,
    benign_class: int,
) -> StageReport:
    """Evaluate the autoencoder as a standalone binary anomaly detector.

    Applied to *every* test sample (not just boosting-filtered ones) —
    isolates what the backstop component can do entirely on its own.

    Args:
        autoencoder: Trained autoencoder.
        anomaly_threshold: Calibrated benign reconstruction-error threshold.
        X_normalized: Per-client-normalized features.
        y_true: True integer class labels.
        benign_class: Integer class index corresponding to "Normal".

    Returns:
        A `StageReport` with a single binary "row" (benign vs. attack).
    """
    errors = reconstruction_error(autoencoder, X_normalized)
    y_true_binary = (y_true != benign_class).astype(int)
    y_pred_binary = (errors > anomaly_threshold).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average=None, zero_division=0, labels=[0, 1]
    )
    if 0 < y_true_binary.sum() < len(y_true_binary) and np.isfinite(anomaly_threshold):
        auroc = [roc_auc_score(y_true_binary, errors)] * 2
    else:
        auroc = [float("nan")] * 2

    per_class = pd.DataFrame(
        {
            "class": ["benign", "attack"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": auroc,
            "support": support,
        }
    )

    _, _, weighted_f1_arr, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="weighted", zero_division=0
    )
    _, _, macro_f1_arr, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="macro", zero_division=0
    )

    return StageReport(
        stage="autoencoder_alone",
        per_class=per_class,
        accuracy=float((y_pred_binary == y_true_binary).mean()),
        macro_f1=float(macro_f1_arr),
        weighted_f1=float(weighted_f1_arr),
        false_positive_rate=_false_positive_rate(y_true_binary, y_pred_binary),
        extra={"anomaly_threshold": anomaly_threshold},
    )


def evaluate_cascade(
    boosting_model: BoostingClassifier,
    autoencoder: Autoencoder,
    anomaly_threshold: float,
    X_raw: np.ndarray,
    X_normalized: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    benign_class: int,
    cascade_config: CascadeConfig,
) -> StageReport:
    """Evaluate the full boosting -> autoencoder cascade combined.

    Reports at the binary (benign vs. attack) framing, since the cascade's
    output mixes specific attack-type names with the generic "anomalous"
    label — a per-specific-label breakdown is available via `extra`
    (`per_label`) for inspecting which stage made which call.

    Args:
        boosting_model: Trained boosting classifier.
        autoencoder: Trained autoencoder.
        anomaly_threshold: Calibrated benign reconstruction-error threshold.
        X_raw: Raw-scale features (boosting's input).
        X_normalized: Per-client-normalized features (autoencoder's input),
            same rows as `X_raw`.
        y_true: True integer class labels.
        class_names: Class names in class-index order.
        benign_class: Integer class index corresponding to "Normal".
        cascade_config: Cascade decision rule config.

    Returns:
        A `StageReport`; `extra["per_label"]` holds the full per-predicted-
        label breakdown (including which stage made each call), and
        `extra["stage_counts"]` holds how many samples each stage decided.
    """
    output = cascade_predict(
        boosting_model, autoencoder, anomaly_threshold, X_raw, X_normalized, class_names, cascade_config
    )

    y_true_binary = (y_true != benign_class).astype(int)
    y_pred_binary = (output.predicted_label != BENIGN_LABEL).astype(int)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average=None, zero_division=0, labels=[0, 1]
    )
    per_class = pd.DataFrame(
        {
            "class": ["benign", "attack"],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "auroc": [float("nan")] * 2,
            "support": support,
        }
    )

    _, _, weighted_f1_arr, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="weighted", zero_division=0
    )
    _, _, macro_f1_arr, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="macro", zero_division=0
    )

    true_label_names = np.array(class_names, dtype=object)[y_true]
    per_label_df = pd.DataFrame(
        {"true_label": true_label_names, "predicted_label": output.predicted_label, "stage": output.stage}
    )
    stage_counts = per_label_df["stage"].value_counts().to_dict()

    return StageReport(
        stage="cascade_combined",
        per_class=per_class,
        accuracy=float((y_pred_binary == y_true_binary).mean()),
        macro_f1=float(macro_f1_arr),
        weighted_f1=float(weighted_f1_arr),
        false_positive_rate=_false_positive_rate(y_true_binary, y_pred_binary),
        extra={"per_label": per_label_df, "stage_counts": stage_counts},
    )


def stage_report_summary(report: StageReport) -> dict:
    """A JSON-serializable summary of a `StageReport`: headline metrics plus per-class F1/recall.

    Used where a report has to travel through a file rather than stay in
    memory -- e.g. the boosting model's held-out metrics that the Flower
    server stamps into its live per-round state for the dashboard.
    """
    return {
        "accuracy": report.accuracy,
        "macro_f1": report.macro_f1,
        "weighted_f1": report.weighted_f1,
        "false_positive_rate": report.false_positive_rate,
        "per_class": {
            row["class"]: {"f1": float(row["f1"]), "recall": float(row["recall"]), "support": int(row["support"])}
            for _, row in report.per_class.iterrows()
        },
    }


def build_per_stage_table(reports: list[StageReport]) -> pd.DataFrame:
    """Flatten several `StageReport`s into one CSV-ready comparison table.

    Args:
        reports: E.g. `[evaluate_boosting_alone(...), evaluate_autoencoder_alone(...),
            evaluate_cascade(...)]`.

    Returns:
        One row per (stage, class), with accuracy/macro_f1/weighted_f1/FPR
        repeated per stage for easy at-a-glance comparison.
    """
    rows = []
    for report in reports:
        for _, row in report.per_class.iterrows():
            rows.append(
                {
                    "stage": report.stage,
                    "class": row["class"],
                    "precision": row["precision"],
                    "recall": row["recall"],
                    "f1": row["f1"],
                    "auroc": row["auroc"],
                    "support": row["support"],
                    "stage_accuracy": report.accuracy,
                    "stage_macro_f1": report.macro_f1,
                    "stage_weighted_f1": report.weighted_f1,
                    "stage_false_positive_rate": report.false_positive_rate,
                }
            )
    return pd.DataFrame(rows)
