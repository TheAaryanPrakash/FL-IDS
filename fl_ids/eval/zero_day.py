"""Zero-day (leave-one-attack-class-out) experiment (component 12).

The autoencoder stage exists to catch attacks boosting's label taxonomy
doesn't cover. The standard per-stage evaluation can't show that: every
attack type in its test set is also in boosting's training labels, so the
autoencoder can only ever add false positives on top of boosting, and a
"boosting-only" baseline looks strictly better. This experiment measures
the thing the cascade is actually for.

For each held-out attack class H:

1. H is withheld from boosting's calibration set *and* from every
   client's federated data (`build_evaluation_setup_from_arrays`'s
   `excluded_classes`), so neither stage ever trains on it — H is a true
   zero-day for the whole system, not just for boosting. (If clients held
   H, boosting would pass it through as "normal" and the autoencoder would
   learn to reconstruct it — that's the attack being present in training
   traffic, a different scenario.)
2. FL training runs with no malicious clients, isolating detection
   ability from poisoning (the poisoning sweep covers that separately).
3. Every H row in the dataset is scored (capped at
   `evaluation.zero_day_max_holdout_rows`), not just the few that land in
   the shared test set — H is never trained on anywhere, so all of its
   rows are valid unseen traffic, and rare classes (MITM, Fingerprinting)
   would otherwise have single-digit test counts.

Per held-out class it reports detection rate for **boosting only** (the
cascade rule with no autoencoder backstop: a confident attack call, else
benign), the **autoencoder alone**, and the **full cascade**, plus what
that costs: benign false-positive rate and known-attack recall on the
shared test set, boosting-only vs. cascade.

Boosting can still "detect" an unseen attack by confidently calling it a
*different* known attack type. That counts as detection (for mitigation,
a wrong attack label still blocks the device), but the misattributed
label is reported too, since it's a wrong label.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from fl_ids.eval.common import (
    EvaluationSetup,
    assign_rows_to_clients,
    build_evaluation_setup_from_arrays,
    select_malicious_clients,
)
from fl_ids.eval.metrics import detection_metrics
from fl_ids.eval.variants import FULL_PIPELINE, TrainedVariant, Variant, flag_rows, train_variant
from fl_ids.models.autoencoder import reconstruction_error
from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)


def resolve_holdout_classes(configured: list[str], class_names: list[str], benign_class: int) -> list[int]:
    """Map configured held-out class names to indices; empty means every attack class.

    Args:
        configured: `evaluation.zero_day_holdout_classes`.
        class_names: Class names in class-index order.
        benign_class: Integer class index corresponding to "Normal".

    Returns:
        Held-out class indices, in the order given (or class-index order).

    Raises:
        ValueError: On an unknown class name, or the benign class.
    """
    if not configured:
        return [c for c in range(len(class_names)) if c != benign_class]
    indices = []
    for name in configured:
        if name not in class_names:
            raise ValueError(f"Unknown zero-day holdout class {name!r}; known classes: {class_names}")
        idx = class_names.index(name)
        if idx == benign_class:
            raise ValueError("The benign class can't be a zero-day holdout")
        indices.append(idx)
    return indices


def sample_holdout_rows(y: np.ndarray, holdout_class: int, max_rows: int, seed: int) -> np.ndarray:
    """Indices of the held-out class's rows to score, capped at `max_rows`.

    Seeded per class, so a class's rows don't depend on which other
    classes ran before it.
    """
    idx = np.where(y == holdout_class)[0]
    if len(idx) > max_rows:
        idx = np.sort(np.random.default_rng([seed, holdout_class]).choice(idx, max_rows, replace=False))
    return idx


def evaluate_zero_day_holdout(
    setup: EvaluationSetup,
    trained: TrainedVariant,
    X_holdout_raw: np.ndarray,
    config: Config,
    seed: int,
) -> dict:
    """Score one trained cascade stage by stage against a held-out (never-trained-on) attack class.

    Held-out rows belong to no client, so each is assigned to a random
    client and scored the way that client would score it: its scaler, its
    threshold (`assign_rows_to_clients`).

    Args:
        setup: The evaluation setup the models were trained from; its
            `excluded_classes` must contain exactly the held-out class.
        trained: The trained cascade (`train_variant(FULL_PIPELINE, ...)`).
        X_holdout_raw: Raw-scale rows of the held-out attack class.
        config: Full project config (cascade decision rule).
        seed: Random seed for assigning held-out rows to clients.

    Returns:
        A flat dict of held-out detection rates and shared-test-set costs
        (one row of the zero-day results table).
    """
    if len(setup.excluded_classes) != 1:
        raise ValueError("evaluate_zero_day_holdout expects a setup with exactly one excluded class")
    (holdout_class,) = setup.excluded_classes

    X_holdout_norm, holdout_clients = assign_rows_to_clients(setup, X_holdout_raw, seed)

    def _flags(X_raw, X_norm, clients, mode):
        return flag_rows(trained, setup, X_raw, X_norm, clients, config, mode=mode)

    holdout = {
        mode: _flags(X_holdout_raw, X_holdout_norm, holdout_clients, mode)
        for mode in ("boosting_only", "autoencoder_only", "cascade")
    }
    confident = trained.boosting_model.predict_cascade_stage1(X_holdout_raw)
    misattributed = pd.Series(
        np.array(setup.class_names, dtype=object)[confident.predicted_class[confident.is_confident_attack]]
    ).value_counts()

    test = {
        mode: detection_metrics(
            _flags(setup.X_test_raw, setup.X_test_norm, setup.test_client_ids, mode),
            setup.y_test, setup.benign_class, setup.class_names,
        )
        for mode in ("boosting_only", "cascade")
    }

    holdout_errors = reconstruction_error(trained.autoencoder, X_holdout_norm)
    benign_errors = reconstruction_error(trained.autoencoder, setup.X_test_norm[setup.y_test == setup.benign_class])
    scores = np.concatenate([benign_errors, holdout_errors])
    labels = np.concatenate([np.zeros(len(benign_errors)), np.ones(len(holdout_errors))])
    autoencoder_auroc = float(roc_auc_score(labels, scores)) if np.isfinite(scores).all() else float("nan")

    return {
        "holdout_class": setup.class_names[holdout_class],
        "holdout_rows": len(X_holdout_raw),
        "boosting_only_detection_rate": float(holdout["boosting_only"].mean()),
        "autoencoder_alone_detection_rate": float(holdout["autoencoder_only"].mean()),
        "cascade_detection_rate": float(holdout["cascade"].mean()),
        # The autoencoder only sees what boosting let through, so the
        # cascade's detections are exactly boosting's plus these.
        "autoencoder_added_detection_rate": float((holdout["cascade"] & ~holdout["boosting_only"]).mean()),
        "boosting_top_misattributed_label": str(misattributed.index[0]) if len(misattributed) else "",
        "boosting_top_misattributed_share": float(misattributed.iloc[0] / len(X_holdout_raw)) if len(misattributed) else 0.0,
        "autoencoder_auroc_vs_benign": autoencoder_auroc,
        "boosting_only_benign_fpr": test["boosting_only"]["benign_fpr"],
        "cascade_benign_fpr": test["cascade"]["benign_fpr"],
        "boosting_only_known_attack_macro_recall": test["boosting_only"]["attack_macro_recall"],
        "cascade_known_attack_macro_recall": test["cascade"]["attack_macro_recall"],
    }


def run_zero_day_experiment(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    benign_class: int,
    config: Config,
    num_rounds: int,
    seed: int,
    pool_subsample_size: int = 60_000,
) -> pd.DataFrame:
    """Run the leave-one-attack-class-out experiment over every configured holdout class.

    Trains the full pipeline with no malicious clients per holdout, and
    breaks detection down by cascade stage.

    Args:
        X: Full raw-scale feature matrix (`load_and_encode` output).
        y: Integer class labels.
        class_names: Class names in class-index order.
        benign_class: Integer class index corresponding to "Normal".
        config: Full project config (`evaluation.zero_day_*` picks the
            holdout classes and the per-class row cap).
        num_rounds: FL rounds per holdout run.
        seed: Random seed (splits, partitioning, model init, row sampling).
        pool_subsample_size: Caps the federated-client pool size.

    Returns:
        One row per held-out class (see `evaluate_zero_day_holdout`).
    """
    rows = []
    for holdout_class in resolve_holdout_classes(config.evaluation.zero_day_holdout_classes, class_names, benign_class):
        name = class_names[holdout_class]
        logger.info("Zero-day run: holding out %s", name)
        setup = build_evaluation_setup_from_arrays(
            X, y, class_names, benign_class, config, seed, pool_subsample_size,
            excluded_classes=frozenset({holdout_class}),
        )
        trained = train_variant(FULL_PIPELINE, setup, config, num_rounds, set(), seed)
        holdout_idx = sample_holdout_rows(y, holdout_class, config.evaluation.zero_day_max_holdout_rows, seed)
        row = evaluate_zero_day_holdout(setup, trained, X[holdout_idx], config, seed)
        row["final_val_loss"] = trained.simulation.rounds[-1].mean_val_loss
        rows.append(row)
        logger.info(
            "holdout=%s: boosting_only=%.3f autoencoder_alone=%.3f cascade=%.3f (benign FPR %.4f -> %.4f)",
            name, row["boosting_only_detection_rate"], row["autoencoder_alone_detection_rate"],
            row["cascade_detection_rate"], row["boosting_only_benign_fpr"], row["cascade_benign_fpr"],
        )
    return pd.DataFrame(rows)


def zero_day_detection_by_run(
    runs: list[tuple[str, Variant, float]],
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    benign_class: int,
    config: Config,
    num_rounds: int,
    seed: int,
    pool_subsample_size: int = 60_000,
) -> pd.DataFrame:
    """Held-out-attack detection rate for several pipeline variants, one row per (run, holdout class).

    The ablation and poisoning sweep use this for their zero-day column:
    for each holdout class, one setup is built and shared by every run, so
    runs differ only in the variant and poisoning fraction.

    Args:
        runs: `(label, variant, malicious_fraction)` per run.
        X: Full raw-scale feature matrix.
        y: Integer class labels.
        class_names: Class names in class-index order.
        benign_class: Integer class index corresponding to "Normal".
        config: Full project config (`evaluation.zero_day_*`).
        num_rounds: FL rounds per training.
        seed: Random seed.
        pool_subsample_size: Caps the federated-client pool size.

    Returns:
        Columns `run`, `holdout_class`, `detection_rate`.
    """
    rows = []
    for holdout_class in resolve_holdout_classes(config.evaluation.zero_day_holdout_classes, class_names, benign_class):
        name = class_names[holdout_class]
        setup = build_evaluation_setup_from_arrays(
            X, y, class_names, benign_class, config, seed, pool_subsample_size,
            excluded_classes=frozenset({holdout_class}),
        )
        holdout_idx = sample_holdout_rows(y, holdout_class, config.evaluation.zero_day_max_holdout_rows, seed)
        X_holdout_raw = X[holdout_idx]
        X_holdout_norm, holdout_clients = assign_rows_to_clients(setup, X_holdout_raw, seed)
        client_ids = list(setup.client_data)
        for label, variant, fraction in runs:
            malicious = select_malicious_clients(client_ids, fraction, seed)
            trained = train_variant(variant, setup, config, num_rounds, malicious, seed)
            rate = float(flag_rows(trained, setup, X_holdout_raw, X_holdout_norm, holdout_clients, config).mean())
            rows.append({"run": label, "holdout_class": name, "detection_rate": rate})
            logger.info("zero-day holdout=%s run=%s: detection_rate=%.3f", name, label, rate)
    return pd.DataFrame(rows)


# Reference data-viz palette (categorical slots 1 and 2, light surface) --
# a static paper figure, so light mode only.
_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"
_BOOSTING_COLOR = "#2a78d6"
_AUTOENCODER_COLOR = "#eb6834"


def plot_zero_day(results: pd.DataFrame, output_path: str | Path) -> None:
    """Save a stacked horizontal bar chart of unseen-attack detection per held-out class.

    Each bar is the full cascade's detection rate for that class, split
    into what boosting caught on its own (by misattributing it to a known
    attack type) and what the autoencoder backstop added on top.

    Args:
        results: `run_zero_day_experiment` output.
        output_path: Where to write the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = results.sort_values("cascade_detection_rate")
    y_pos = np.arange(len(df))
    boosting = df["boosting_only_detection_rate"].to_numpy() * 100
    added = df["autoencoder_added_detection_rate"].to_numpy() * 100

    fig, ax = plt.subplots(figsize=(9, 0.42 * len(df) + 1.8))
    fig.patch.set_facecolor(_SURFACE)
    ax.set_facecolor(_SURFACE)

    bar_style = {"height": 0.62, "edgecolor": _SURFACE, "linewidth": 2}
    ax.barh(y_pos, boosting, color=_BOOSTING_COLOR, label="Caught by boosting (as another known attack)", **bar_style)
    ax.barh(y_pos, added, left=boosting, color=_AUTOENCODER_COLOR, label="Added by autoencoder backstop", **bar_style)

    for yi, total in zip(y_pos, boosting + added):
        ax.text(total + 1.2, yi, f"{total:.0f}%", va="center", fontsize=9, color=_TEXT_PRIMARY)

    ax.set_yticks(y_pos, df["holdout_class"], fontsize=9, color=_TEXT_PRIMARY)
    ax.set_xlim(0, 110)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel("Held-out attack rows flagged (%)", color=_TEXT_SECONDARY)
    ax.tick_params(axis="x", colors=_TEXT_SECONDARY)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fpr_before = results["boosting_only_benign_fpr"].mean() * 100
    fpr_after = results["cascade_benign_fpr"].mean() * 100
    ax.set_title(
        "Zero-day detection: each attack type withheld from all training\n"
        f"Benign false-positive rate, mean over runs: boosting only {fpr_before:.2f}%, cascade {fpr_after:.2f}%",
        loc="left", fontsize=10, color=_TEXT_PRIMARY,
    )
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=_TEXT_SECONDARY)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    from fl_ids.data.pipeline import load_and_encode
    from fl_ids.utils.config import load_config
    from fl_ids.utils.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Run the zero-day (leave-one-attack-class-out) experiment")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--real-csv-path", default="data/raw/DNN-EdgeIIoT-dataset.csv")
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--holdout-classes", nargs="*", default=None,
        help="Override evaluation.zero_day_holdout_classes (class names)",
    )
    args = parser.parse_args()

    setup_logging()
    run_config = load_config(args.config_path)
    if args.holdout_classes is not None:
        run_config.evaluation.zero_day_holdout_classes = args.holdout_classes
    output_dir = Path(args.output_dir or run_config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_all, y_all, label_encoder, _feature_names, benign = load_and_encode(args.real_csv_path)
    results_df = run_zero_day_experiment(
        X_all, y_all, list(label_encoder.classes_), benign, run_config,
        num_rounds=args.num_rounds, seed=args.seed,
    )

    csv_path = output_dir / "zero_day_holdout.csv"
    results_df.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    plot_path = output_dir / "zero_day_holdout.png"
    plot_zero_day(results_df, plot_path)
    logger.info("Saved %s", plot_path)

    print(results_df.to_string(index=False))
