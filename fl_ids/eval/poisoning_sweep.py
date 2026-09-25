"""Poisoning-resistance sweep (component 12).

Reruns FL training at several malicious-client fractions (0/20/30/40/50%
by default — `config.evaluation.poisoning_fractions`) and scores the full
pipeline at each point the same way the ablation does
(`fl_ids.eval.variants`).

Headline numbers are attack macro-recall, zero-day macro-recall and
benign FPR, not accuracy: boosting isn't FL-trained, so poisoning can't
touch it, and a poisoned autoencoder that stops flagging anything lowers
the benign FPR — which on ~71%-benign traffic makes accuracy go *up* as
the backstop breaks. Zero-day detection (held-out attack classes) is
where a broken backstop shows, since that's the traffic only it catches.
The trust filter's own breakdown point shows directly in
`malicious_client_survival_rate`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fl_ids.eval.common import EvaluationSetup, select_malicious_clients
from fl_ids.eval.variants import FULL_PIPELINE, malicious_survival_rate, score_on_test_set, train_variant
from fl_ids.eval.checkpoint import RowCheckpoint, run_fingerprint
from fl_ids.eval.zero_day import zero_day_detection_by_run
from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)

SWEEP_COLUMNS = [
    "malicious_fraction",
    "num_malicious_clients",
    "num_total_clients",
    "attack_macro_recall",
    "attack_micro_recall",
    "benign_fpr",
    "detection_f1",
    "zero_day_macro_recall",
    "autoencoder_alone_attack_macro_recall",
    "malicious_client_survival_rate",
    "final_val_loss",
    "rounds_to_convergence",
    "total_communication_bytes",
]


def _run_label(fraction: float) -> str:
    return f"malicious_fraction={fraction:.2f}"


def run_poisoning_sweep(
    setup: EvaluationSetup,
    config: Config,
    num_rounds: int,
    seed: int,
) -> pd.DataFrame:
    """Train and score the full pipeline at every configured poisoning fraction.

    Args:
        setup: Shared evaluation setup (see `fl_ids.eval.common.build_evaluation_setup`).
        config: Full project config (`evaluation.poisoning_fractions` drives the sweep points).
        num_rounds: FL rounds per sweep point.
        seed: Random seed (also used for malicious-client selection).

    Returns:
        One row per poisoning fraction with `SWEEP_COLUMNS`
        (`zero_day_macro_recall` left NaN — see `add_zero_day_column`).
    """
    client_ids = list(setup.client_data)
    rows = []

    for fraction in config.evaluation.poisoning_fractions:
        malicious_ids = select_malicious_clients(client_ids, fraction, seed)
        logger.info("Poisoning sweep: fraction=%.2f -> malicious clients=%s", fraction, sorted(malicious_ids))

        trained = train_variant(FULL_PIPELINE, setup, config, num_rounds, malicious_ids, seed)
        scores = score_on_test_set(trained, setup, config)
        scores.pop("per_class_recall")
        rows.append(
            {
                "malicious_fraction": fraction,
                "num_malicious_clients": len(malicious_ids),
                "num_total_clients": len(client_ids),
                **scores,
                "zero_day_macro_recall": float("nan"),
                "malicious_client_survival_rate": malicious_survival_rate(trained, malicious_ids),
            }
        )
        logger.info(
            "fraction=%.2f: attack_macro_recall=%.3f benign_fpr=%.4f autoencoder_alone=%.3f survival=%.3f",
            fraction, scores["attack_macro_recall"], scores["benign_fpr"],
            scores["autoencoder_alone_attack_macro_recall"], rows[-1]["malicious_client_survival_rate"],
        )

    return pd.DataFrame(rows)[SWEEP_COLUMNS]


def add_zero_day_column(
    sweep: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    benign_class: int,
    config: Config,
    num_rounds: int,
    seed: int,
    checkpoint: RowCheckpoint | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill `zero_day_macro_recall` by retraining at every fraction once per held-out attack class.

    Args:
        sweep: `run_poisoning_sweep` output.
        X, y, class_names, benign_class: The full dataset (`load_and_encode` output).
        config: Full project config (`evaluation.zero_day_*`, poisoning fractions).
        num_rounds: FL rounds per training.
        seed: Same seed as the sweep.
        checkpoint: Passed to `zero_day_detection_by_run`.

    Returns:
        (sweep with the column filled, per-(fraction, holdout class) detection rates).
    """
    fractions = list(sweep["malicious_fraction"])
    runs = [(_run_label(f), FULL_PIPELINE, f) for f in fractions]
    detail = zero_day_detection_by_run(
        runs, X, y, class_names, benign_class, config, num_rounds, seed, checkpoint=checkpoint
    )
    means = detail.groupby("run")["detection_rate"].mean()
    sweep = sweep.copy()
    sweep["zero_day_macro_recall"] = [means[_run_label(f)] for f in fractions]
    return sweep, detail


# Reference data-viz palette, light surface (a static paper figure).
_SURFACE = "#fcfcfb"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"
_SERIES = {
    "attack_macro_recall": ("Known attacks (macro recall)", "#2a78d6", "o"),
    "zero_day_macro_recall": ("Held-out attacks (zero-day macro recall)", "#eb6834", "s"),
    "autoencoder_alone_attack_macro_recall": ("Autoencoder alone (macro recall)", "#1baf7a", "^"),
    "benign_fpr": ("Benign false-positive rate", "#eda100", "D"),
}
_SURVIVAL_COLOR = "#4a3aa7"


def plot_poisoning_sweep(sweep_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a two-panel plot: detection vs. poisoning fraction, and the trust filter's breakdown.

    Left: the full pipeline's detection metrics on one 0-1 axis (series
    that are all NaN — e.g. zero-day when skipped — are left out). Right:
    the mean per-round fraction of malicious clients that survived the
    trust filter.

    Args:
        sweep_df: `run_poisoning_sweep` output (optionally with the zero-day column filled).
        output_path: Where to write the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = sweep_df["malicious_fraction"] * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [3, 2]})
    fig.patch.set_facecolor(_SURFACE)

    for column, (label, color, marker) in _SERIES.items():
        if sweep_df[column].notna().any():
            ax1.plot(x, sweep_df[column], color=color, marker=marker, markersize=7, linewidth=2, label=label)
    ax1.set_title("Full pipeline detection", loc="left", fontsize=10, color=_TEXT_PRIMARY)
    ax1.set_ylabel("Rate", color=_TEXT_SECONDARY)
    ax1.legend(frameon=False, fontsize=8.5, labelcolor=_TEXT_SECONDARY, loc="center left")

    ax2.plot(x, sweep_df["malicious_client_survival_rate"], color=_SURVIVAL_COLOR, marker="o", markersize=7, linewidth=2)
    ax2.set_title("Malicious clients surviving the trust filter", loc="left", fontsize=10, color=_TEXT_PRIMARY)
    ax2.set_ylabel("Mean share per round", color=_TEXT_SECONDARY)

    for ax in (ax1, ax2):
        ax.set_facecolor(_SURFACE)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(x)
        ax.set_xlabel("Malicious client fraction (%)", color=_TEXT_SECONDARY)
        ax.tick_params(colors=_TEXT_SECONDARY)
        ax.grid(axis="y", color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=_SURFACE)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    from fl_ids.data.pipeline import load_and_encode
    from fl_ids.eval.common import build_evaluation_setup_from_arrays
    from fl_ids.utils.config import load_config
    from fl_ids.utils.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Run the poisoning-resistance sweep (component 12)")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--real-csv-path", default="data/raw/DNN-EdgeIIoT-dataset.csv")
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--skip-zero-day", action="store_true",
        help="Leave zero_day_macro_recall empty (skips one retraining per fraction per held-out class)",
    )
    args = parser.parse_args()

    setup_logging()
    run_config = load_config(args.config_path)
    output_dir = Path(args.output_dir or run_config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_all, y_all, label_encoder, _feature_names, benign = load_and_encode(args.real_csv_path, run_config.data.capture_repairs)
    names = list(label_encoder.classes_)
    eval_setup = build_evaluation_setup_from_arrays(X_all, y_all, names, benign, run_config, args.seed)
    sweep_df = run_poisoning_sweep(eval_setup, run_config, num_rounds=args.num_rounds, seed=args.seed)

    checkpoint = None
    if not args.skip_zero_day:
        checkpoint = RowCheckpoint(
            output_dir / "poisoning_zero_day.checkpoint.csv",
            run_fingerprint(run_config, args.num_rounds, args.seed), ("run", "holdout_class"),
        )
        sweep_df, zero_day_df = add_zero_day_column(
            sweep_df, X_all, y_all, names, benign, run_config, num_rounds=args.num_rounds, seed=args.seed,
            checkpoint=checkpoint,
        )
        zero_day_df.to_csv(output_dir / "poisoning_zero_day_detail.csv", index=False)

    csv_path = output_dir / "poisoning_resistance_sweep.csv"
    sweep_df.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    plot_path = output_dir / "poisoning_resistance_sweep.png"
    plot_poisoning_sweep(sweep_df, plot_path)
    logger.info("Saved %s", plot_path)
    if checkpoint:
        checkpoint.remove()

    print(sweep_df.to_string(index=False))
