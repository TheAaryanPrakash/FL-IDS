"""Poisoning-resistance sweep (component 12).

Reruns FL training at several malicious-client fractions (0/20/30/40/50%
by default — `config.evaluation.poisoning_fractions`), evaluating the
resulting cascade's accuracy at each point, so the trust filter's real
contribution is visible as a curve, not asserted in the abstract.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fl_ids.eval.common import EvaluationSetup, calibrate_threshold_from_final_weights, select_malicious_clients
from fl_ids.eval.metrics import evaluate_autoencoder_alone, evaluate_cascade
from fl_ids.eval.simulation import run_simulated_fl_training
from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)


def run_poisoning_sweep(
    setup: EvaluationSetup,
    config: Config,
    num_rounds: int,
    seed: int,
) -> pd.DataFrame:
    """Run the full poisoning-resistance sweep and return one row per fraction.

    Args:
        setup: Shared evaluation setup (see `fl_ids.eval.common.build_evaluation_setup`).
        config: Full project config (`evaluation.poisoning_fractions` drives the sweep points).
        num_rounds: FL rounds per sweep point.
        seed: Random seed (also used for malicious-client selection).

    Returns:
        A DataFrame with one row per poisoning fraction: accuracy,
        weighted_f1, false_positive_rate, rounds_to_convergence,
        total_communication_bytes.
    """
    client_ids = list(setup.client_data.keys())
    rows = []

    for fraction in config.evaluation.poisoning_fractions:
        malicious_ids = select_malicious_clients(client_ids, fraction, seed)
        logger.info("Poisoning sweep: fraction=%.2f -> malicious clients=%s", fraction, sorted(malicious_ids))

        result = run_simulated_fl_training(
            setup.client_data,
            setup.boosting_model,
            len(setup.class_names),
            setup.benign_class,
            config,
            num_rounds=num_rounds,
            aggregation="trust_filtered",
            malicious_client_ids=malicious_ids,
            seed=seed,
        )

        autoencoder, threshold = calibrate_threshold_from_final_weights(result.final_weights, setup, config)
        report = evaluate_cascade(
            setup.boosting_model,
            autoencoder,
            threshold,
            setup.X_test_raw,
            setup.X_test_norm,
            setup.y_test,
            setup.class_names,
            setup.benign_class,
            config.cascade,
        )

        # Cascade accuracy alone can be deceptively stable under heavy
        # poisoning: boosting (never FL-trained, so unaffected by client
        # poisoning) still confidently handles a large share of samples,
        # masking a collapsed autoencoder backstop behind a fine-looking
        # top-line number. The autoencoder's own standalone attack recall,
        # and the trust filter's actual survival rate, surface that
        # collapse directly rather than leaving it implicit in a loss
        # number the reader has to know to go looking for.
        ae_report = evaluate_autoencoder_alone(
            autoencoder, threshold, setup.X_test_norm, setup.y_test, setup.benign_class
        )
        ae_attack_recall = float(ae_report.per_class.loc[ae_report.per_class["class"] == "attack", "recall"].iloc[0])

        malicious_survival_rate = float("nan")
        if malicious_ids:
            per_round_malicious_survival = [
                len(set(r.survivors) & malicious_ids) / len(malicious_ids) for r in result.rounds if r.survivors
            ]
            if per_round_malicious_survival:
                malicious_survival_rate = float(np.mean(per_round_malicious_survival))

        rows.append(
            {
                "malicious_fraction": fraction,
                "num_malicious_clients": len(malicious_ids),
                "num_total_clients": len(client_ids),
                "accuracy": report.accuracy,
                "weighted_f1": report.weighted_f1,
                "false_positive_rate": report.false_positive_rate,
                "rounds_to_convergence": result.rounds_to_convergence,
                "total_communication_bytes": result.total_communication_bytes,
                "final_val_loss": result.rounds[-1].mean_val_loss,
                "autoencoder_alone_attack_recall": ae_attack_recall,
                "malicious_client_survival_rate": malicious_survival_rate,
            }
        )
        logger.info(
            "fraction=%.2f: accuracy=%.4f weighted_f1=%.4f fpr=%.4f",
            fraction, report.accuracy, report.weighted_f1, report.false_positive_rate,
        )

    return pd.DataFrame(rows)


def plot_poisoning_sweep(sweep_df: pd.DataFrame, output_path: str | Path) -> None:
    """Save a two-panel plot: cascade accuracy/F1, and the autoencoder's own
    standalone attack recall + final validation loss (log scale).

    Cascade accuracy alone can look deceptively stable under heavy
    poisoning — boosting still confidently handles a large share of
    samples regardless of how poisoned the FL-trained autoencoder is,
    masking a collapsed backstop behind a fine-looking top-line number.
    The second panel is what actually shows the trust filter's
    breakdown point (autoencoder attack recall collapsing, validation
    loss exploding) rather than leaving it implicit.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    x = sweep_df["malicious_fraction"] * 100
    ax1.plot(x, sweep_df["accuracy"], marker="o", label="Cascade accuracy")
    ax1.plot(x, sweep_df["weighted_f1"], marker="s", label="Cascade weighted F1")
    ax1.set_xlabel("Malicious client fraction (%)")
    ax1.set_ylabel("Score")
    ax1.set_title("Cascade-level accuracy retained")
    ax1.set_ylim(0.0, 1.05)
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2b = ax2.twinx()
    ax2.plot(x, sweep_df["autoencoder_alone_attack_recall"], marker="o", color="tab:red", label="Autoencoder attack recall")
    ax2.set_ylabel("Autoencoder-alone attack recall", color="tab:red")
    ax2.set_ylim(0.0, 1.05)
    ax2b.plot(x, sweep_df["final_val_loss"], marker="s", color="tab:gray", label="Final val loss (log)")
    ax2b.set_yscale("log")
    ax2b.set_ylabel("Final validation loss (log scale)", color="tab:gray")
    ax2.set_xlabel("Malicious client fraction (%)")
    ax2.set_title("Autoencoder backstop's actual health")
    ax2.grid(alpha=0.3)

    fig.suptitle("Poisoning-Resistance Sweep (trust-filtered aggregation)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import argparse

    from fl_ids.eval.common import build_evaluation_setup
    from fl_ids.utils.config import load_config
    from fl_ids.utils.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Run the poisoning-resistance sweep (component 12)")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--real-csv-path", default="data/raw/DNN-EdgeIIoT-dataset.csv")
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    setup_logging()
    run_config = load_config(args.config_path)
    output_dir = Path(args.output_dir or run_config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_setup = build_evaluation_setup(run_config, args.real_csv_path, seed=args.seed)
    sweep_df = run_poisoning_sweep(eval_setup, run_config, num_rounds=args.num_rounds, seed=args.seed)

    csv_path = output_dir / "poisoning_resistance_sweep.csv"
    sweep_df.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    plot_path = output_dir / "poisoning_resistance_sweep.png"
    plot_poisoning_sweep(sweep_df, plot_path)
    logger.info("Saved %s", plot_path)

    print(sweep_df.to_string(index=False))
