"""Ablation table (component 12).

Full pipeline vs. plain FedAvg vs. trimmed-mean-only (no cosine filter)
vs. autoencoder-only (no boosting pre-filter) vs. boosting-only (no
autoencoder backstop) — same data, same seed, one comparison table, at a
fixed moderate poisoning fraction (so the robustness variants actually
have something to defend against).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from fl_ids.eval.common import EvaluationSetup, calibrate_threshold_from_final_weights, select_malicious_clients
from fl_ids.eval.metrics import evaluate_autoencoder_alone, evaluate_cascade
from fl_ids.eval.simulation import run_simulated_fl_training
from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)


def _binary_metrics(y_true_binary: np.ndarray, y_pred_binary: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_binary, y_pred_binary, average="weighted", zero_division=0
    )
    accuracy = float((y_true_binary == y_pred_binary).mean())
    benign_mask = y_true_binary == 0
    fpr = float((y_pred_binary[benign_mask] == 1).mean()) if benign_mask.any() else 0.0
    return {"accuracy": accuracy, "weighted_f1": float(f1), "false_positive_rate": fpr}


def run_ablation(
    setup: EvaluationSetup,
    config: Config,
    num_rounds: int,
    poisoning_fraction: float,
    seed: int,
) -> pd.DataFrame:
    """Run all five ablation variants and return the comparison table.

    Args:
        setup: Shared evaluation setup.
        config: Full project config.
        num_rounds: FL rounds per variant (variants 4 use FL training too;
            variant 5 uses no FL training at all).
        poisoning_fraction: Fraction of clients malicious, held fixed
            across variants 1-4 so the robustness comparison is meaningful.
        seed: Random seed (data/model init and malicious-client selection).

    Returns:
        One row per variant: accuracy, weighted_f1, false_positive_rate,
        rounds_to_convergence, total_communication_bytes (NaN for the
        no-FL-training boosting-only variant).
    """
    client_ids = list(setup.client_data.keys())
    malicious_ids = select_malicious_clients(client_ids, poisoning_fraction, seed)
    logger.info("Ablation: poisoning_fraction=%.2f -> malicious clients=%s", poisoning_fraction, sorted(malicious_ids))

    rows = []

    def _run_variant(name: str, aggregation: str, use_boosting_filter: bool, eval_mode: str) -> None:
        result = run_simulated_fl_training(
            setup.client_data,
            setup.boosting_model,
            len(setup.class_names),
            setup.benign_class,
            config,
            num_rounds=num_rounds,
            aggregation=aggregation,
            malicious_client_ids=malicious_ids,
            use_boosting_filter=use_boosting_filter,
            seed=seed,
        )
        autoencoder, threshold = calibrate_threshold_from_final_weights(result.final_weights, setup, config)

        if eval_mode == "cascade":
            report = evaluate_cascade(
                setup.boosting_model, autoencoder, threshold, setup.X_test_raw, setup.X_test_norm,
                setup.y_test, setup.class_names, setup.benign_class, config.cascade,
            )
            metrics = {
                "accuracy": report.accuracy,
                "weighted_f1": report.weighted_f1,
                "false_positive_rate": report.false_positive_rate,
            }
        else:  # "autoencoder_alone"
            report = evaluate_autoencoder_alone(
                autoencoder, threshold, setup.X_test_norm, setup.y_test, setup.benign_class
            )
            metrics = {
                "accuracy": report.accuracy,
                "weighted_f1": report.weighted_f1,
                "false_positive_rate": report.false_positive_rate,
            }

        # Cascade-level accuracy can be deceptively similar across variants:
        # boosting (never FL-trained, so identical across every variant)
        # confidently handles a large, fixed share of samples regardless of
        # how good or bad the *autoencoder* each variant actually produced
        # is, diluting real differences between the variants this table
        # exists to compare. Always also report the trained autoencoder's
        # own standalone attack recall and final validation loss, isolated
        # from boosting's stabilizing effect, so the real story shows.
        ae_report = evaluate_autoencoder_alone(
            autoencoder, threshold, setup.X_test_norm, setup.y_test, setup.benign_class
        )
        ae_attack_recall = float(ae_report.per_class.loc[ae_report.per_class["class"] == "attack", "recall"].iloc[0])

        rows.append(
            {
                "variant": name,
                **metrics,
                "autoencoder_alone_attack_recall": ae_attack_recall,
                "final_val_loss": result.rounds[-1].mean_val_loss,
                "rounds_to_convergence": result.rounds_to_convergence,
                "total_communication_bytes": result.total_communication_bytes,
            }
        )
        logger.info("variant=%s: %s (autoencoder_attack_recall=%.3f)", name, metrics, ae_attack_recall)

    _run_variant("full_pipeline", "trust_filtered", use_boosting_filter=True, eval_mode="cascade")
    _run_variant("plain_fedavg", "fedavg", use_boosting_filter=True, eval_mode="cascade")
    _run_variant("trimmed_mean_only", "trimmed_mean_only", use_boosting_filter=True, eval_mode="cascade")
    _run_variant("autoencoder_only", "trust_filtered", use_boosting_filter=False, eval_mode="autoencoder_alone")

    # boosting_only: no FL/autoencoder at all -- the cascade degrades to
    # boosting's confident-attack call, with every fallthrough defaulting
    # to "benign" (no anomaly backstop).
    y_pred = np.argmax(setup.boosting_model.predict_proba(setup.X_test_raw), axis=1)
    y_pred_binary = (y_pred != setup.benign_class).astype(int)
    y_true_binary = (setup.y_test != setup.benign_class).astype(int)
    boosting_only_metrics = _binary_metrics(y_true_binary, y_pred_binary)
    rows.append(
        {
            "variant": "boosting_only",
            **boosting_only_metrics,
            "autoencoder_alone_attack_recall": float("nan"),  # no autoencoder in this variant
            "final_val_loss": float("nan"),
            "rounds_to_convergence": float("nan"),
            "total_communication_bytes": 0,
        }
    )
    logger.info("variant=boosting_only: %s", boosting_only_metrics)

    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    from fl_ids.eval.common import build_evaluation_setup
    from fl_ids.utils.config import load_config
    from fl_ids.utils.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Run the ablation table (component 12)")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--real-csv-path", default="data/raw/DNN-EdgeIIoT-dataset.csv")
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--poisoning-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    setup_logging()
    run_config = load_config(args.config_path)
    output_dir = Path(args.output_dir or run_config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    eval_setup = build_evaluation_setup(run_config, args.real_csv_path, seed=args.seed)
    ablation_df = run_ablation(
        eval_setup, run_config, num_rounds=args.num_rounds, poisoning_fraction=args.poisoning_fraction, seed=args.seed
    )

    csv_path = output_dir / "ablation_table.csv"
    ablation_df.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    print(ablation_df.to_string(index=False))
