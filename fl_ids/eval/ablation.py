"""Ablation table (component 12).

Full pipeline vs. plain FedAvg vs. trimmed-mean-only (no cosine filter)
vs. autoencoder-only (no boosting) vs. boosting-only (no autoencoder
backstop) — same data, same seed, one comparison table, at a fixed
moderate poisoning fraction (so the robustness variants actually have
something to defend against).

Every row is scored the same way (`fl_ids.eval.variants`):

- `attack_macro_recall` / `attack_micro_recall` / `benign_fpr` /
  `detection_f1` on the shared test set. Its attack types are all in
  boosting's training labels, so this half of the table can only show the
  autoencoder's *cost* (extra false positives), never its purpose.
- `zero_day_macro_recall`: mean detection rate over held-out attack
  classes, each withheld from all training (`fl_ids.eval.zero_day`) —
  the column that shows what the autoencoder backstop is for.
- `autoencoder_alone_attack_macro_recall` and `final_val_loss`: the
  trained autoencoder's own health, isolated from boosting (which isn't
  FL-trained, so poisoning can't touch it and it would otherwise mask a
  broken backstop).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fl_ids.eval.common import EvaluationSetup, select_malicious_clients
from fl_ids.eval.variants import ABLATION_VARIANTS, score_on_test_set, train_variant
from fl_ids.eval.zero_day import zero_day_detection_by_run
from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)

TABLE_COLUMNS = [
    "variant",
    "attack_macro_recall",
    "attack_micro_recall",
    "benign_fpr",
    "detection_f1",
    "zero_day_macro_recall",
    "autoencoder_alone_attack_macro_recall",
    "final_val_loss",
    "rounds_to_convergence",
    "total_communication_bytes",
]


def run_ablation(
    setup: EvaluationSetup,
    config: Config,
    num_rounds: int,
    poisoning_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and score every ablation variant on the shared test set.

    Args:
        setup: Shared evaluation setup.
        config: Full project config.
        num_rounds: FL rounds per variant (boosting-only has no FL training).
        poisoning_fraction: Fraction of clients malicious, held fixed
            across variants so the robustness comparison is meaningful.
        seed: Random seed (model init and malicious-client selection).

    Returns:
        (table, per_class): one row per variant with `TABLE_COLUMNS`
        (`zero_day_macro_recall` left NaN — see `add_zero_day_column`),
        and a long table of per-attack-type recall (`variant`,
        `attack_type`, `recall`).
    """
    malicious_ids = select_malicious_clients(list(setup.client_data), poisoning_fraction, seed)
    logger.info("Ablation: poisoning_fraction=%.2f -> malicious clients=%s", poisoning_fraction, sorted(malicious_ids))

    rows, per_class_rows = [], []
    for variant in ABLATION_VARIANTS:
        trained = train_variant(variant, setup, config, num_rounds, malicious_ids, seed)
        scores = score_on_test_set(trained, setup, config)
        per_class = scores.pop("per_class_recall")
        rows.append({"variant": variant.name, **scores, "zero_day_macro_recall": float("nan")})
        per_class_rows += [{"variant": variant.name, "attack_type": k, "recall": v} for k, v in per_class.items()]
        logger.info(
            "variant=%s: attack_macro_recall=%.3f benign_fpr=%.4f autoencoder_alone=%.3f",
            variant.name, scores["attack_macro_recall"], scores["benign_fpr"],
            scores["autoencoder_alone_attack_macro_recall"],
        )

    return pd.DataFrame(rows)[TABLE_COLUMNS], pd.DataFrame(per_class_rows)


def add_zero_day_column(
    table: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    benign_class: int,
    config: Config,
    num_rounds: int,
    poisoning_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fill `zero_day_macro_recall` by retraining every variant once per held-out attack class.

    Args:
        table: `run_ablation` output.
        X, y, class_names, benign_class: The full dataset (`load_and_encode` output).
        config: Full project config (`evaluation.zero_day_*`).
        num_rounds: FL rounds per training.
        poisoning_fraction: Same fraction as the main table.
        seed: Same seed as the main table.

    Returns:
        (table with the column filled, per-(variant, holdout class) detection rates).
    """
    runs = [(v.name, v, poisoning_fraction) for v in ABLATION_VARIANTS]
    detail = zero_day_detection_by_run(runs, X, y, class_names, benign_class, config, num_rounds, seed)
    means = detail.groupby("run")["detection_rate"].mean()
    table = table.copy()
    table["zero_day_macro_recall"] = table["variant"].map(means)
    return table, detail.rename(columns={"run": "variant"})


if __name__ == "__main__":
    import argparse

    from fl_ids.data.pipeline import load_and_encode
    from fl_ids.eval.common import build_evaluation_setup_from_arrays
    from fl_ids.utils.config import load_config
    from fl_ids.utils.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Run the ablation table (component 12)")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--real-csv-path", default="data/raw/DNN-EdgeIIoT-dataset.csv")
    parser.add_argument("--num-rounds", type=int, default=10)
    parser.add_argument("--poisoning-fraction", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--skip-zero-day", action="store_true",
        help="Leave zero_day_macro_recall empty (skips one retraining per variant per held-out class)",
    )
    args = parser.parse_args()

    setup_logging()
    run_config = load_config(args.config_path)
    output_dir = Path(args.output_dir or run_config.evaluation.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_all, y_all, label_encoder, _feature_names, benign = load_and_encode(args.real_csv_path, run_config.data.capture_repairs)
    names = list(label_encoder.classes_)
    eval_setup = build_evaluation_setup_from_arrays(X_all, y_all, names, benign, run_config, args.seed)
    ablation_df, per_class_df = run_ablation(
        eval_setup, run_config, num_rounds=args.num_rounds, poisoning_fraction=args.poisoning_fraction, seed=args.seed
    )
    per_class_df.to_csv(output_dir / "ablation_per_class_recall.csv", index=False)

    if not args.skip_zero_day:
        ablation_df, zero_day_df = add_zero_day_column(
            ablation_df, X_all, y_all, names, benign, run_config,
            num_rounds=args.num_rounds, poisoning_fraction=args.poisoning_fraction, seed=args.seed,
        )
        zero_day_df.to_csv(output_dir / "ablation_zero_day_detail.csv", index=False)

    csv_path = output_dir / "ablation_table.csv"
    ablation_df.to_csv(csv_path, index=False)
    logger.info("Saved %s", csv_path)

    print(ablation_df.to_string(index=False))
