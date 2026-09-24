"""Pipeline variants for component 12's comparisons, trained and scored one shared way.

The ablation table, the poisoning sweep and the zero-day experiment all
compare versions of the pipeline. Defining each variant once — how its
autoencoder is aggregated, whether clients pre-filter with boosting, and
how it decides "attack" at inference — and scoring every one with the
same `flag_attacks`/`detection_metrics` path keeps their rows comparable:
the earlier ablation scored boosting-only by bare argmax but the cascade
rows through the confidence threshold, so the columns didn't mean the
same thing across rows.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from fl_ids.eval.common import EvaluationSetup, calibrate_client_thresholds, per_row_thresholds
from fl_ids.eval.metrics import detection_metrics, flag_attacks
from fl_ids.eval.simulation import SimulationResult, run_simulated_fl_training
from fl_ids.models.autoencoder import Autoencoder
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.models.boosting_update import BoostingUpdater
from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Variant:
    """One version of the pipeline.

    Attributes:
        name: Row label.
        aggregation: `run_simulated_fl_training` aggregation mode, or None
            for a variant with no autoencoder (so no FL training).
        use_boosting_filter: Whether clients train their autoencoder only
            on traffic boosting passes through (component 3).
        detection_mode: `fl_ids.eval.metrics.DETECTION_MODES` entry used at inference.
    """

    name: str
    aggregation: str | None
    use_boosting_filter: bool
    detection_mode: str


FULL_PIPELINE = Variant("full_pipeline", "trust_filtered", True, "cascade")

ABLATION_VARIANTS: list[Variant] = [
    FULL_PIPELINE,
    Variant("plain_fedavg", "fedavg", True, "cascade"),
    Variant("trimmed_mean_only", "trimmed_mean_only", True, "cascade"),
    # No boosting anywhere: autoencoders train on unfiltered traffic, and
    # the autoencoder alone decides at inference.
    Variant("autoencoder_only", "trust_filtered", False, "autoencoder_only"),
    # No autoencoder: the cascade rule with no backstop.
    Variant("boosting_only", None, True, "boosting_only"),
]


@dataclass
class TrainedVariant:
    """A variant after training: its final boosting model, and its autoencoder and per-client thresholds if it has one.

    `boosting_model` differs from the setup's bootstrap model when the
    incremental update ran during training.
    """

    variant: Variant
    boosting_model: BoostingClassifier
    autoencoder: Autoencoder | None
    thresholds: dict[int, float] | None
    simulation: SimulationResult | None


def train_variant(
    variant: Variant,
    setup: EvaluationSetup,
    config: Config,
    num_rounds: int,
    malicious_client_ids: set[int],
    seed: int,
) -> TrainedVariant:
    """Run the variant's FL training (if any) and calibrate each client's threshold.

    Args:
        variant: What to train.
        setup: Shared evaluation setup.
        config: Full project config.
        num_rounds: FL rounds.
        malicious_client_ids: Clients running the sign-flip attacker.
        seed: Random seed.

    Returns:
        The trained variant.
    """
    if variant.aggregation is None:
        # No autoencoder, so no alerts: boosting-only never gets updated.
        return TrainedVariant(variant, setup.boosting_model, None, None, None)
    updater = None
    if config.boosting.update_every_n_rounds > 0 and setup.X_calib is not None:
        updater = BoostingUpdater(config.boosting, setup.X_calib, setup.y_calib, setup.class_names)
    result = run_simulated_fl_training(
        setup.client_data,
        setup.boosting_model,
        len(setup.class_names),
        setup.benign_class,
        config,
        num_rounds=num_rounds,
        aggregation=variant.aggregation,
        malicious_client_ids=malicious_client_ids,
        use_boosting_filter=variant.use_boosting_filter,
        seed=seed,
        boosting_updater=updater,
    )
    autoencoder, thresholds = calibrate_client_thresholds(result.final_weights, setup, config)
    return TrainedVariant(variant, result.final_boosting_model, autoencoder, thresholds, result)


def flag_rows(
    trained: TrainedVariant,
    setup: EvaluationSetup,
    X_raw: np.ndarray,
    X_norm: np.ndarray,
    client_ids: np.ndarray,
    config: Config,
    mode: str | None = None,
) -> np.ndarray:
    """Per-row attack flags for a trained variant, each row against its own client's threshold.

    Args:
        trained: `train_variant` output.
        setup: The setup it was trained from.
        X_raw: Raw-scale rows.
        X_norm: The same rows, normalized by their clients' scalers.
        client_ids: Each row's client.
        config: Full project config.
        mode: Override the variant's detection mode (e.g. to score its
            autoencoder on its own); defaults to `variant.detection_mode`.

    Returns:
        Boolean flags, one per row.
    """
    thresholds = per_row_thresholds(trained.thresholds, client_ids) if trained.thresholds is not None else None
    return flag_attacks(
        mode or trained.variant.detection_mode,
        trained.boosting_model,
        trained.autoencoder,
        thresholds,
        X_raw,
        X_norm,
        setup.class_names,
        config.cascade,
    )


def score_on_test_set(trained: TrainedVariant, setup: EvaluationSetup, config: Config) -> dict:
    """The variant's detection metrics on the shared test set, plus training diagnostics.

    Args:
        trained: `train_variant` output.
        setup: The setup it was trained from.
        config: Full project config (`evaluation.convergence_tolerance`).

    Returns:
        `detection_metrics` output plus `autoencoder_alone_attack_macro_recall`
        (the trained autoencoder's own health, isolated from boosting — NaN
        without one), `final_val_loss`, `rounds_to_convergence` and
        `total_communication_bytes`.
    """
    flags = flag_rows(trained, setup, setup.X_test_raw, setup.X_test_norm, setup.test_client_ids, config)
    row = detection_metrics(flags, setup.y_test, setup.benign_class, setup.class_names)

    if trained.autoencoder is None:
        row.update(
            autoencoder_alone_attack_macro_recall=float("nan"),
            final_val_loss=float("nan"),
            rounds_to_convergence=float("nan"),
            total_communication_bytes=0,
        )
        return row

    ae_flags = flag_rows(
        trained, setup, setup.X_test_raw, setup.X_test_norm, setup.test_client_ids, config, mode="autoencoder_only"
    )
    ae_metrics = detection_metrics(ae_flags, setup.y_test, setup.benign_class, setup.class_names)
    sim = trained.simulation
    row.update(
        autoencoder_alone_attack_macro_recall=ae_metrics["attack_macro_recall"],
        final_val_loss=sim.rounds[-1].mean_val_loss,
        rounds_to_convergence=sim.rounds_to_convergence(config.evaluation.convergence_tolerance),
        total_communication_bytes=sim.total_communication_bytes,
    )
    return row


def malicious_survival_rate(trained: TrainedVariant, malicious_client_ids: set[int]) -> float:
    """Mean per-round fraction of malicious clients that survived the trust filter; NaN if not applicable."""
    if not malicious_client_ids or trained.simulation is None:
        return float("nan")
    per_round = [
        len(set(r.survivors) & malicious_client_ids) / len(malicious_client_ids)
        for r in trained.simulation.rounds
        if r.survivors is not None
    ]
    return float(np.mean(per_round)) if per_round else float("nan")
