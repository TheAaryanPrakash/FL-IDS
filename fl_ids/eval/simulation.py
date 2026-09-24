"""In-process FL training simulation, for evaluation sweeps (component 12).

Phase 3/4 already proved the real thing works: real separate OS
processes talking over a real gRPC socket (see
tests/test_fl_integration.py, tests/test_fl_robustness_integration.py).
The poisoning-resistance sweep and ablation table need to run that same
training logic dozens of times over (multiple poisoning fractions,
multiple ablation variants) — paying real subprocess/gRPC overhead for
each run would make that impractical, and isn't what's being tested at
this point. This module drives the *exact same* `AutoencoderClient` /
`SignFlipAttackerClient` / robustness-layer code directly via function
calls instead of a network, which is the standard "simulation harness"
approach (Flower itself ships an analogous `Simulation` API for this
reason) — a legitimate different use case from re-proving the plumbing
works, not a shortcut around it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from fl_ids.fl.client import AutoencoderClient
from fl_ids.models.autoencoder import Autoencoder, get_weights, set_weights
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.models.boosting_update import SURFACE_ALERTS_KEY, BoostingUpdater, BoostingUpdateRecord, decode_alerts
from fl_ids.robustness.aggregation import apply_delta, trimmed_mean_delta
from fl_ids.robustness.attackers import SignFlipAttackerClient, attacker_claimed_examples
from fl_ids.robustness.trust_filter import TrustTracker, compute_delta, filter_client_deltas, flatten_weights
from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)

AggregationMode = str  # "trust_filtered" | "fedavg" | "trimmed_mean_only"


@dataclass
class SimulationRoundRecord:
    """One round's diagnostics."""

    round: int
    mean_val_loss: float
    communication_bytes: int
    trust_scores: dict[int, float] | None = None
    survivors: list[int] | None = None
    boosting_update: BoostingUpdateRecord | None = None


@dataclass
class SimulationResult:
    """Full simulation output."""

    final_weights: list[np.ndarray]
    rounds: list[SimulationRoundRecord] = field(default_factory=list)
    # The boosting model broadcast at the end: the input model, plus any
    # incremental updates.
    final_boosting_model: BoostingClassifier | None = None

    @property
    def total_communication_bytes(self) -> int:
        return sum(r.communication_bytes for r in self.rounds)

    def rounds_to_convergence(self, tolerance: float) -> float:
        """First round after which validation loss stays within `tolerance` of its best; NaN if it never settles.

        A run counts as converged only if it *ends* within the band
        (loss <= best * (1 + tolerance)). A run that diverges — e.g. plain
        FedAvg under poisoning, whose loss is lowest in round 1 and then
        explodes — reports NaN rather than "converged in round 1".

        Args:
            tolerance: Relative band around the best loss
                (`evaluation.convergence_tolerance`).

        Returns:
            A 1-based round number, or NaN.
        """
        losses = np.array([r.mean_val_loss for r in self.rounds], dtype=np.float64)
        if len(losses) == 0 or not np.isfinite(losses[-1]):
            return float("nan")
        band = np.nanmin(losses) * (1 + tolerance)
        within = np.isfinite(losses) & (losses <= band)
        if not within[-1]:
            return float("nan")
        # Last round outside the band; convergence starts right after it.
        outside = np.where(~within)[0]
        return float(outside[-1] + 2) if len(outside) else 1.0


def _make_clients(
    client_data: dict[int, dict[str, np.ndarray]],
    config: Config,
    input_dim: int,
    malicious_client_ids: set[int],
    use_boosting_filter: bool,
    amplification: float,
) -> dict[int, AutoencoderClient]:
    clients = {}
    claimed = attacker_claimed_examples(config.robustness, [len(d["X"]) for d in client_data.values()])
    for cid, data in client_data.items():
        args = (cid, data["X"], data["X_raw"], data["X_val_benign"], config, input_dim)
        kwargs = {"use_boosting_filter": use_boosting_filter, "y_train": data["y"]}
        if cid in malicious_client_ids:
            clients[cid] = SignFlipAttackerClient(
                *args, amplification=amplification, claimed_num_examples=claimed, **kwargs
            )
        else:
            clients[cid] = AutoencoderClient(*args, **kwargs)
    return clients


def run_simulated_fl_training(
    client_data: dict[int, dict[str, np.ndarray]],
    boosting_model: BoostingClassifier,
    num_classes: int,
    benign_class: int,
    config: Config,
    num_rounds: int,
    aggregation: AggregationMode = "trust_filtered",
    malicious_client_ids: set[int] | None = None,
    use_boosting_filter: bool = True,
    amplification: float = 5.0,
    seed: int = 0,
    boosting_updater: BoostingUpdater | None = None,
) -> SimulationResult:
    """Run `num_rounds` of in-process FL training and return per-round diagnostics.

    Args:
        client_data: `{client_id: {"X", "X_raw", "y", "X_val_benign", ...}}`
            (component 1's per-client output shape).
        boosting_model: Trained boosting classifier, broadcast unchanged
            each round (matches Phase 3/4's approach).
        num_classes: Number of attack-type classes.
        benign_class: Integer class index corresponding to "Normal".
        config: Full project config.
        num_rounds: Number of FL rounds to run.
        aggregation: `"trust_filtered"` (components 5+6, the full
            pipeline), `"fedavg"` (plain weighted average, no robustness
            layer at all), or `"trimmed_mean_only"` (trimmed mean over
            every client's raw delta, skipping the cosine/MAD filter).
        malicious_client_ids: Client IDs that run as
            `SignFlipAttackerClient` instead of the honest client.
        use_boosting_filter: Forwarded to every client — False trains
            autoencoders on unfiltered local traffic (the "autoencoder-only,
            no boosting pre-filter" ablation case).
        amplification: Sign-flip attacker delta amplification.
        seed: Random seed for initial global weights.
        boosting_updater: If given, the incremental boosting update
            (component 2), exactly as `TrustFilteredStrategy` runs it: on
            update rounds clients surface alerts, and those from surviving
            clients (every client, for the aggregation modes without a
            trust filter) continue boosting's training for later rounds.

    Returns:
        A `SimulationResult` with final weights and per-round diagnostics
        (validation loss, communication bytes, and — for
        `"trust_filtered"` — trust scores/survivors per round).
    """
    malicious_client_ids = malicious_client_ids or set()
    input_dim = next(iter(client_data.values()))["X"].shape[1]
    input_dim_raw = next(iter(client_data.values()))["X_raw"].shape[1]

    clients = _make_clients(client_data, config, input_dim, malicious_client_ids, use_boosting_filter, amplification)

    # Explicit seed: torch's default generator differs per process, so an
    # unseeded init made each run (and each ablation variant) start from
    # different weights -- breaking "same data, same seed" comparisons.
    torch.manual_seed(seed)
    torch_seed_model = Autoencoder(input_dim, config.autoencoder.hidden_dims, config.autoencoder.bottleneck_dim)
    global_weights = get_weights(torch_seed_model)

    boosting_bytes = boosting_model.to_bytes()
    trust_tracker = TrustTracker(config.robustness.trust_ema_alpha) if aggregation == "trust_filtered" else None

    result = SimulationResult(final_weights=global_weights)

    for round_num in range(1, num_rounds + 1):
        update_round = boosting_updater is not None and boosting_updater.is_update_round(round_num)
        fit_config = {
            "boosting_model_bytes": boosting_bytes,
            "num_classes": num_classes,
            "benign_class": benign_class,
            "server_round": round_num,
            SURFACE_ALERTS_KEY: update_round,
        }

        fit_results: dict[int, tuple[list[np.ndarray], int]] = {}
        fit_metrics: dict[int, dict] = {}
        comm_bytes = 0
        down_bytes = flatten_weights(global_weights).nbytes
        for cid, client in clients.items():
            new_weights, num_examples, metrics = client.fit(global_weights, fit_config)
            fit_results[cid] = (new_weights, num_examples)
            fit_metrics[cid] = metrics
            comm_bytes += down_bytes + flatten_weights(new_weights).nbytes

        trust_scores = None
        survivors = None
        if aggregation == "trust_filtered":
            deltas = {cid: compute_delta(w, global_weights) for cid, (w, _n) in fit_results.items()}
            filter_result = filter_client_deltas(deltas, config.robustness)
            trust_scores = trust_tracker.update(filter_result.client_ids, filter_result.similarities)
            survivors = list(filter_result.survivors)
            if survivors:
                surviving_deltas = [filter_result.clipped_deltas[cid] for cid in survivors]
                aggregated_delta = trimmed_mean_delta(surviving_deltas, config.robustness.trim_fraction)
                global_weights = apply_delta(global_weights, aggregated_delta)
        elif aggregation == "trimmed_mean_only":
            deltas = [compute_delta(w, global_weights) for w, _n in fit_results.values()]
            aggregated_delta = trimmed_mean_delta(deltas, config.robustness.trim_fraction)
            global_weights = apply_delta(global_weights, aggregated_delta)
        elif aggregation == "fedavg":
            total_examples = sum(n for _w, n in fit_results.values()) or 1
            deltas = [compute_delta(w, global_weights) * (n / total_examples) for w, n in fit_results.values()]
            aggregated_delta = np.sum(deltas, axis=0)
            global_weights = apply_delta(global_weights, aggregated_delta)
        else:
            raise ValueError(f"Unknown aggregation mode: {aggregation}")

        update_record = None
        if update_round:
            alerts = {cid: decode_alerts(m, input_dim_raw) for cid, m in fit_metrics.items()}
            surviving = set(survivors) if survivors is not None else set(clients)
            boosting_model, update_record = boosting_updater.update(boosting_model, round_num, alerts, surviving)
            boosting_bytes = boosting_model.to_bytes()

        val_losses, val_weights = [], []
        for client in clients.values():
            loss, num_examples, _ = client.evaluate(global_weights, {})
            if num_examples > 0:
                val_losses.append(loss)
                val_weights.append(num_examples)
        mean_val_loss = float(np.average(val_losses, weights=val_weights)) if val_losses else float("nan")

        result.rounds.append(
            SimulationRoundRecord(
                round=round_num,
                mean_val_loss=mean_val_loss,
                communication_bytes=comm_bytes,
                trust_scores=trust_scores,
                survivors=survivors,
                boosting_update=update_record,
            )
        )
        logger.info("Round %d: mean_val_loss=%.4f, comm_bytes=%d", round_num, mean_val_loss, comm_bytes)

    result.final_weights = global_weights
    result.final_boosting_model = boosting_model
    return result
