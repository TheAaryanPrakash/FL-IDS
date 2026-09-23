"""Custom Flower Strategy (component 7) — the piece that actually replaces FedAvg.

Subclasses `FedAvg`, overriding:
- `configure_fit`: captures the round's starting parameters (needed by
  component 5's delta computation) and carries the current boosting model
  out to clients in the per-round fit config (component 2's broadcast
  mechanism — same approach Phase 3's plain FedAvg used via
  `on_fit_config_fn`, just now on a subclass instead of a constructor hook).
- `aggregate_fit`: runs the cosine-similarity trust filter (component 5)
  and trimmed-mean aggregation (component 6) instead of FedAvg's plain
  weighted average, and records per-round trust diagnostics
  (`round_history`) for tests/the dashboard to inspect.
- `aggregate_evaluate`: folds each round's reconstruction-error loss into
  the same `round_history` entry, so one file has everything the
  dashboard (component 13) needs.

When `live_state_path` is given, `round_history` is written to that path
as JSON after every round — the dashboard polls this file rather than
needing a second, parallel state-tracking system.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from flwr.common import FitIns, FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from fl_ids.robustness.aggregation import apply_delta, trimmed_mean_delta
from fl_ids.robustness.trust_filter import TrustTracker, compute_delta, filter_client_deltas
from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)


class TrustFilteredStrategy(FedAvg):
    """FedAvg replacement: trust-filtered, trimmed-mean aggregation (components 5, 6)."""

    def __init__(
        self,
        config: Config,
        boosting_model_bytes: bytes,
        num_classes: int,
        benign_class: int,
        live_state_path: str | Path | None = None,
        boosting_metrics: dict | None = None,
        **kwargs,
    ) -> None:
        """Initialize the strategy.

        Args:
            config: Full project config (robustness thresholds, EMA alpha).
            boosting_model_bytes: Serialized boosting model, broadcast to
                clients unchanged each round (see `fl_ids.fl.server`).
            num_classes: Number of attack-type classes the boosting model
                was trained on.
            benign_class: Integer class index corresponding to "Normal".
            live_state_path: If given, `round_history` is written to this
                path as JSON after every round (fit and evaluate alike),
                for the dashboard (component 13) to poll live.
            boosting_metrics: Held-out metrics of the boosting model being
                broadcast (see `fl_ids.eval.metrics.stage_report_summary`),
                stamped into every round's entry alongside
                `boosting_model_version` so the dashboard shows which
                boosting model clients filtered with that round, and how
                good it was.
            **kwargs: Forwarded to `FedAvg.__init__` (min_fit_clients,
                initial_parameters, etc.).
        """
        super().__init__(**kwargs)
        self.config = config
        self.boosting_model_bytes = boosting_model_bytes
        self.num_classes = num_classes
        self.benign_class = benign_class
        self.live_state_path = Path(live_state_path) if live_state_path else None
        # Bumped whenever `boosting_model_bytes` is replaced. Component 2's
        # incremental update isn't implemented yet, so today this stays at 1
        # for the whole run.
        self.boosting_model_version = 1
        self.boosting_metrics = boosting_metrics
        self.trust_tracker = TrustTracker(config.robustness.trust_ema_alpha)
        self._round_start_weights: list = []
        self.round_history: list[dict] = []

    def _write_live_state(self) -> None:
        if self.live_state_path is None:
            return
        # Write-then-rename, so the polling dashboard never reads a half-written file.
        tmp_path = self.live_state_path.with_suffix(self.live_state_path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(self.round_history, indent=2))
        tmp_path.replace(self.live_state_path)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> list[tuple[ClientProxy, FitIns]]:
        """Capture this round's starting weights (for delta computation) and broadcast the boosting model."""
        self._round_start_weights = parameters_to_ndarrays(parameters)

        fit_config: dict[str, Scalar] = {
            "boosting_model_bytes": self.boosting_model_bytes,
            "num_classes": self.num_classes,
            "benign_class": self.benign_class,
            "server_round": server_round,
        }
        # Reuse FedAvg's client-sampling logic; just swap in our fit config.
        sampled = super().configure_fit(server_round, parameters, client_manager)
        return [(client_proxy, FitIns(parameters, fit_config)) for client_proxy, _ in sampled]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list,
    ) -> tuple[Optional[Parameters], dict[str, Scalar]]:
        """Trust-filter client deltas, trimmed-mean aggregate, and record diagnostics.

        Returns:
            (new_global_parameters, metrics). If every client is excluded
            by the trust filter, the round's starting parameters are
            returned unchanged (no aggregation happens on zero survivors).
        """
        if not results:
            return None, {}

        client_deltas: dict[int, np.ndarray] = {}
        filtered_fractions: dict[int, float] = {}
        for client_proxy, fit_res in results:
            new_weights = parameters_to_ndarrays(fit_res.parameters)
            delta = compute_delta(new_weights, self._round_start_weights)
            cid = int(fit_res.metrics.get("client_id", hash(getattr(client_proxy, "cid", id(client_proxy)))))
            client_deltas[cid] = delta
            if "filtered_fraction" in fit_res.metrics:
                filtered_fractions[cid] = float(fit_res.metrics["filtered_fraction"])

        filter_result = filter_client_deltas(client_deltas, self.config.robustness)
        trust_scores = self.trust_tracker.update(filter_result.client_ids, filter_result.similarities)

        logger.info(
            "Round %d trust filter: %d/%d clients survived. Trust scores: %s",
            server_round,
            len(filter_result.survivors),
            len(filter_result.client_ids),
            {cid: round(trust_scores[cid], 3) for cid in filter_result.client_ids},
        )

        self.round_history.append(
            {
                "round": server_round,
                "similarities": {cid: float(sim) for cid, sim in zip(filter_result.client_ids, filter_result.similarities)},
                "is_outlier": {cid: bool(flag) for cid, flag in zip(filter_result.client_ids, filter_result.is_outlier)},
                "trust_scores": dict(trust_scores),
                "survivors": list(filter_result.survivors),
                "filtered_fractions": filtered_fractions,
                "mean_filtered_fraction": float(np.mean(list(filtered_fractions.values()))) if filtered_fractions else None,
                "boosting_model_version": self.boosting_model_version,
                "boosting_metrics": self.boosting_metrics,
            }
        )
        self._write_live_state()

        if not filter_result.survivors:
            logger.warning("Round %d: every client excluded by the trust filter; skipping aggregation", server_round)
            return ndarrays_to_parameters(self._round_start_weights), {"num_survivors": 0}

        surviving_deltas = [filter_result.clipped_deltas[cid] for cid in filter_result.survivors]
        aggregated_delta = trimmed_mean_delta(surviving_deltas, self.config.robustness.trim_fraction)
        new_global_weights = apply_delta(self._round_start_weights, aggregated_delta)

        return ndarrays_to_parameters(new_global_weights), {"num_survivors": len(filter_result.survivors)}

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list,
        failures: list,
    ) -> tuple[Optional[float], dict[str, Scalar]]:
        """Fold this round's per-client evaluate results into `round_history`, then defer to FedAvg.

        Component 3's per-round reconstruction error (global + per-client)
        lives here: each client's `evaluate()` returns its own mean
        reconstruction error on benign validation data, plus its
        recomputed anomaly threshold.
        """
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        per_client_loss: dict[int, float] = {}
        per_client_threshold: dict[int, float] = {}
        for client_proxy, evaluate_res in results:
            cid = int(evaluate_res.metrics.get("client_id", hash(getattr(client_proxy, "cid", id(client_proxy)))))
            per_client_loss[cid] = float(evaluate_res.loss)
            if "anomaly_threshold" in evaluate_res.metrics:
                per_client_threshold[cid] = float(evaluate_res.metrics["anomaly_threshold"])

        for entry in reversed(self.round_history):
            if entry["round"] == server_round:
                entry["mean_reconstruction_error"] = float(loss) if loss is not None else None
                entry["per_client_reconstruction_error"] = per_client_loss
                entry["per_client_anomaly_threshold"] = per_client_threshold
                break
        self._write_live_state()

        return loss, metrics
