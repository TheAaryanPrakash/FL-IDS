"""Flower server (component 8) — real distributed FL run.

Phase 3 introduced vanilla `flwr.server.strategy.FedAvg`, broadcasting the
boosting model each round via `FedAvg`'s `on_fit_config_fn` hook. Phase 4
adds `TrustFilteredStrategy` (component 7), which replaces FedAvg's plain
weighted average with trust-filtered, trimmed-mean aggregation (components
5, 6) while broadcasting the boosting model the same way. Both strategies
are built here and handed to `run_server`, which just starts the server
and blocks until `num_rounds` complete — it doesn't care which strategy
it's running.

Runs against real separate client processes (component 4's
`python -m fl_ids.fl.client`), not an in-process simulation.
"""

from __future__ import annotations

import logging

import torch
from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import ServerConfig, start_server
from flwr.server.history import History
from flwr.server.strategy import FedAvg, Strategy

from fl_ids.fl.strategy import TrustFilteredStrategy
from fl_ids.models.autoencoder import Autoencoder, get_weights
from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)


def make_fit_config_fn(boosting_model_bytes: bytes, num_classes: int, benign_class: int):
    """Build the `on_fit_config_fn` callback that broadcasts the boosting model.

    Args:
        boosting_model_bytes: Serialized boosting model (see
            `BoostingClassifier.to_bytes`), broadcast unchanged each round
            for Phase 3 (incremental server-side updates are a later
            refinement, not this phase's concern).
        num_classes: Number of attack-type classes the boosting model was
            trained on.
        benign_class: Integer class index corresponding to "Normal".

    Returns:
        A callable `(server_round: int) -> dict[str, Scalar]` suitable for
        `FedAvg(on_fit_config_fn=...)`.
    """

    def fit_config(server_round: int) -> dict[str, Scalar]:
        return {
            "boosting_model_bytes": boosting_model_bytes,
            "num_classes": num_classes,
            "benign_class": benign_class,
            "server_round": server_round,
        }

    return fit_config


class RecordingFedAvg(FedAvg):
    """Plain FedAvg that keeps the latest aggregated weights, like `TrustFilteredStrategy.latest_weights`.

    Flower's `start_server` returns only the run's history, so without this
    a FedAvg run's trained model would be unrecoverable once the server exits.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.latest_weights: list | None = None

    def aggregate_fit(self, server_round, results, failures):
        """FedAvg aggregation, recording the resulting weights."""
        parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if parameters is not None:
            self.latest_weights = parameters_to_ndarrays(parameters)
        return parameters, metrics


def build_initial_parameters(config: Config, input_dim: int) -> Parameters:
    """Build initial global autoencoder weights, so every client starts identically.

    Seeded from `config.seed`: torch's default generator is *not* a fixed
    constant across processes, so an unseeded init made every real FL run
    start from different weights -- and every trust score downstream of it
    irreproducible.
    """
    torch.manual_seed(config.seed)
    init_model = Autoencoder(input_dim, config.autoencoder.hidden_dims, config.autoencoder.bottleneck_dim)
    weights: NDArrays = get_weights(init_model)
    return ndarrays_to_parameters(weights)


def build_strategy(
    config: Config,
    boosting_model_bytes: bytes,
    num_classes: int,
    benign_class: int,
    input_dim: int,
    min_clients: int,
) -> RecordingFedAvg:
    """Build the Phase 3 vanilla FedAvg strategy.

    Args:
        config: Full project config.
        boosting_model_bytes: Serialized boosting model to broadcast.
        num_classes: Number of attack-type classes.
        benign_class: Integer class index corresponding to "Normal".
        input_dim: Autoencoder input width, for building initial parameters.
        min_clients: Minimum number of clients required for fit/evaluate
            to proceed each round (also used as fraction_fit/evaluate's
            denominator via min_available_clients).

    Returns:
        A configured `RecordingFedAvg` (plain FedAvg that keeps its final weights).
    """
    return RecordingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        on_fit_config_fn=make_fit_config_fn(boosting_model_bytes, num_classes, benign_class),
        initial_parameters=build_initial_parameters(config, input_dim),
    )


def build_trust_filtered_strategy(
    config: Config,
    boosting_model_bytes: bytes,
    num_classes: int,
    benign_class: int,
    input_dim: int,
    min_clients: int,
    live_state_path: str | None = None,
    boosting_metrics: dict | None = None,
) -> TrustFilteredStrategy:
    """Build the Phase 4 trust-filtered strategy (component 7).

    Args:
        live_state_path: If given, forwarded to `TrustFilteredStrategy` so
            the dashboard (component 13) can poll live per-round state.
        boosting_metrics: If given, forwarded to `TrustFilteredStrategy`
            (the broadcast boosting model's held-out metrics).
        (remaining args: same as `build_strategy`.)

    Returns:
        A configured `TrustFilteredStrategy`.
    """
    return TrustFilteredStrategy(
        config,
        boosting_model_bytes,
        num_classes,
        benign_class,
        live_state_path=live_state_path,
        boosting_metrics=boosting_metrics,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        initial_parameters=build_initial_parameters(config, input_dim),
    )


def run_server(
    config: Config,
    strategy: Strategy,
    server_address: str,
    num_rounds: int | None = None,
) -> History:
    """Start a real Flower server with the given strategy and block until `num_rounds` complete.

    Args:
        config: Full project config.
        strategy: A configured `FedAvg` or `TrustFilteredStrategy` (or any
            other `Strategy`) — this function doesn't care which.
        server_address: Address to bind, e.g. "127.0.0.1:8080".
        num_rounds: Overrides `config.fl.num_rounds` if given (useful for
            fast tests).

    Returns:
        The Flower `History` (per-round distributed losses/metrics).
    """
    rounds = num_rounds if num_rounds is not None else config.fl.num_rounds
    logger.info("Starting Flower server on %s for %d rounds", server_address, rounds)
    return start_server(
        server_address=server_address,
        config=ServerConfig(num_rounds=rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    from fl_ids.utils.config import load_config
    from fl_ids.utils.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Run a real Flower server process")
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--boosting-model-path", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--benign-class", type=int, required=True)
    parser.add_argument("--input-dim", type=int, required=True)
    parser.add_argument("--min-clients", type=int, required=True)
    parser.add_argument("--num-rounds", type=int, default=None)
    parser.add_argument(
        "--strategy",
        choices=["fedavg", "custom_trust_filtered"],
        default="fedavg",
        help="'custom_trust_filtered' uses component 7's TrustFilteredStrategy (Phase 4)",
    )
    parser.add_argument("--history-output", required=True)
    parser.add_argument(
        "--live-state-path", default=None,
        help="Poll this path (custom_trust_filtered strategy only) with the dashboard for live per-round state",
    )
    parser.add_argument(
        "--boosting-metrics-path", default=None,
        help="JSON file of the boosting model's held-out metrics (fl_ids.eval.metrics.stage_report_summary), "
        "recorded in the live state for the dashboard",
    )
    parser.add_argument(
        "--final-weights-output", default=None,
        help="Save the final global autoencoder weights to this .npz (Phase A's trained model)",
    )
    args = parser.parse_args()

    setup_logging()
    run_config = load_config(args.config_path)
    model_bytes = Path(args.boosting_model_path).read_bytes()
    boosting_metrics = json.loads(Path(args.boosting_metrics_path).read_text()) if args.boosting_metrics_path else None

    if args.strategy == "custom_trust_filtered":
        strategy = build_trust_filtered_strategy(
            run_config, model_bytes, args.num_classes, args.benign_class, args.input_dim, args.min_clients,
            live_state_path=args.live_state_path, boosting_metrics=boosting_metrics,
        )
    else:
        strategy = build_strategy(
            run_config, model_bytes, args.num_classes, args.benign_class, args.input_dim, args.min_clients
        )

    history = run_server(run_config, strategy, args.server_address, num_rounds=args.num_rounds)

    output = {
        "losses_distributed": history.losses_distributed,
        "metrics_distributed_fit": {
            k: [(r, v) for r, v in vs] for k, vs in history.metrics_distributed_fit.items()
        },
        "metrics_distributed": {
            k: [(r, v) for r, v in vs] for k, vs in history.metrics_distributed.items()
        },
    }
    if isinstance(strategy, TrustFilteredStrategy):
        output["round_history"] = strategy.round_history

    Path(args.history_output).write_text(json.dumps(output))

    if args.final_weights_output:
        import numpy as np

        if strategy.latest_weights is None:
            raise RuntimeError("No aggregation happened, so there are no final weights to save")
        np.savez(args.final_weights_output, *strategy.latest_weights)
        logger.info("Saved final global weights to %s", args.final_weights_output)
