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

from flwr.common import NDArrays, Parameters, Scalar, ndarrays_to_parameters
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


def build_initial_parameters(config: Config, input_dim: int) -> Parameters:
    """Build initial global autoencoder weights, so every client starts identically."""
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
) -> FedAvg:
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
        A configured `FedAvg` strategy.
    """
    return FedAvg(
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
) -> TrustFilteredStrategy:
    """Build the Phase 4 trust-filtered strategy (component 7).

    Args: same as `build_strategy`.

    Returns:
        A configured `TrustFilteredStrategy`.
    """
    return TrustFilteredStrategy(
        config,
        boosting_model_bytes,
        num_classes,
        benign_class,
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
    args = parser.parse_args()

    setup_logging()
    run_config = load_config(args.config_path)
    model_bytes = Path(args.boosting_model_path).read_bytes()

    if args.strategy == "custom_trust_filtered":
        strategy = build_trust_filtered_strategy(
            run_config, model_bytes, args.num_classes, args.benign_class, args.input_dim, args.min_clients
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
