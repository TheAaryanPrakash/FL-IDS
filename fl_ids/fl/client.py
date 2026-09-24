"""Flower client (component 4).

Wraps the client-side autoencoder (component 3) in `flwr.client.NumPyClient`.
`fit()` first runs the current global boosting model (broadcast via the
strategy's per-round fit config, component 2) locally to filter local
traffic down to the "normal" subset per the shared cascade rule, trains
the autoencoder on exactly that subset, and returns updated weights plus
a fixed-bin reconstruction-error histogram (not raw traffic). On the
server's update rounds it also surfaces a capped number of alerts —
traffic boosting passed that the global autoencoder flags, with their
analyst-confirmed labels — for the incremental boosting update
(`fl_ids.models.boosting_update`), the one case where feature rows leave
the client.

Runnable both as a real separate process (this module's `__main__`, using
`flwr.client.start_client`) and instantiated directly for testing
(`AutoencoderClient(...)`, no networking involved).
"""

from __future__ import annotations

import logging

import numpy as np
from flwr.client import NumPyClient
from flwr.common import NDArrays, Scalar

from fl_ids.models.autoencoder import (
    Autoencoder,
    compute_anomaly_threshold,
    get_weights,
    reconstruction_error,
    reconstruction_error_histogram,
    set_weights,
    train_autoencoder,
)
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.models.boosting_update import SURFACE_ALERTS_KEY, SurfacedAlerts, encode_alerts, select_alerts
from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)


class AutoencoderClient(NumPyClient):
    """Federated client for the autoencoder cascade stage.

    Filters local traffic through the global boosting model (cascade
    stage 1) before training the local autoencoder on the resulting
    "normal" subset — never on unfiltered local traffic (component 3's
    requirement). The boosting filter itself runs on *raw* features
    (`X_train_raw`) — see `fl_ids.models.boosting`'s module docstring for
    why per-client-normalized input silently breaks a shared tree model —
    while the autoencoder trains on the *normalized* counterpart
    (`X_train`) of exactly the rows the filter passed.
    """

    def __init__(
        self,
        client_id: int,
        X_train: np.ndarray,
        X_train_raw: np.ndarray,
        X_val_benign: np.ndarray,
        config: Config,
        input_dim: int,
        use_boosting_filter: bool = True,
        y_train: np.ndarray | None = None,
    ) -> None:
        """Initialize the client.

        Args:
            client_id: This client's identifier (used only for logging).
            X_train: This client's full local training features,
                per-client normalized (component 1's "X") — what the
                autoencoder actually trains on, after boosting filtering.
            X_train_raw: The same rows as `X_train`, in raw (unnormalized)
                scale (component 1's "X_raw") — what the boosting filter
                runs on.
            X_val_benign: This client's held-out benign validation slice
                (normalized), for recomputing the anomaly threshold each
                round.
            config: Full project config.
            input_dim: Number of input features (autoencoder input width).
            use_boosting_filter: When False, trains on *all* local traffic
                unfiltered instead of the boosting-confident-"normal"
                subset. Default True is component 3's actual requirement;
                False exists only to support the "autoencoder-only (no
                boosting pre-filter)" ablation row (component 12) — never
                the default for real training.
            y_train: Labels for `X_train`'s rows — the label an analyst
                would confirm for a surfaced alert (see
                `fl_ids.models.boosting_update`). Without it the client
                can't surface alerts, and ignores requests to.
        """
        self.client_id = client_id
        self.X_train = X_train
        self.X_train_raw = X_train_raw
        self.X_val_benign = X_val_benign
        self.config = config
        self.use_boosting_filter = use_boosting_filter
        self.y_train = y_train
        self.model = Autoencoder(
            input_dim, config.autoencoder.hidden_dims, config.autoencoder.bottleneck_dim
        )
        self.last_filtered_fraction: float | None = None

    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        """Return current local autoencoder weights."""
        return get_weights(self.model)

    def fit(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[NDArrays, int, dict[str, Scalar]]:
        """Load global weights, filter local data via boosting, train, return updates.

        Args:
            parameters: Global autoencoder weights for this round.
            config: Per-round fit config from the strategy — must include
                `boosting_model_bytes`, `num_classes`, `benign_class`.

        Returns:
            (updated_weights, num_training_examples, metrics), where
            metrics includes a serialized fixed-bin reconstruction-error
            histogram (`reconstruction_error_histogram`, raw int64 bytes)
            and the fraction of local traffic that passed the boosting
            filter (`filtered_fraction`).
        """
        set_weights(self.model, parameters)

        boosting_model = BoostingClassifier.from_bytes(
            config["boosting_model_bytes"],
            self.config.boosting,
            num_classes=int(config["num_classes"]),
            benign_class=int(config["benign_class"]),
            seed=self.config.seed,
            confidence_threshold=self.config.cascade.confidence_threshold,
        )

        if self.use_boosting_filter:
            mask = boosting_model.passes_to_autoencoder(self.X_train_raw)
        else:
            mask = np.ones(len(self.X_train_raw), dtype=bool)
        X_filtered = self.X_train[mask]
        self.last_filtered_fraction = float(mask.mean()) if len(mask) else 0.0

        # Alerts are raised by the model deployed this round -- the global
        # weights just received -- so score before local training changes them.
        alert_metrics: dict[str, Scalar] = {}
        if config.get(SURFACE_ALERTS_KEY):
            alerts = self._surface_alerts(mask, int(config.get("server_round", 0)))
            alert_metrics = encode_alerts(alerts)

        if len(X_filtered) > 0:
            train_autoencoder(self.model, X_filtered, self.config.autoencoder, seed=self.config.seed)
            errors = reconstruction_error(self.model, X_filtered)
        else:
            logger.warning(
                "Client %d: boosting filter passed zero of %d local samples this round",
                self.client_id,
                len(self.X_train_raw),
            )
            errors = np.array([], dtype=np.float32)

        hist = reconstruction_error_histogram(
            errors,
            self.config.autoencoder.reconstruction_error_bins,
            tuple(self.config.autoencoder.reconstruction_error_range),
        )

        metrics: dict[str, Scalar] = {
            "reconstruction_error_histogram": hist.tobytes(),
            "filtered_fraction": self.last_filtered_fraction,
            "client_id": self.client_id,
            **alert_metrics,
        }
        return get_weights(self.model), len(X_filtered), metrics

    def _surface_alerts(self, passed_mask: np.ndarray, server_round: int) -> SurfacedAlerts:
        """Rows boosting passed that the current global autoencoder flags, capped and labeled.

        Uses this client's own threshold (computed on its benign validation
        slice with the same global weights), as its detector would in
        deployment. Returns nothing without labels or benign validation data.
        """
        n_features = self.X_train_raw.shape[1]
        empty = SurfacedAlerts(np.empty((0, n_features), dtype=np.float32), np.empty(0, dtype=np.int64))
        if self.y_train is None or len(self.X_val_benign) == 0 or not passed_mask.any():
            if self.y_train is None:
                logger.warning("Client %d: asked to surface alerts but has no labels to confirm them", self.client_id)
            return empty
        threshold = compute_anomaly_threshold(
            reconstruction_error(self.model, self.X_val_benign), self.config.autoencoder.anomaly_percentile
        )
        errors = reconstruction_error(self.model, self.X_train[passed_mask])
        alerts = select_alerts(
            errors, threshold, self.X_train_raw[passed_mask], self.y_train[passed_mask],
            self.config.boosting.alert_budget_per_client, seed=self.config.seed + 1000 * server_round + self.client_id,
        )
        logger.info(
            "Client %d round %d: %d of %d boosting-passed rows flagged; surfacing %d",
            self.client_id, server_round, int((errors > threshold).sum()), len(errors), len(alerts),
        )
        return alerts

    def evaluate(
        self, parameters: NDArrays, config: dict[str, Scalar]
    ) -> tuple[float, int, dict[str, Scalar]]:
        """Load global weights, evaluate reconstruction error on benign validation data.

        Recomputes the anomaly threshold every round (component 3 — never
        calibrated once and frozen), since the global autoencoder keeps
        changing.

        Returns:
            (mean_reconstruction_error, num_val_benign_examples, metrics),
            where metrics includes the recomputed `anomaly_threshold`.
        """
        set_weights(self.model, parameters)

        if len(self.X_val_benign) == 0:
            logger.warning("Client %d: no benign validation samples to evaluate on", self.client_id)
            return 0.0, 0, {"anomaly_threshold": float("inf"), "client_id": self.client_id}

        errors = reconstruction_error(self.model, self.X_val_benign)
        threshold = compute_anomaly_threshold(errors, self.config.autoencoder.anomaly_percentile)
        return (
            float(np.mean(errors)),
            len(self.X_val_benign),
            {"anomaly_threshold": threshold, "client_id": self.client_id},
        )


def make_client(
    client_id: int, client_data: dict[str, np.ndarray], config: Config, input_dim: int
) -> AutoencoderClient:
    """Construct an `AutoencoderClient` from a component-1-shaped client data dict."""
    return AutoencoderClient(
        client_id,
        client_data["X"],
        client_data["X_raw"],
        client_data["X_val_benign"],
        config,
        input_dim,
        y_train=client_data["y"],
    )


if __name__ == "__main__":
    import argparse

    from flwr.client import start_client

    from fl_ids.fl.data_io import load_client_data
    from fl_ids.utils.config import load_config
    from fl_ids.utils.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Run a real Flower autoencoder client process")
    parser.add_argument("--client-id", type=int, required=True)
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--data-path", required=True, help="Path to a .npz file from save_client_data")
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--max-retries", type=int, default=10)
    parser.add_argument("--max-wait-time", type=float, default=30.0)
    parser.add_argument(
        "--client-type",
        choices=["honest", "sign_flip"],
        default="honest",
        help="'sign_flip' runs the component 5 test attacker (fl_ids.robustness.attackers)",
    )
    parser.add_argument("--amplification", type=float, default=5.0, help="sign_flip attacker's delta amplification")
    parser.add_argument(
        "--claimed-examples", type=int, default=None,
        help="sign_flip attacker: example count to report instead of the true one (FedAvg weight inflation)",
    )
    args = parser.parse_args()

    setup_logging()
    run_config = load_config(args.config_path)
    data = load_client_data(args.data_path)
    input_dim = data["X"].shape[1] if data["X"].shape[0] > 0 else data["X_val_benign"].shape[1]

    if args.client_type == "sign_flip":
        from fl_ids.robustness.attackers import SignFlipAttackerClient

        client = SignFlipAttackerClient(
            args.client_id,
            data["X"],
            data["X_raw"],
            data["X_val_benign"],
            run_config,
            input_dim,
            amplification=args.amplification,
            claimed_num_examples=args.claimed_examples,
            y_train=data["y"],
        )
    else:
        client = make_client(args.client_id, data, run_config, input_dim)

    start_client(
        server_address=args.server_address,
        client=client.to_client(),
        max_retries=args.max_retries,
        max_wait_time=args.max_wait_time,
    )
