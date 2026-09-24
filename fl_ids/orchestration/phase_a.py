"""Phase A entrypoint (component 11): offline FL training, end to end, to a saved model bundle.

    python -m fl_ids.orchestration.phase_a [--num-rounds N] [--malicious-fraction F]

Steps, each the real component rather than a stand-in:

1. Load and encode the dataset (component 1), carve out the server-held
   calibration set, and train the bootstrap boosting model on it before
   any FL round (component 2's cold start).
2. Partition the rest across clients non-IID and normalize per client
   (component 1); hand each client its slice as a file.
3. Start a real Flower server (component 8, with component 7's
   trust-filtered strategy unless `fl.strategy` says `fedavg`) and one
   real client process per client (component 4) — optionally some as
   sign-flip attackers (`orchestration.malicious_fraction`). The server
   writes per-round state to `orchestration.live_state_path`, which the
   dashboard's training view polls.
4. Take the final global autoencoder, calibrate each client's anomaly
   threshold on its own benign validation slice (the same calculation the
   client did in the final round — cross-checked against the server's
   record), score the cascade on the clients' held-out test slices, and
   save the bundle Phase B loads (`fl_ids.orchestration.artifacts`).
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fl_ids.data.pipeline import load_and_encode, partition_and_normalize_clients
from fl_ids.eval.common import concat_client_test_slices, per_row_thresholds, select_malicious_clients
from fl_ids.eval.metrics import detection_metrics, evaluate_boosting_alone, flag_attacks, stage_report_summary
from fl_ids.fl.data_io import save_client_data
from fl_ids.models.autoencoder import Autoencoder, compute_anomaly_threshold, reconstruction_error, set_weights
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.orchestration.artifacts import save_phase_a_artifacts
from fl_ids.utils.config import Config, write_config_yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESS_POLL_SECONDS = 0.5


@dataclass
class PreparedData:
    """Step 1-2 output: the bootstrap boosting model and every client's data."""

    boosting_model: BoostingClassifier
    boosting_metrics: dict
    client_data: dict[int, dict[str, np.ndarray]]
    class_names: list[str]
    benign_class: int
    feature_names: list[str]


def prepare_data(
    X: np.ndarray,
    y: np.ndarray,
    class_names: list[str],
    benign_class: int,
    feature_names: list[str],
    config: Config,
) -> PreparedData:
    """Calibration split, bootstrap boosting, and the per-client partition.

    Args:
        X, y, class_names, benign_class, feature_names: `load_and_encode` output.
        config: Full project config.

    Returns:
        A `PreparedData`. `boosting_metrics` is the bootstrap model's
        held-out summary on the clients' test slices, which the dashboard
        shows next to each round.
    """
    X_calib, X_pool, y_calib, y_pool = train_test_split(
        X, y, train_size=config.boosting.calibration_fraction, random_state=config.seed, stratify=y
    )
    cap = config.orchestration.pool_subsample_size
    if len(X_pool) > cap:
        X_pool, _, y_pool, _ = train_test_split(X_pool, y_pool, train_size=cap, random_state=config.seed, stratify=y_pool)

    boosting_model = BoostingClassifier(
        config.boosting, len(class_names), benign_class, config.seed, config.cascade.confidence_threshold
    )
    boosting_model.train(X_calib, y_calib)

    client_data = partition_and_normalize_clients(X_pool, y_pool, benign_class, config.data, config.seed)
    X_test_raw, _X_test_norm, y_test, _ids = concat_client_test_slices(client_data)
    boosting_metrics = stage_report_summary(
        evaluate_boosting_alone(boosting_model, X_test_raw, y_test, class_names, benign_class)
    )
    logger.info(
        "Prepared %d calibration rows, %d federated rows across %d clients; bootstrap boosting accuracy %.3f",
        len(y_calib), len(y_pool), len(client_data), boosting_metrics["accuracy"],
    )
    return PreparedData(boosting_model, boosting_metrics, client_data, class_names, benign_class, feature_names)


def run_federated_training(
    prepared: PreparedData,
    config: Config,
    run_dir: Path,
    num_rounds: int,
    malicious_client_ids: set[int],
) -> tuple[list[np.ndarray], list[dict]]:
    """Run a real multi-process Flower training and return its final weights and per-round state.

    Args:
        prepared: `prepare_data` output.
        config: Full project config (written to `run_dir` for the child processes).
        run_dir: Working directory for data files, logs and server outputs.
        num_rounds: FL rounds.
        malicious_client_ids: Clients launched as sign-flip attackers.

    Returns:
        (final global weights, per-round state from the trust-filtered
        strategy — empty for plain FedAvg).

    Raises:
        RuntimeError: If any process fails or the run exceeds
            `orchestration.fl_timeout_seconds`.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = write_config_yaml(config, run_dir / "config.yaml")
    boosting_path = run_dir / "boosting_model.txt"
    boosting_path.write_bytes(prepared.boosting_model.to_bytes())
    metrics_path = run_dir / "boosting_metrics.json"
    metrics_path.write_text(json.dumps(prepared.boosting_metrics))
    weights_path = run_dir / "final_weights.npz"
    weights_path.unlink(missing_ok=True)
    live_state_path = Path(config.orchestration.live_state_path)
    live_state_path.unlink(missing_ok=True)  # never let the dashboard show a previous run

    input_dim = len(prepared.feature_names)
    strategy = "custom_trust_filtered" if config.fl.strategy == "custom_trust_filtered" else "fedavg"
    server_cmd = [
        sys.executable, "-m", "fl_ids.fl.server",
        "--server-address", config.orchestration.server_address,
        "--config-path", str(config_path),
        "--boosting-model-path", str(boosting_path),
        "--boosting-metrics-path", str(metrics_path),
        "--num-classes", str(len(prepared.class_names)),
        "--benign-class", str(prepared.benign_class),
        "--input-dim", str(input_dim),
        "--min-clients", str(len(prepared.client_data)),
        "--num-rounds", str(num_rounds),
        "--strategy", strategy,
        "--history-output", str(run_dir / "history.json"),
        "--final-weights-output", str(weights_path),
        "--live-state-path", str(live_state_path),
    ]

    processes: list[tuple[str, subprocess.Popen]] = []
    log_files = []

    def _launch(name: str, cmd: list[str]) -> None:
        log_file = open(run_dir / f"{name}.log", "w")
        log_files.append(log_file)
        processes.append((name, subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT)))

    try:
        _launch("server", server_cmd)
        for cid, data in prepared.client_data.items():
            data_path = run_dir / f"client_{cid}.npz"
            save_client_data(data_path, data)
            client_type = "sign_flip" if cid in malicious_client_ids else "honest"
            _launch(f"client_{cid}", [
                sys.executable, "-m", "fl_ids.fl.client",
                "--client-id", str(cid),
                "--server-address", config.orchestration.server_address,
                "--data-path", str(data_path),
                "--config-path", str(config_path),
                "--client-type", client_type,
            ])
        logger.info(
            "Started Flower server on %s and %d clients (%d sign-flip attackers) for %d rounds; logs in %s",
            config.orchestration.server_address, len(prepared.client_data), len(malicious_client_ids),
            num_rounds, run_dir,
        )

        # Poll everyone rather than waiting on one process at a time: a client
        # that crashes in round 1 leaves the server waiting for it forever,
        # which would otherwise only surface at the timeout.
        deadline = time.monotonic() + config.orchestration.fl_timeout_seconds
        while True:
            codes = {name: proc.poll() for name, proc in processes}
            failed = {name: code for name, code in codes.items() if code not in (None, 0)}
            if failed:
                name, code = next(iter(failed.items()))
                raise RuntimeError(f"{name} exited with code {code}; see {run_dir / (name + '.log')}")
            if all(code == 0 for code in codes.values()):
                break
            if time.monotonic() > deadline:
                running = [name for name, code in codes.items() if code is None]
                raise RuntimeError(
                    f"Phase A exceeded {config.orchestration.fl_timeout_seconds}s; still running: {running}"
                )
            time.sleep(PROCESS_POLL_SECONDS)
    finally:
        for _name, proc in processes:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for log_file in log_files:
            log_file.close()

    with np.load(weights_path) as npz:
        final_weights = [npz[f"arr_{i}"] for i in range(len(npz.files))]
    round_history = json.loads(live_state_path.read_text()) if live_state_path.exists() else []
    return final_weights, round_history


def calibrate_thresholds(
    autoencoder: Autoencoder,
    client_data: dict[int, dict[str, np.ndarray]],
    percentile: float,
    round_history: list[dict],
) -> dict[int, float]:
    """Each client's anomaly threshold on its own benign validation slice, cross-checked against the server's record.

    Same calculation each client's `evaluate()` did on the final global
    weights in the last round (component 3). When the trust-filtered
    strategy recorded those thresholds, a mismatch means the saved weights
    aren't the ones the clients last evaluated — worth failing loudly on.

    Args:
        autoencoder: The final global autoencoder.
        client_data: Every client's data.
        percentile: `autoencoder.anomaly_percentile`.
        round_history: The strategy's per-round state (may be empty).

    Returns:
        `{client_id: threshold}` (infinite for a client with no benign validation rows).

    Raises:
        RuntimeError: If a recomputed threshold disagrees with the server's record.
    """
    thresholds = {}
    for cid, data in client_data.items():
        if len(data["X_val_benign"]) == 0:
            thresholds[cid] = float("inf")
            continue
        thresholds[cid] = compute_anomaly_threshold(reconstruction_error(autoencoder, data["X_val_benign"]), percentile)

    recorded = round_history[-1].get("per_client_anomaly_threshold", {}) if round_history else {}
    for cid, value in recorded.items():
        ours = thresholds.get(int(cid))
        if ours is not None and np.isfinite(ours) and not np.isclose(ours, value, rtol=1e-4):
            raise RuntimeError(
                f"Client {cid}: threshold {ours:.6f} from the saved weights, but the client reported {value:.6f} "
                "in the final round -- saved weights don't match the final global model"
            )
    if recorded:
        logger.info("Per-client thresholds match the %d values clients reported in the final round", len(recorded))
    return thresholds


def evaluate_bundle(
    boosting_model: BoostingClassifier,
    autoencoder: Autoencoder,
    thresholds: dict[int, float],
    prepared: PreparedData,
    config: Config,
) -> dict:
    """Detection metrics on the clients' held-out test slices, per cascade stage and combined."""
    X_raw, X_norm, y, client_ids = concat_client_test_slices(prepared.client_data)
    row_thresholds = per_row_thresholds(thresholds, client_ids)
    metrics = {}
    for mode in ("boosting_only", "autoencoder_only", "cascade"):
        flags = flag_attacks(
            mode, boosting_model, autoencoder, row_thresholds, X_raw, X_norm, prepared.class_names, config.cascade
        )
        metrics[mode] = detection_metrics(flags, y, prepared.benign_class, prepared.class_names)
    return metrics


def _git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True).stdout
        return commit + ("-dirty" if dirty.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_phase_a(
    config: Config,
    csv_path: str | Path,
    num_rounds: int | None = None,
    artifact_dir: str | Path | None = None,
    run_dir: str | Path | None = None,
) -> Path:
    """Run Phase A end to end and return the saved bundle's directory.

    Args:
        config: Full project config (`orchestration.*` for paths and the
            attacker fraction).
        csv_path: `DNN-EdgeIIoT-dataset.csv`, or any CSV with the same schema.
        num_rounds: Overrides `fl.num_rounds`.
        artifact_dir: Overrides `orchestration.artifact_dir`.
        run_dir: Overrides `orchestration.run_dir`.

    Returns:
        The bundle directory.
    """
    num_rounds = num_rounds or config.fl.num_rounds
    artifact_dir = Path(artifact_dir or config.orchestration.artifact_dir)
    run_dir = Path(run_dir or config.orchestration.run_dir)

    X, y, label_encoder, feature_names, benign_class = load_and_encode(csv_path)
    prepared = prepare_data(X, y, list(label_encoder.classes_), benign_class, feature_names, config)

    malicious = select_malicious_clients(
        list(prepared.client_data), config.orchestration.malicious_fraction, config.seed
    )
    final_weights, round_history = run_federated_training(prepared, config, run_dir, num_rounds, malicious)

    autoencoder = Autoencoder(len(feature_names), config.autoencoder.hidden_dims, config.autoencoder.bottleneck_dim)
    set_weights(autoencoder, final_weights)
    thresholds = calibrate_thresholds(
        autoencoder, prepared.client_data, config.autoencoder.anomaly_percentile, round_history
    )
    test_metrics = evaluate_bundle(prepared.boosting_model, autoencoder, thresholds, prepared, config)

    scalers = {}
    for cid, data in prepared.client_data.items():
        if config.data.normalize_per_client and len(data["X_raw"]) > 0:
            scaler = StandardScaler().fit(data["X_raw"])
            scalers[cid] = (scaler.mean_, scaler.scale_)
        else:  # the client trained on raw features
            scalers[cid] = (np.zeros(len(feature_names)), np.ones(len(feature_names)))

    last_round = round_history[-1] if round_history else {}
    save_phase_a_artifacts(
        artifact_dir,
        prepared.boosting_model,
        final_weights,
        prepared.class_names,
        benign_class,
        feature_names,
        config.autoencoder.hidden_dims,
        config.autoencoder.bottleneck_dim,
        config.cascade,
        scalers,
        thresholds,
        extra_manifest={
            "test_metrics": test_metrics,
            "bootstrap_boosting_metrics": prepared.boosting_metrics,
            "training": {
                "num_rounds": num_rounds,
                "strategy": config.fl.strategy,
                "num_clients": len(prepared.client_data),
                "malicious_client_ids": sorted(malicious),
                "final_trust_scores": last_round.get("trust_scores", {}),
                "final_survivors": last_round.get("survivors", []),
                "final_mean_reconstruction_error": last_round.get("mean_reconstruction_error"),
            },
            "provenance": {
                "seed": config.seed,
                "dataset": str(csv_path),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "git_commit": _git_commit(),
                "run_dir": str(run_dir),
            },
            "seed": config.seed,
        },
    )
    cascade = test_metrics["cascade"]
    logger.info(
        "Phase A done: cascade attack macro-recall %.3f, benign FPR %.4f on %d clients' held-out test slices; "
        "bundle at %s",
        cascade["attack_macro_recall"], cascade["benign_fpr"], len(prepared.client_data), artifact_dir,
    )
    return artifact_dir


if __name__ == "__main__":
    import argparse

    from fl_ids.utils.config import load_config
    from fl_ids.utils.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Phase A: federated training end to end, saving the model bundle")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--real-csv-path", default=None, help="Defaults to data.dnn_csv_path")
    parser.add_argument("--num-rounds", type=int, default=None, help="Defaults to fl.num_rounds")
    parser.add_argument("--malicious-fraction", type=float, default=None,
                        help="Overrides orchestration.malicious_fraction")
    parser.add_argument("--artifact-dir", default=None, help="Overrides orchestration.artifact_dir")
    args = parser.parse_args()

    run_config = load_config(args.config_path)
    setup_logging(run_config.logging.level, run_config.logging.log_dir, run_config.logging.log_file)
    if args.malicious_fraction is not None:
        run_config.orchestration.malicious_fraction = args.malicious_fraction
    run_phase_a(
        run_config,
        args.real_csv_path or run_config.data.dnn_csv_path,
        num_rounds=args.num_rounds,
        artifact_dir=args.artifact_dir,
    )
