"""Phase 3 milestone: a real multi-process Flower run.

Spawns one real server process (component 8, plain FedAvg) and several
real client processes (component 4) — actual OS processes talking over a
real gRPC socket, not an in-process simulation or manual driver — and
confirms several rounds complete without errors and that reconstruction
error trends downward on held-out benign data.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from sklearn.model_selection import train_test_split

from fl_ids.data.pipeline import partition_and_normalize_clients
from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.fl.data_io import save_client_data
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import BoostingConfig, DataConfig

REPO_ROOT = Path(__file__).resolve().parents[1]

NUM_CLIENTS = 3
NUM_ROUNDS = 6
SEED = 123


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_test_config(path: Path, num_clients: int, num_rounds: int, input_dim_hidden: list[int]) -> None:
    config_dict = {
        "seed": SEED,
        "data": {
            "dnn_csv_path": "unused.csv",
            "pcap_dir": "unused",
            "num_clients": num_clients,
            "dirichlet_alpha": 0.5,
            "val_benign_fraction": 0.2,
            "test_fraction": 0.1,
            "normalize_per_client": True,
        },
        "boosting": {
            "label_source": "server_held_calibration_set",
            "calibration_fraction": 0.1,
            "confidence_threshold": 0.6,
            "num_boost_round": 50,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "broadcast_every_n_rounds": 1,
            "update_every_n_rounds": 3,
        },
        "autoencoder": {
            "bottleneck_dim": 4,
            "hidden_dims": input_dim_hidden,
            "learning_rate": 0.02,
            "local_epochs": 3,
            "batch_size": 32,
            "anomaly_percentile": 97,
            "reconstruction_error_bins": 20,
            "reconstruction_error_range": [0.0, 5.0],
        },
        "cascade": {"confidence_threshold": 0.6, "anomaly_confidence_clip": [0.0, 1.0]},
        "fl": {"num_rounds": num_rounds, "clients_per_round": num_clients, "local_epochs": 3, "strategy": "fedavg"},
        "robustness": {
            "trim_fraction": 0.15,
            "norm_clip_multiplier": 2.0,
            "mad_outlier_threshold": 3.0,
            "trust_ema_alpha": 0.3,
        },
        "sdn": {
            "block_confidence_threshold": 0.85,
            "rate_limit_confidence_threshold": 0.5,
            "controller_host": "127.0.0.1",
            "controller_port": 6653,
            "bridge_api_host": "127.0.0.1",
            "bridge_api_port": 8080,
        },
        "evaluation": {"poisoning_fractions": [0.0], "output_dir": "results"},
        "logging": {"level": "WARNING", "log_dir": "logs", "log_file": "fl_ids_test.log"},
    }
    path.write_text(yaml.safe_dump(config_dict))


@pytest.fixture(scope="module")
def fl_run_artifacts(tmp_path_factory):
    """Build the synthetic dataset, bootstrap boosting model, per-client data
    files, and config once for this module's tests.
    """
    tmp_path = tmp_path_factory.mktemp("fl_integration")

    X, y, class_names = make_synthetic_attack_dataset(n_samples=6000, n_features=12, seed=SEED)
    X_calib, X_pool, y_calib, y_pool = train_test_split(
        X, y, train_size=0.1, random_state=SEED, stratify=y
    )

    # Boosting model trained on its own standardized calibration set — see
    # fl_ids.models.boosting's module docstring for the feature-scale
    # design decision this mirrors.
    from sklearn.preprocessing import StandardScaler

    calib_scaler = StandardScaler().fit(X_calib)
    boosting_config = BoostingConfig(
        label_source="server_held_calibration_set",
        calibration_fraction=0.1,
        confidence_threshold=0.6,
        num_boost_round=50,
        learning_rate=0.1,
        num_leaves=31,
        broadcast_every_n_rounds=1,
        update_every_n_rounds=3,
    )
    boosting_model = BoostingClassifier(boosting_config, len(class_names), benign_class=0, seed=SEED)
    boosting_model.train(calib_scaler.transform(X_calib), y_calib)

    data_config = DataConfig(
        dnn_csv_path="unused.csv",
        pcap_dir="unused",
        num_clients=NUM_CLIENTS,
        dirichlet_alpha=0.5,
        val_benign_fraction=0.2,
        test_fraction=0.1,
        normalize_per_client=True,
    )
    client_data = partition_and_normalize_clients(X_pool, y_pool, benign_class=0, config=data_config, seed=SEED)

    data_paths = {}
    for cid, data in client_data.items():
        path = tmp_path / f"client_{cid}.npz"
        save_client_data(path, data)
        data_paths[cid] = path

    boosting_path = tmp_path / "boosting_model.bin"
    boosting_path.write_bytes(boosting_model.to_bytes())

    config_path = tmp_path / "config.yaml"
    _write_test_config(config_path, NUM_CLIENTS, NUM_ROUNDS, input_dim_hidden=[16, 8])

    history_path = tmp_path / "history.json"

    return {
        "data_paths": data_paths,
        "boosting_path": boosting_path,
        "config_path": config_path,
        "history_path": history_path,
        "input_dim": X.shape[1],
        "num_classes": len(class_names),
    }


def test_real_multiprocess_fl_run_completes_and_reduces_reconstruction_error(fl_run_artifacts):
    port = _free_port()
    server_address = f"127.0.0.1:{port}"
    history_path = fl_run_artifacts["history_path"]

    server_cmd = [
        sys.executable,
        "-m",
        "fl_ids.fl.server",
        "--server-address", server_address,
        "--config-path", str(fl_run_artifacts["config_path"]),
        "--boosting-model-path", str(fl_run_artifacts["boosting_path"]),
        "--num-classes", str(fl_run_artifacts["num_classes"]),
        "--benign-class", "0",
        "--input-dim", str(fl_run_artifacts["input_dim"]),
        "--min-clients", str(NUM_CLIENTS),
        "--num-rounds", str(NUM_ROUNDS),
        "--history-output", str(history_path),
    ]
    server_proc = subprocess.Popen(
        server_cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    client_procs = []
    try:
        for cid, data_path in fl_run_artifacts["data_paths"].items():
            client_cmd = [
                sys.executable,
                "-m",
                "fl_ids.fl.client",
                "--client-id", str(cid),
                "--server-address", server_address,
                "--data-path", str(data_path),
                "--config-path", str(fl_run_artifacts["config_path"]),
                "--max-retries", "15",
                "--max-wait-time", "40",
            ]
            client_procs.append(
                subprocess.Popen(
                    client_cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
            )

        try:
            server_out, _ = server_proc.communicate(timeout=180)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_out, _ = server_proc.communicate()
            pytest.fail(f"Server process did not finish in time. Output:\n{server_out}")

        assert server_proc.returncode == 0, f"Server process failed. Output:\n{server_out}"

        for i, proc in enumerate(client_procs):
            try:
                client_out, _ = proc.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                client_out, _ = proc.communicate()
            assert proc.returncode == 0, f"Client {i} process failed. Output:\n{client_out}"
    finally:
        for proc in client_procs:
            if proc.poll() is None:
                proc.kill()
        if server_proc.poll() is None:
            server_proc.kill()

    assert history_path.exists(), "Server should have written a history file on completion"
    history = json.loads(history_path.read_text())
    losses = history["losses_distributed"]

    assert len(losses) == NUM_ROUNDS, f"Expected {NUM_ROUNDS} rounds of distributed loss, got {losses}"

    round_losses = [loss for _round, loss in losses]
    assert all(loss >= 0.0 for loss in round_losses)

    # Trend check tolerant of round-to-round noise: compare the mean of the
    # last third of rounds against the first third, rather than requiring
    # strict monotonicity.
    third = max(1, NUM_ROUNDS // 3)
    early_mean = float(np.mean(round_losses[:third]))
    late_mean = float(np.mean(round_losses[-third:]))
    assert late_mean < early_mean, (
        f"Expected reconstruction error to trend downward across rounds, "
        f"got early={early_mean:.4f} late={late_mean:.4f} (losses={round_losses})"
    )
