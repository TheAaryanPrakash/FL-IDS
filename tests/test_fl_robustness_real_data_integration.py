"""Phase 5: re-validates Phase 4's robustness-layer milestone on real data.

Same structure as tests/test_fl_robustness_integration.py (real separate
OS processes, TrustFilteredStrategy, a mix of honest and sign-flip-
attacking clients) but sourcing client data from the real Edge-IIoTset
pipeline instead of synthetic data. The federated pool is subsampled
after the calibration split purely to keep runtime bounded — still real
traffic/features/skew. Skipped (not failed) if the dataset file isn't
present.

Client count: 8 honest + 2 sign-flip attackers (20% malicious), inside
CLAUDE.md's 8-10-client range for Phases 2-4. An earlier 4 honest + 1
attacker version passed only because the bootstrap boosting filter was
degenerate (see tests/real_data_boosting.py); with a working filter, 5
clients' worth of non-IID similarity spread was wide enough to hide the
attacker from the MAD check, and a 15% trim of 5 clients trims nobody.

Attackers hold ordinary Dirichlet shards, drawn in the same partition as
the honest clients. An earlier version gave each attacker the *entire*
pool: ~7x more training rows than any honest client, IID where they're
skewed, normalized over mixed traffic. That made the attackers' true
(pre-flip) deltas only ~0.2 cosine-aligned with the honest clients', so
negating them landed at ~0 -- inside honest non-IID spread, where the MAD
check can't see them. That tested "unusual data", which component 5
explicitly doesn't claim to catch, rather than the corrupted update it's
built for.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from sklearn.model_selection import train_test_split

from fl_ids.data.pipeline import load_and_encode, partition_and_normalize_clients
from fl_ids.fl.data_io import save_client_data
from fl_ids.utils.config import DataConfig
from tests.real_data_boosting import (
    assert_boosting_filter_is_real,
    boosting_config_section,
    cascade_config_section,
    train_bootstrap_boosting,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_DATASET_PATH = Path("data/raw/DNN-EdgeIIoT-dataset.csv")

NUM_HONEST_CLIENTS = 8
ATTACKER_CLIENT_IDS = (98, 99)
NUM_ROUNDS = 8
SEED = 777
POOL_SUBSAMPLE_SIZE = 40_000


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _write_test_config(path: Path, num_clients: int, num_rounds: int) -> None:
    config_dict = {
        "seed": SEED,
        "data": {
            "dnn_csv_path": "unused.csv",
            "pcap_dir": "unused",
            "num_clients": num_clients,
            "dirichlet_alpha": 0.3,
            "val_benign_fraction": 0.2,
            "test_fraction": 0.1,
            "normalize_per_client": True,
        },
        "boosting": boosting_config_section(),
        "autoencoder": {
            "bottleneck_dim": 8,
            "hidden_dims": [32, 16],
            "learning_rate": 0.01,
            "local_epochs": 3,
            "batch_size": 64,
            "anomaly_percentile": 97,
            "reconstruction_error_bins": 20,
            "reconstruction_error_range": [0.0, 5.0],
        },
        "cascade": cascade_config_section(),
        "fl": {
            "num_rounds": num_rounds,
            "clients_per_round": num_clients,
            "local_epochs": 3,
            "strategy": "custom_trust_filtered",
        },
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
def real_robustness_run_artifacts(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("fl_robustness_real_data_integration")

    X, y, label_encoder, feature_names, benign_class = load_and_encode(REAL_DATASET_PATH)
    X_calib, X_pool, y_calib, y_pool = train_test_split(X, y, train_size=0.05, random_state=SEED, stratify=y)

    if len(X_pool) > POOL_SUBSAMPLE_SIZE:
        X_pool, _, y_pool, _ = train_test_split(
            X_pool, y_pool, train_size=POOL_SUBSAMPLE_SIZE, random_state=SEED, stratify=y_pool
        )

    # Boosting always trains on raw features -- see fl_ids.models.boosting's
    # module docstring. Hyperparameters come from configs/config.yaml (see
    # tests/real_data_boosting.py for why they must not be hard-coded here).
    boosting_model = train_bootstrap_boosting(
        X_calib, y_calib, num_classes=len(label_encoder.classes_), benign_class=benign_class, seed=SEED
    )

    data_config = DataConfig(
        dnn_csv_path=str(REAL_DATASET_PATH),
        pcap_dir="unused",
        num_clients=NUM_HONEST_CLIENTS + len(ATTACKER_CLIENT_IDS),
        dirichlet_alpha=0.3,
        val_benign_fraction=0.2,
        test_fraction=0.1,
        normalize_per_client=True,
    )
    shards = partition_and_normalize_clients(X_pool, y_pool, benign_class, config=data_config, seed=SEED)

    # One Dirichlet partition for everyone; the last shards go to the
    # attackers, so their data is as ordinary as any honest client's --
    # the attack is in the update they send, not their data (component 5).
    client_ids = list(range(NUM_HONEST_CLIENTS)) + list(ATTACKER_CLIENT_IDS)
    data_paths = {}
    for cid, data in zip(client_ids, shards.values()):
        path = tmp_path / f"client_{cid}.npz"
        save_client_data(path, data)
        data_paths[cid] = path

    boosting_path = tmp_path / "boosting_model.bin"
    boosting_path.write_bytes(boosting_model.to_bytes())

    config_path = tmp_path / "config.yaml"
    _write_test_config(config_path, NUM_HONEST_CLIENTS + len(ATTACKER_CLIENT_IDS), NUM_ROUNDS)

    history_path = tmp_path / "history.json"

    return {
        "data_paths": data_paths,
        "boosting_path": boosting_path,
        "config_path": config_path,
        "history_path": history_path,
        "input_dim": X.shape[1],
        "num_classes": len(label_encoder.classes_),
        "benign_class": benign_class,
        "boosting_model": boosting_model,
        "X_pool": X_pool,
        "y_pool": y_pool,
    }


@pytest.mark.skipif(not REAL_DATASET_PATH.exists(), reason="real dataset not present")
def test_real_data_bootstrap_boosting_filter_actually_filters(real_robustness_run_artifacts):
    """The bootstrap filter clients run must actually filter, or Phase 4's milestone isn't the real pipeline's."""
    artifacts = real_robustness_run_artifacts
    assert_boosting_filter_is_real(
        artifacts["boosting_model"], artifacts["X_pool"], artifacts["y_pool"], artifacts["benign_class"]
    )


@pytest.mark.skipif(not REAL_DATASET_PATH.exists(), reason="real dataset not present")
def test_real_multiprocess_run_on_real_data_separates_honest_and_attacker_trust(real_robustness_run_artifacts):
    port = _free_port()
    server_address = f"127.0.0.1:{port}"
    history_path = real_robustness_run_artifacts["history_path"]
    num_clients = NUM_HONEST_CLIENTS + len(ATTACKER_CLIENT_IDS)

    server_cmd = [
        sys.executable, "-m", "fl_ids.fl.server",
        "--server-address", server_address,
        "--config-path", str(real_robustness_run_artifacts["config_path"]),
        "--boosting-model-path", str(real_robustness_run_artifacts["boosting_path"]),
        "--num-classes", str(real_robustness_run_artifacts["num_classes"]),
        "--benign-class", str(real_robustness_run_artifacts["benign_class"]),
        "--input-dim", str(real_robustness_run_artifacts["input_dim"]),
        "--min-clients", str(num_clients),
        "--num-rounds", str(NUM_ROUNDS),
        "--strategy", "custom_trust_filtered",
        "--history-output", str(history_path),
    ]
    server_proc = subprocess.Popen(
        server_cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )

    client_procs = []
    try:
        for cid, data_path in real_robustness_run_artifacts["data_paths"].items():
            client_type = "sign_flip" if cid in ATTACKER_CLIENT_IDS else "honest"
            client_cmd = [
                sys.executable, "-m", "fl_ids.fl.client",
                "--client-id", str(cid),
                "--server-address", server_address,
                "--data-path", str(data_path),
                "--config-path", str(real_robustness_run_artifacts["config_path"]),
                "--client-type", client_type,
                "--max-retries", "15",
                "--max-wait-time", "40",
            ]
            client_procs.append(
                subprocess.Popen(
                    client_cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
                )
            )

        try:
            server_out, _ = server_proc.communicate(timeout=240)
        except subprocess.TimeoutExpired:
            server_proc.kill()
            server_out, _ = server_proc.communicate()
            pytest.fail(f"Server process did not finish in time. Output:\n{server_out}")

        assert server_proc.returncode == 0, f"Server process failed. Output:\n{server_out}"

        for i, proc in enumerate(client_procs):
            try:
                client_out, _ = proc.communicate(timeout=60)
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

    assert history_path.exists()
    history = json.loads(history_path.read_text())
    round_history = history["round_history"]
    assert len(round_history) == NUM_ROUNDS

    print("Per-round trust scores:")
    for entry in round_history:
        print(f"  round {entry['round']}: {entry['trust_scores']}")

    half = NUM_ROUNDS // 2
    honest_trajectories = {
        cid: [entry["trust_scores"][str(cid)] for entry in round_history] for cid in range(NUM_HONEST_CLIENTS)
    }
    honest_late_avgs = {cid: sum(traj[half:]) / len(traj[half:]) for cid, traj in honest_trajectories.items()}
    assert all(avg > 0.55 for avg in honest_late_avgs.values()), (
        f"Honest clients' trust should stay high on real data, got late-round averages={honest_late_avgs}"
    )

    for attacker_id in ATTACKER_CLIENT_IDS:
        attacker_key = str(attacker_id)
        excluded_rounds = sum(1 for entry in round_history if entry["is_outlier"].get(attacker_key, False))
        print(f"Real-data: attacker {attacker_id} excluded in {excluded_rounds}/{NUM_ROUNDS} rounds")
        assert excluded_rounds >= (NUM_ROUNDS * 0.6), (
            f"Attacker {attacker_id} should be excluded from aggregation in most rounds on real data, "
            f"got {excluded_rounds}/{NUM_ROUNDS}. Round history: {round_history}"
        )

        attacker_trajectory = [entry["trust_scores"][attacker_key] for entry in round_history]
        print(f"Attacker {attacker_id} trust trajectory: {attacker_trajectory}")
        attacker_late_avg = sum(attacker_trajectory[half:]) / len(attacker_trajectory[half:])
        assert min(honest_late_avgs.values()) - attacker_late_avg > 0.08, (
            f"Expected a clear separation between honest and attacker {attacker_id} trust on real data, "
            f"got honest={honest_late_avgs}, attacker={attacker_late_avg}"
        )

        attacker_early_avg = sum(attacker_trajectory[:half]) / len(attacker_trajectory[:half])
        assert attacker_late_avg < attacker_early_avg * 0.9, (
            f"Attacker {attacker_id}'s trust should visibly decay over rounds on real data, "
            f"got early_avg={attacker_early_avg:.3f} late_avg={attacker_late_avg:.3f} "
            f"trajectory={attacker_trajectory}"
        )
