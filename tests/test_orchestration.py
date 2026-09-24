"""Tests for component 11's Phase A orchestration and model bundle.

The Phase A test is a real run: a Flower server and client processes
over gRPC on a small synthetic CSV in the real schema, asserting on what
the saved bundle does, not just that files appeared.
"""

from __future__ import annotations

import dataclasses
import json
import socket

import numpy as np
import pytest
from sklearn.preprocessing import StandardScaler

from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.models.autoencoder import Autoencoder, compute_anomaly_threshold, get_weights, reconstruction_error
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.orchestration.artifacts import (
    AUTOENCODER_FILE,
    MANIFEST_FILE,
    load_phase_a_artifacts,
    save_phase_a_artifacts,
)
from fl_ids.orchestration.phase_a import run_phase_a
from fl_ids.utils.config import load_config
from tests.test_data_pipeline import _make_synthetic_dataframe


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def small_bundle(tmp_path):
    config = load_config()
    X, y, class_names = make_synthetic_attack_dataset(n_samples=1500, n_features=6, seed=0)
    boosting = BoostingClassifier(
        dataclasses.replace(config.boosting, num_boost_round=20), len(class_names), 0, 0,
        config.cascade.confidence_threshold,
    )
    boosting.train(X, y)
    torch_model = Autoencoder(6, [8, 4], 2)
    scalers = {cid: (X[cid::2].mean(axis=0), X[cid::2].std(axis=0) + 1) for cid in (0, 1)}
    save_phase_a_artifacts(
        tmp_path / "bundle", boosting, get_weights(torch_model), class_names, 0,
        [f"f{i}" for i in range(6)], [8, 4], 2, config.cascade, scalers, {0: 0.5, 1: float("inf")},
        extra_manifest={"seed": 0},
    )
    return tmp_path / "bundle", config, X, boosting, torch_model, scalers


def test_bundle_round_trip_reproduces_the_saved_models(small_bundle):
    bundle_dir, config, X, boosting, torch_model, scalers = small_bundle
    loaded = load_phase_a_artifacts(bundle_dir, config.boosting)

    np.testing.assert_allclose(loaded.boosting_model.predict_proba(X[:50]), boosting.predict_proba(X[:50]), atol=1e-9)
    np.testing.assert_allclose(
        reconstruction_error(loaded.autoencoder, X[:50]), reconstruction_error(torch_model, X[:50]), rtol=1e-6
    )
    assert loaded.client_ids == [0, 1]
    assert loaded.client_thresholds == {0: 0.5, 1: float("inf")}
    mean, scale = scalers[1]
    np.testing.assert_allclose(loaded.normalize(1, X[:5]), (X[:5] - mean) / scale, rtol=1e-5)
    assert loaded.cascade_config == config.cascade


def test_bundle_with_mismatched_architecture_is_rejected(small_bundle):
    bundle_dir, config, *_ = small_bundle
    manifest = json.loads((bundle_dir / MANIFEST_FILE).read_text())
    manifest["autoencoder"]["hidden_dims"] = [16, 4]
    (bundle_dir / MANIFEST_FILE).write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_phase_a_artifacts(bundle_dir, config.boosting)


def test_incomplete_bundle_is_rejected(small_bundle):
    bundle_dir, config, *_ = small_bundle
    (bundle_dir / AUTOENCODER_FILE).unlink()
    with pytest.raises(FileNotFoundError):
        load_phase_a_artifacts(bundle_dir, config.boosting)


@pytest.fixture(scope="module")
def phase_a_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("phase_a")
    csv_path = tmp_path / "synthetic_edgeiiot.csv"
    _make_synthetic_dataframe(n_per_class=400, seed=0).to_csv(csv_path, index=False)

    config = load_config()
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, num_clients=3, dirichlet_alpha=1.0),
        boosting=dataclasses.replace(config.boosting, calibration_fraction=0.3, num_boost_round=30),
        autoencoder=dataclasses.replace(config.autoencoder, local_epochs=2),
        logging=dataclasses.replace(config.logging, level="WARNING"),
        orchestration=dataclasses.replace(
            config.orchestration,
            server_address=f"127.0.0.1:{_free_port()}",
            live_state_path=str(tmp_path / "live_state.json"),
            malicious_fraction=0.0,
            fl_timeout_seconds=300,
        ),
    )
    bundle_dir = run_phase_a(
        config, csv_path, num_rounds=3, artifact_dir=tmp_path / "bundle", run_dir=tmp_path / "run"
    )
    return config, bundle_dir, tmp_path


def test_phase_a_saves_a_bundle_matching_what_the_real_run_trained(phase_a_run):
    config, bundle_dir, tmp_path = phase_a_run
    artifacts = load_phase_a_artifacts(bundle_dir, config.boosting)
    live_state = json.loads((tmp_path / "live_state.json").read_text())

    assert artifacts.client_ids == [0, 1, 2]
    assert len(live_state) == 3  # one entry per round, written by the real server
    # The saved weights are the final global model: recomputing each
    # client's threshold from them reproduces what the client reported in
    # the final round (run_phase_a checks this too, and would have raised).
    final_reported = {int(k): v for k, v in live_state[-1]["per_client_anomaly_threshold"].items()}
    for cid, threshold in artifacts.client_thresholds.items():
        assert threshold == pytest.approx(final_reported[cid], rel=1e-4)

    # Each client's scaler is the one its own training rows produce.
    client_data = np.load(tmp_path / "run" / "client_0.npz")
    expected = StandardScaler().fit(client_data["X_raw"])
    np.testing.assert_allclose(artifacts.client_scalers[0][0], expected.mean_, rtol=1e-6)
    val_errors = reconstruction_error(artifacts.autoencoder, client_data["X_val_benign"])
    assert artifacts.client_thresholds[0] == pytest.approx(
        compute_anomaly_threshold(val_errors, config.autoencoder.anomaly_percentile), rel=1e-5
    )


def test_phase_a_manifest_records_metrics_and_provenance(phase_a_run):
    config, bundle_dir, _ = phase_a_run
    manifest = json.loads((bundle_dir / MANIFEST_FILE).read_text())

    assert set(manifest["test_metrics"]) == {"boosting_only", "autoencoder_only", "cascade"}
    cascade = manifest["test_metrics"]["cascade"]
    # feature_a separates the synthetic classes cleanly, so a working
    # pipeline catches nearly every attack.
    assert cascade["attack_macro_recall"] > 0.9
    assert cascade["attack_macro_recall"] >= manifest["test_metrics"]["boosting_only"]["attack_macro_recall"]
    assert manifest["training"]["num_rounds"] == 3
    assert set(manifest["training"]["final_trust_scores"]) == {"0", "1", "2"}
    assert manifest["provenance"]["git_commit"]
    assert manifest["feature_names"] and manifest["input_dim"] == len(manifest["feature_names"])


def test_phase_a_fails_fast_when_a_client_process_dies(tmp_path):
    import time

    from fl_ids.data.pipeline import load_and_encode
    from fl_ids.orchestration.phase_a import prepare_data, run_federated_training

    csv_path = tmp_path / "synthetic_edgeiiot.csv"
    _make_synthetic_dataframe(n_per_class=200, seed=1).to_csv(csv_path, index=False)
    config = load_config()
    config = dataclasses.replace(
        config,
        data=dataclasses.replace(config.data, num_clients=3, dirichlet_alpha=1.0),
        boosting=dataclasses.replace(config.boosting, calibration_fraction=0.3, num_boost_round=10),
        orchestration=dataclasses.replace(
            config.orchestration,
            server_address=f"127.0.0.1:{_free_port()}",
            live_state_path=str(tmp_path / "live_state.json"),
            fl_timeout_seconds=300,
        ),
    )
    X, y, encoder, feature_names, benign = load_and_encode(csv_path)
    prepared = prepare_data(X, y, list(encoder.classes_), benign, feature_names, config)
    # A client whose data has the wrong width can't load the global weights.
    prepared.client_data[1]["X"] = prepared.client_data[1]["X"][:, :-1]

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="client_1 exited"):
        run_federated_training(prepared, config, tmp_path / "run", num_rounds=3, malicious_client_ids=set())
    # Well under the 300s timeout: the dead client is noticed, not waited out.
    assert time.monotonic() - started < 120
