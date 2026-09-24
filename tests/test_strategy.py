"""Tests for the custom Flower Strategy (component 7), no networking involved.

Builds fake `FitRes` results directly (no real client processes) to
precisely control which "clients" are honest vs. sign-flip attackers and
assert on `aggregate_fit`'s exact behavior — a fast, deterministic
complement to the heavier real multi-process test in
tests/test_fl_robustness_integration.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from flwr.common import Code, FitRes, Status, ndarrays_to_parameters, parameters_to_ndarrays

from fl_ids.fl.strategy import TrustFilteredStrategy
from fl_ids.utils.config import (
    AutoencoderConfig,
    BoostingConfig,
    CascadeConfig,
    Config,
    DataConfig,
    EvaluationConfig,
    FLConfig,
    LoggingConfig,
    RobustnessConfig,
    SDNConfig,
)


class _FakeClientProxy:
    def __init__(self, cid: str):
        self.cid = cid


def _full_config() -> Config:
    return Config(
        seed=42,
        data=DataConfig(
            dnn_csv_path="unused.csv", pcap_dir="unused", num_clients=5, dirichlet_alpha=0.3,
            val_benign_fraction=0.2, test_fraction=0.2, normalize_per_client=True,
        ),
        boosting=BoostingConfig(
            label_source="server_held_calibration_set", calibration_fraction=0.05,
            num_boost_round=50, learning_rate=0.1, num_leaves=31,
            broadcast_every_n_rounds=1, update_every_n_rounds=3,
        ),
        autoencoder=AutoencoderConfig(
            bottleneck_dim=4, hidden_dims=[8], learning_rate=0.01, local_epochs=2, batch_size=32,
            anomaly_percentile=97, reconstruction_error_bins=20, reconstruction_error_range=(0.0, 5.0),
        ),
        cascade=CascadeConfig(confidence_threshold=0.7, anomaly_confidence_clip=(0.0, 1.0)),
        fl=FLConfig(num_rounds=5, clients_per_round=5, local_epochs=2, strategy="custom_trust_filtered"),
        robustness=RobustnessConfig(
            trim_fraction=0.15, norm_clip_multiplier=2.0, mad_outlier_threshold=3.0, trust_ema_alpha=0.3
        ),
        sdn=SDNConfig(
            block_confidence_threshold=0.85, rate_limit_confidence_threshold=0.5,
            controller_host="127.0.0.1", controller_port=6653,
            bridge_api_host="127.0.0.1", bridge_api_port=8080,
        ),
        evaluation=EvaluationConfig(poisoning_fractions=[0.0], output_dir="results"),
        logging=LoggingConfig(level="WARNING", log_dir="logs", log_file="test.log"),
    )


def _make_strategy(config: Config, min_clients: int = 5) -> TrustFilteredStrategy:
    return TrustFilteredStrategy(
        config,
        boosting_model_bytes=b"fake-model-bytes",
        num_classes=5,
        benign_class=0,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
    )


def _fit_res(weights: list[np.ndarray], client_id: int, num_examples: int = 100) -> FitRes:
    return FitRes(
        status=Status(code=Code.OK, message="ok"),
        parameters=ndarrays_to_parameters(weights),
        num_examples=num_examples,
        metrics={"client_id": client_id},
    )


def test_configure_fit_captures_round_start_weights_and_broadcasts_boosting_model():
    config = _full_config()
    strategy = _make_strategy(config)
    base_weights = [np.ones((3, 3)), np.zeros(3)]
    parameters = ndarrays_to_parameters(base_weights)

    class _FakeClientManager:
        def sample(self, num_clients, min_num_clients=None):
            return [_FakeClientProxy(str(i)) for i in range(num_clients)]

        def num_available(self):
            return 5

        def wait_for(self, num_clients, timeout):
            return True

    fit_ins_list = strategy.configure_fit(server_round=1, parameters=parameters, client_manager=_FakeClientManager())

    assert len(strategy._round_start_weights) == len(base_weights)
    for a, b in zip(strategy._round_start_weights, base_weights):
        assert np.allclose(a, b)

    for _client_proxy, fit_ins in fit_ins_list:
        assert fit_ins.config["boosting_model_bytes"] == b"fake-model-bytes"
        assert fit_ins.config["num_classes"] == 5
        assert fit_ins.config["benign_class"] == 0
        assert fit_ins.config["server_round"] == 1


def test_aggregate_fit_excludes_sign_flip_attacker_and_averages_honest_clients():
    config = _full_config()
    strategy = _make_strategy(config)

    base_weights = [np.zeros(50)]
    strategy._round_start_weights = base_weights

    rng = np.random.default_rng(0)
    honest_direction = rng.normal(0, 1, size=50)

    results = []
    for cid in range(5):
        honest_delta = honest_direction + rng.normal(0, 0.1, size=50)
        results.append((_FakeClientProxy(str(cid)), _fit_res([honest_delta], cid)))

    attacker_delta = -honest_direction * 5.0  # sign-flipped, amplified
    results.append((_FakeClientProxy("attacker"), _fit_res([attacker_delta], 99)))

    new_parameters, metrics = strategy.aggregate_fit(server_round=1, results=results, failures=[])

    assert 99 not in strategy.round_history[-1]["survivors"]
    assert set(range(5)).issubset(set(strategy.round_history[-1]["survivors"]))
    assert metrics["num_survivors"] == 5

    new_weights = parameters_to_ndarrays(new_parameters)[0]
    # Aggregated result should point roughly in the honest direction, not
    # be dominated by the (excluded) attacker.
    cos_to_honest = np.dot(new_weights, honest_direction) / (
        np.linalg.norm(new_weights) * np.linalg.norm(honest_direction)
    )
    assert cos_to_honest > 0.9


def test_aggregate_fit_records_trust_scores_that_separate_honest_from_attacker_over_rounds():
    config = _full_config()
    strategy = _make_strategy(config)
    base_weights = [np.zeros(50)]

    rng = np.random.default_rng(1)
    honest_direction = rng.normal(0, 1, size=50)

    for round_num in range(6):
        strategy._round_start_weights = base_weights
        results = []
        for cid in range(5):
            honest_delta = honest_direction + rng.normal(0, 0.1, size=50)
            results.append((_FakeClientProxy(str(cid)), _fit_res([honest_delta], cid)))
        attacker_delta = -honest_direction * 5.0
        results.append((_FakeClientProxy("attacker"), _fit_res([attacker_delta], 99)))

        strategy.aggregate_fit(server_round=round_num, results=results, failures=[])

    honest_trust = [strategy.trust_tracker.scores[cid] for cid in range(5)]
    attacker_trust = strategy.trust_tracker.scores[99]

    assert all(t > 0.7 for t in honest_trust)
    assert attacker_trust < 0.2
    assert min(honest_trust) - attacker_trust > 0.5

    excluded_rounds = sum(1 for entry in strategy.round_history if 99 not in entry["survivors"])
    assert excluded_rounds >= 5  # attacker excluded in most (>=5/6) rounds


def test_aggregate_fit_with_empty_results_returns_none():
    config = _full_config()
    strategy = _make_strategy(config)
    parameters, metrics = strategy.aggregate_fit(server_round=1, results=[], failures=[])
    assert parameters is None
    assert metrics == {}


def test_aggregate_evaluate_merges_reconstruction_error_into_same_round_entry():
    from flwr.common import Code, EvaluateRes, Status

    config = _full_config()
    strategy = _make_strategy(config, min_clients=2)
    strategy._round_start_weights = [np.zeros(10)]

    fit_results = [
        (_FakeClientProxy("0"), _fit_res([np.ones(10)], client_id=0)),
        (_FakeClientProxy("1"), _fit_res([np.ones(10)], client_id=1)),
    ]
    strategy.aggregate_fit(server_round=1, results=fit_results, failures=[])
    assert len(strategy.round_history) == 1
    assert "mean_reconstruction_error" not in strategy.round_history[0]

    evaluate_results = [
        (
            _FakeClientProxy("0"),
            EvaluateRes(status=Status(Code.OK, "ok"), loss=0.5, num_examples=10, metrics={"client_id": 0, "anomaly_threshold": 1.2}),
        ),
        (
            _FakeClientProxy("1"),
            EvaluateRes(status=Status(Code.OK, "ok"), loss=0.7, num_examples=10, metrics={"client_id": 1, "anomaly_threshold": 1.4}),
        ),
    ]
    strategy.aggregate_evaluate(server_round=1, results=evaluate_results, failures=[])

    assert len(strategy.round_history) == 1  # merged into the same entry, not a new one
    entry = strategy.round_history[0]
    assert entry["round"] == 1
    assert entry["mean_reconstruction_error"] == pytest.approx(0.6)
    assert entry["per_client_reconstruction_error"] == {0: 0.5, 1: 0.7}
    assert entry["per_client_anomaly_threshold"] == {0: 1.2, 1: 1.4}


def test_live_state_path_is_written_after_each_round(tmp_path):
    config = _full_config()
    state_path = tmp_path / "live_state.json"
    strategy = TrustFilteredStrategy(
        config,
        boosting_model_bytes=b"fake-model-bytes",
        num_classes=5,
        benign_class=0,
        live_state_path=state_path,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
    )
    strategy._round_start_weights = [np.zeros(10)]

    fit_results = [(_FakeClientProxy("0"), _fit_res([np.ones(10)], client_id=0))]
    strategy.aggregate_fit(server_round=1, results=fit_results, failures=[])

    assert state_path.exists()
    import json

    written = json.loads(state_path.read_text())
    assert len(written) == 1
    assert written[0]["round"] == 1


def test_round_history_records_broadcast_boosting_model_version_and_metrics():
    config = _full_config()
    metrics = {"accuracy": 0.9, "weighted_f1": 0.88, "macro_f1": 0.7, "false_positive_rate": 0.02, "per_class": {}}
    strategy = TrustFilteredStrategy(
        config,
        boosting_model_bytes=b"fake-model-bytes",
        num_classes=5,
        benign_class=0,
        boosting_metrics=metrics,
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )
    strategy._round_start_weights = [np.zeros(10)]

    fit_results = [(_FakeClientProxy("0"), _fit_res([np.ones(10)], client_id=0))]
    strategy.aggregate_fit(server_round=1, results=fit_results, failures=[])

    entry = strategy.round_history[0]
    assert entry["boosting_model_version"] == 1
    assert entry["boosting_metrics"] == metrics
