"""Tests for Phase B's live classification and mitigation check (no root or Mininet needed)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.models.autoencoder import Autoencoder, get_weights
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.orchestration.artifacts import load_phase_a_artifacts, save_phase_a_artifacts
from fl_ids.orchestration.phase_b import check_mitigation, flow_rule_for_ip
from fl_ids.sdn.live_inference import classify_device_traffic, client_id_for_host
from fl_ids.utils.config import load_config

FLOW_TABLE = """ cookie=0x0, duration=3.1s, table=0, n_packets=0, n_bytes=0, priority=100,ip,nw_src=10.0.0.10 actions=drop
 cookie=0x0, duration=9.8s, table=0, n_packets=24000, n_bytes=1, priority=0 actions=NORMAL"""


@pytest.fixture(scope="module")
def bundle_and_data(tmp_path_factory):
    config = load_config()
    X, y, class_names = make_synthetic_attack_dataset(n_samples=3000, n_features=6, seed=0)
    boosting = BoostingClassifier(
        dataclasses.replace(config.boosting, num_boost_round=50), len(class_names), 0, 0,
        config.cascade.confidence_threshold,
    )
    boosting.train(X, y)
    identity = (np.zeros(6), np.ones(6))
    # Client 0's threshold of 0 makes every fallthrough row anomalous; client
    # 1's infinite threshold (no benign validation data) makes none of them.
    bundle_dir = save_phase_a_artifacts(
        tmp_path_factory.mktemp("bundle"), boosting, get_weights(Autoencoder(6, [8, 4], 2)), class_names, 0,
        [f"f{i}" for i in range(6)], [8, 4], 2, config.cascade, {0: identity, 1: identity},
        {0: 0.0, 1: float("inf")}, None,
    )
    return load_phase_a_artifacts(bundle_dir, config.boosting), X, y, boosting


def test_host_maps_to_client():
    assert client_id_for_host("h1") == 0
    assert client_id_for_host("h10") == 9
    for bad in ("h0", "s1", "host1", "h"):
        with pytest.raises(ValueError):
            client_id_for_host(bad)


def test_device_is_scored_with_its_own_clients_threshold(bundle_and_data):
    artifacts, X, y, boosting = bundle_and_data
    # Rows boosting doesn't call a confident attack fall through to the autoencoder.
    fallthrough = X[boosting.passes_to_autoencoder(X)][:40]

    strict = classify_device_traffic(artifacts, "h1", fallthrough)
    lenient = classify_device_traffic(artifacts, "h2", fallthrough)

    assert (strict.client_id, lenient.client_id) == (0, 1)
    assert strict.classification == "anomalous" and strict.stage == "autoencoder"
    assert lenient.classification == "benign" and lenient.num_non_benign == 0


def test_confident_boosting_call_decides_regardless_of_threshold(bundle_and_data):
    artifacts, X, y, boosting = bundle_and_data
    stage1 = boosting.predict_cascade_stage1(X)
    attack_class = int(np.bincount(stage1.predicted_class[stage1.is_confident_attack]).argmax())
    rows = X[stage1.is_confident_attack & (stage1.predicted_class == attack_class)][:30]

    verdict = classify_device_traffic(artifacts, "h2", rows)

    assert verdict.classification == artifacts.class_names[attack_class]
    assert verdict.stage == "boosting"
    assert verdict.confidence >= artifacts.cascade_config.confidence_threshold
    assert verdict.num_non_benign == len(rows)


def test_host_without_a_client_in_the_bundle_is_rejected(bundle_and_data):
    artifacts, X, *_ = bundle_and_data
    with pytest.raises(ValueError, match="only has clients"):
        classify_device_traffic(artifacts, "h3", X[:5])
    with pytest.raises(ValueError):
        classify_device_traffic(artifacts, "h1", X[:0])


def test_flow_rule_lookup_matches_the_exact_ip():
    assert flow_rule_for_ip(FLOW_TABLE, "10.0.0.1") is None
    assert "actions=drop" in flow_rule_for_ip(FLOW_TABLE, "10.0.0.10")


@pytest.mark.parametrize(
    ("action", "ip", "consistent"),
    [("block", "10.0.0.10", True), ("block", "10.0.0.1", False), ("allow", "10.0.0.1", True),
     ("allow", "10.0.0.10", False), ("rate_limit", "10.0.0.10", True)],
)
def test_check_mitigation_requires_the_flow_table_to_match_the_decision(action, ip, consistent):
    result = {"mitigation": {"action": action}, "device_ip": ip, "flow_table": FLOW_TABLE}
    assert check_mitigation(result)["consistent"] is consistent
