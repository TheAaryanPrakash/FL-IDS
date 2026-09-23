"""Tests for the SDN mitigation bridge's decision logic (component 9).

Only exercises `classify_mitigation_action` and `DeviceRegistry`, which
don't touch eventlet/os-ken — safe to run in the normal test process.
The real controller integration is tested separately (a live-OVS
integration test, run manually/via the SDN milestone check since it
needs root).
"""

from __future__ import annotations

from fl_ids.models.cascade import ANOMALOUS_LABEL, BENIGN_LABEL
from fl_ids.sdn.bridge import DeviceRegistry, classify_mitigation_action
from fl_ids.utils.config import SDNConfig


def _sdn_config(**overrides) -> SDNConfig:
    defaults = dict(
        block_confidence_threshold=0.85,
        rate_limit_confidence_threshold=0.5,
        controller_host="127.0.0.1",
        controller_port=6653,
        bridge_api_host="127.0.0.1",
        bridge_api_port=8080,
    )
    defaults.update(overrides)
    return SDNConfig(**defaults)


def test_benign_never_mitigated_regardless_of_confidence():
    config = _sdn_config()
    assert classify_mitigation_action(BENIGN_LABEL, 0.99, config) == "allow"
    assert classify_mitigation_action(BENIGN_LABEL, 0.0, config) == "allow"


def test_high_confidence_attack_blocks():
    config = _sdn_config()
    assert classify_mitigation_action("DDoS_UDP", 0.9, config) == "block"
    assert classify_mitigation_action(ANOMALOUS_LABEL, 0.85, config) == "block"


def test_medium_confidence_attack_rate_limits():
    config = _sdn_config()
    assert classify_mitigation_action("DDoS_UDP", 0.6, config) == "rate_limit"
    assert classify_mitigation_action(ANOMALOUS_LABEL, 0.5, config) == "rate_limit"


def test_low_confidence_attack_allows():
    config = _sdn_config()
    assert classify_mitigation_action("DDoS_UDP", 0.3, config) == "allow"
    assert classify_mitigation_action(ANOMALOUS_LABEL, 0.0, config) == "allow"


def test_thresholds_are_configurable_not_hardcoded():
    strict_config = _sdn_config(block_confidence_threshold=0.5, rate_limit_confidence_threshold=0.2)
    assert classify_mitigation_action("Backdoor", 0.6, strict_config) == "block"
    assert classify_mitigation_action("Backdoor", 0.3, strict_config) == "rate_limit"


def test_device_registry_register_and_lookup():
    registry = DeviceRegistry()
    registry.register("device-1", datapath_id=42, ip_address="10.0.0.5")

    assert registry.lookup("device-1") == (42, "10.0.0.5")
    assert registry.lookup("unknown-device") is None


def test_device_registry_as_dict():
    registry = DeviceRegistry()
    registry.register("device-1", datapath_id=42, ip_address="10.0.0.5")
    registry.register("device-2", datapath_id=42, ip_address="10.0.0.6")

    as_dict = registry.as_dict()
    assert as_dict == {
        "device-1": {"datapath_id": 42, "ip_address": "10.0.0.5"},
        "device-2": {"datapath_id": 42, "ip_address": "10.0.0.6"},
    }
