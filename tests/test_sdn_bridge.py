"""Tests for the SDN mitigation bridge's decision logic (component 9).

Only exercises `classify_mitigation_action` and `DeviceRegistry`, which
don't touch eventlet/os-ken — safe to run in the normal test process.
The real controller integration is tested separately (a live-OVS
integration test, run manually/via the SDN milestone check since it
needs root).
"""

from __future__ import annotations

from fl_ids.models.cascade import ANOMALOUS_LABEL, BENIGN_LABEL
from fl_ids.sdn.bridge import DeviceRegistry, classify_mitigation_action, create_bridge_app
from fl_ids.utils.config import SDNConfig


class _FakeController:
    """Mimics MitigationController's public API without needing os-ken/OVS."""

    def __init__(self):
        self.calls: list[tuple] = []

    def block_device(self, datapath_id, ip_address):
        self.calls.append(("block", datapath_id, ip_address))
        return True

    def rate_limit_device(self, datapath_id, ip_address, rate_kbps):
        self.calls.append(("rate_limit", datapath_id, ip_address, rate_kbps))
        return True

    def clear_mitigation(self, datapath_id, ip_address):
        self.calls.append(("clear", datapath_id, ip_address))
        return True

    def get_datapath_ids(self):
        return [1]

    def get_flow_table(self, datapath_id):
        return [{"priority": 100, "match": {"ipv4_src": "10.0.0.1"}, "actions": "drop"}]

    def get_meter_stats(self, datapath_id):
        return []

    def get_port_status(self, datapath_id):
        return [{"port_no": 1, "name": "s1-eth1", "link_up": True}]


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
        "device-1": {"datapath_id": 42, "ip_address": "10.0.0.5", "port_name": None},
        "device-2": {"datapath_id": 42, "ip_address": "10.0.0.6", "port_name": None},
    }


def test_mitigate_endpoint_blocks_and_records_history():
    controller = _FakeController()
    registry = DeviceRegistry()
    registry.register("h1", datapath_id=1, ip_address="10.0.0.1")
    app = create_bridge_app(controller, registry, _sdn_config())
    client = app.test_client()

    response = client.post("/mitigate", json={"device_id": "h1", "classification": "DDoS_HTTP", "confidence": 0.99})
    assert response.status_code == 200
    body = response.get_json()
    assert body["action"] == "block"
    assert body["applied"] is True
    assert controller.calls == [("block", 1, "10.0.0.1")]

    history = client.get("/mitigation_history").get_json()
    assert len(history) == 1
    assert history[0]["device_id"] == "h1"


def test_mitigate_endpoint_unknown_device_returns_404():
    controller = _FakeController()
    registry = DeviceRegistry()
    app = create_bridge_app(controller, registry, _sdn_config())
    client = app.test_client()

    response = client.post("/mitigate", json={"device_id": "unknown", "classification": "Backdoor", "confidence": 0.9})
    assert response.status_code == 404


def test_health_endpoint_reports_connected_datapaths():
    controller = _FakeController()
    registry = DeviceRegistry()
    app = create_bridge_app(controller, registry, _sdn_config())
    client = app.test_client()

    response = client.get("/health").get_json()
    assert response == {"status": "ok", "connected_datapaths": [1]}


def test_register_and_list_devices_endpoints():
    controller = _FakeController()
    registry = DeviceRegistry()
    app = create_bridge_app(controller, registry, _sdn_config())
    client = app.test_client()

    client.post(
        "/devices/register",
        json={"device_id": "h1", "datapath_id": 1, "ip_address": "10.0.0.1", "port_name": "s1-eth1"},
    )
    devices = client.get("/devices").get_json()
    assert devices == {"h1": {"datapath_id": 1, "ip_address": "10.0.0.1", "port_name": "s1-eth1"}}


def test_mitigate_records_cascade_stage_and_timestamp():
    controller = _FakeController()
    registry = DeviceRegistry()
    registry.register("h1", datapath_id=1, ip_address="10.0.0.1")
    app = create_bridge_app(controller, registry, _sdn_config())
    client = app.test_client()

    client.post(
        "/mitigate",
        json={"device_id": "h1", "classification": ANOMALOUS_LABEL, "confidence": 0.6, "stage": "autoencoder"},
    )
    entry = client.get("/mitigation_history").get_json()[0]
    assert entry["stage"] == "autoencoder"
    assert entry["action"] == "rate_limit"
    assert isinstance(entry["timestamp"], float)


def test_mitigate_rejects_unknown_stage():
    controller = _FakeController()
    registry = DeviceRegistry()
    registry.register("h1", datapath_id=1, ip_address="10.0.0.1")
    app = create_bridge_app(controller, registry, _sdn_config())
    client = app.test_client()

    response = client.post(
        "/mitigate", json={"device_id": "h1", "classification": "Backdoor", "confidence": 0.9, "stage": "magic"}
    )
    assert response.status_code == 400
    assert controller.calls == []


def test_switch_state_endpoint_reports_controller_queries_per_datapath():
    controller = _FakeController()
    app = create_bridge_app(controller, DeviceRegistry(), _sdn_config())
    state = app.test_client().get("/switch_state").get_json()

    assert state == {
        "1": {
            "flows": [{"priority": 100, "match": {"ipv4_src": "10.0.0.1"}, "actions": "drop"}],
            "meters": [],
            "ports": [{"port_no": 1, "name": "s1-eth1", "link_up": True}],
        }
    }
