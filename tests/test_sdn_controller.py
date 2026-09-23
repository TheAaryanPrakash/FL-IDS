"""Tests for the OpenFlow controller's mitigation logic (component 10).

Uses a fake datapath/ofproto/parser (mirroring os-ken's real interface)
rather than a real switch, so these run fast and without root/OVS — the
real end-to-end behavior (real HELLO/FEATURES handshake, real FLOW_MOD
installation, real block/rate-limit/clear against actual Open vSwitch)
was verified manually against a live OVS bridge and via the Phase 7
Mininet milestone demo (see fl_ids/sdn/topology.py), not re-asserted
here — that verification needs root and isn't practical to run in a
standard CI environment.
"""

from __future__ import annotations

from fl_ids.sdn.controller import MitigationController


class FakeOFProto:
    OFPP_NORMAL = 0
    OFPIT_APPLY_ACTIONS = 4
    OFPFC_ADD = 0
    OFPFC_DELETE = 3
    OFPP_ANY = 0xFFFFFFFF
    OFPG_ANY = 0xFFFFFFFF
    OFPMC_ADD = 0
    OFPMC_DELETE = 2
    OFPIT_METER = 6
    OFPMF_KBPS = 1


class FakeParser:
    def OFPMatch(self, **kwargs):
        return {"match": kwargs}

    def OFPActionOutput(self, port):
        return {"action": "output", "port": port}

    def OFPInstructionActions(self, type_, actions):
        return {"instruction": "actions", "type": type_, "actions": actions}

    def OFPFlowMod(self, **kwargs):
        return {"type": "flow_mod", **kwargs}

    def OFPMeterBandDrop(self, rate, burst_size):
        return {"band": "drop", "rate": rate, "burst_size": burst_size}

    def OFPMeterMod(self, **kwargs):
        return {"type": "meter_mod", **kwargs}

    def OFPInstructionMeter(self, meter_id, type_):
        return {"instruction": "meter", "meter_id": meter_id, "type": type_}


class FakeDatapath:
    def __init__(self, dp_id: int):
        self.id = dp_id
        self.ofproto = FakeOFProto()
        self.ofproto_parser = FakeParser()
        self.sent_messages: list[dict] = []

    def send_msg(self, msg):
        self.sent_messages.append(msg)


def _controller_with_fake_datapath(dp_id: int = 1) -> tuple[MitigationController, FakeDatapath]:
    controller = MitigationController()
    datapath = FakeDatapath(dp_id)
    controller.datapaths[dp_id] = datapath
    return controller, datapath


def test_block_device_installs_high_priority_drop_rule():
    controller, datapath = _controller_with_fake_datapath()

    assert controller.block_device(1, "10.0.0.5") is True

    assert len(datapath.sent_messages) == 1
    mod = datapath.sent_messages[0]
    assert mod["type"] == "flow_mod"
    assert mod["priority"] == 100
    assert mod["match"] == {"match": {"eth_type": 0x0800, "ipv4_src": "10.0.0.5"}}
    assert mod["instructions"] == []  # empty instructions = drop


def test_block_device_unknown_datapath_returns_false():
    controller = MitigationController()
    assert controller.block_device(999, "10.0.0.5") is False


def test_rate_limit_device_installs_meter_and_flow():
    controller, datapath = _controller_with_fake_datapath()

    assert controller.rate_limit_device(1, "10.0.0.7", rate_kbps=500) is True

    assert len(datapath.sent_messages) == 2
    meter_mod, flow_mod = datapath.sent_messages

    assert meter_mod["type"] == "meter_mod"
    assert meter_mod["bands"][0] == {"band": "drop", "rate": 500, "burst_size": 250}

    assert flow_mod["type"] == "flow_mod"
    assert flow_mod["priority"] == 100
    assert flow_mod["match"] == {"match": {"eth_type": 0x0800, "ipv4_src": "10.0.0.7"}}
    instruction_types = [i["instruction"] for i in flow_mod["instructions"]]
    assert instruction_types == ["meter", "actions"]


def test_rate_limit_device_reuses_same_meter_id_for_same_device():
    controller, datapath = _controller_with_fake_datapath()

    controller.rate_limit_device(1, "10.0.0.7", rate_kbps=500)
    first_meter_id = datapath.sent_messages[0]["meter_id"]

    controller.rate_limit_device(1, "10.0.0.7", rate_kbps=800)
    second_meter_id = datapath.sent_messages[2]["meter_id"]

    assert first_meter_id == second_meter_id


def test_rate_limit_device_different_devices_get_different_meter_ids():
    controller, datapath = _controller_with_fake_datapath()

    controller.rate_limit_device(1, "10.0.0.7", rate_kbps=500)
    controller.rate_limit_device(1, "10.0.0.8", rate_kbps=500)

    meter_id_a = datapath.sent_messages[0]["meter_id"]
    meter_id_b = datapath.sent_messages[2]["meter_id"]
    assert meter_id_a != meter_id_b


def test_clear_mitigation_deletes_flow_and_meter_if_present():
    controller, datapath = _controller_with_fake_datapath()
    controller.rate_limit_device(1, "10.0.0.7", rate_kbps=500)
    datapath.sent_messages.clear()

    assert controller.clear_mitigation(1, "10.0.0.7") is True

    assert len(datapath.sent_messages) == 2  # flow delete + meter delete
    flow_delete, meter_delete = datapath.sent_messages
    assert flow_delete["command"] == FakeOFProto.OFPFC_DELETE
    assert meter_delete["command"] == FakeOFProto.OFPMC_DELETE

    # The device's meter id is forgotten after clearing.
    assert (1, "10.0.0.7") not in controller._meter_ids


def test_clear_mitigation_without_prior_rate_limit_only_deletes_flow():
    controller, datapath = _controller_with_fake_datapath()

    assert controller.clear_mitigation(1, "10.0.0.5") is True
    assert len(datapath.sent_messages) == 1
    assert datapath.sent_messages[0]["command"] == FakeOFProto.OFPFC_DELETE


def test_get_datapath_ids_reflects_connected_switches():
    controller, _ = _controller_with_fake_datapath(dp_id=42)
    assert controller.get_datapath_ids() == [42]
