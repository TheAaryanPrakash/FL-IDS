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


# --- Live switch-state queries (read by the dashboard, component 13) ------

from types import SimpleNamespace  # noqa: E402

from os_ken.ofproto import ofproto_v1_3, ofproto_v1_3_parser  # noqa: E402

from fl_ids.sdn import controller as controller_module  # noqa: E402
from fl_ids.sdn.controller import describe_instructions  # noqa: E402


class StatsDatapath:
    """Answers each stats request by feeding canned reply parts straight back to the controller.

    Uses os-ken's real OpenFlow 1.3 ofproto/parser, so the stats-request
    objects and instruction classes are the real ones.
    """

    def __init__(self, controller: MitigationController, dp_id: int, replies_by_request: dict[str, list[list]]):
        self.id = dp_id
        self.ofproto = ofproto_v1_3
        self.ofproto_parser = ofproto_v1_3_parser
        self._controller = controller
        self._replies = replies_by_request
        self._next_xid = 1

    def set_xid(self, msg):
        msg.xid = self._next_xid
        self._next_xid += 1

    def send_msg(self, msg):
        parts = self._replies.get(type(msg).__name__)
        if parts is None:
            return  # never answer: exercises the timeout path
        for i, body in enumerate(parts):
            flags = ofproto_v1_3.OFPMPF_REPLY_MORE if i < len(parts) - 1 else 0
            reply = SimpleNamespace(xid=msg.xid, body=body, flags=flags, datapath=self)
            self._controller._on_stats_reply(SimpleNamespace(msg=reply))


def _flow_stat(priority, match, instructions, packets=0):
    return SimpleNamespace(
        priority=priority, match=match, instructions=instructions,
        packet_count=packets, byte_count=packets * 100, duration_sec=5,
    )


def test_describe_instructions_renders_drop_meter_and_normal():
    p = ofproto_v1_3_parser
    normal = p.OFPInstructionActions(ofproto_v1_3.OFPIT_APPLY_ACTIONS, [p.OFPActionOutput(ofproto_v1_3.OFPP_NORMAL)])
    assert describe_instructions([], ofproto_v1_3) == "drop"
    assert describe_instructions([normal], ofproto_v1_3) == "output:NORMAL"
    assert describe_instructions([p.OFPInstructionMeter(1000), normal], ofproto_v1_3) == "meter:1000,output:NORMAL"


def test_get_flow_table_accumulates_multipart_reply_and_sorts_by_priority():
    p = ofproto_v1_3_parser
    normal = p.OFPInstructionActions(ofproto_v1_3.OFPIT_APPLY_ACTIONS, [p.OFPActionOutput(ofproto_v1_3.OFPP_NORMAL)])
    default_flow = _flow_stat(0, p.OFPMatch(), [normal], packets=500)
    block_flow = _flow_stat(100, p.OFPMatch(eth_type=0x0800, ipv4_src="10.0.0.1"), [], packets=42)

    controller = MitigationController()
    # Two reply parts: the second only arrives after REPLY_MORE on the first.
    controller.datapaths[1] = StatsDatapath(controller, 1, {"OFPFlowStatsRequest": [[default_flow], [block_flow]]})

    flows = controller.get_flow_table(1)
    assert flows == [
        {
            "priority": 100, "match": {"eth_type": 0x0800, "ipv4_src": "10.0.0.1"}, "actions": "drop",
            "packet_count": 42, "byte_count": 4200, "duration_sec": 5,
        },
        {
            "priority": 0, "match": {}, "actions": "output:NORMAL",
            "packet_count": 500, "byte_count": 50000, "duration_sec": 5,
        },
    ]
    assert controller._pending_stats == {}


def test_get_port_status_joins_description_with_counters_and_skips_local_port():
    up = SimpleNamespace(port_no=1, name=b"s1-eth1", state=0, config=0)
    down = SimpleNamespace(port_no=2, name=b"s1-eth2", state=ofproto_v1_3.OFPPS_LINK_DOWN, config=0)
    local = SimpleNamespace(port_no=ofproto_v1_3.OFPP_LOCAL, name=b"s1", state=0, config=0)
    counters = SimpleNamespace(
        port_no=1, rx_packets=10, tx_packets=20, rx_bytes=1000, tx_bytes=2000, rx_dropped=1, tx_dropped=0
    )

    controller = MitigationController()
    controller.datapaths[1] = StatsDatapath(
        controller, 1,
        {"OFPPortDescStatsRequest": [[up, down, local]], "OFPPortStatsRequest": [[counters]]},
    )

    ports = controller.get_port_status(1)
    assert [p["name"] for p in ports] == ["s1-eth1", "s1-eth2"]
    assert ports[0]["link_up"] is True and ports[0]["rx_packets"] == 10
    assert ports[1]["link_up"] is False and ports[1]["rx_packets"] == 0


def test_get_meter_stats_sums_band_drops():
    meter = SimpleNamespace(
        meter_id=1000, flow_count=1, packet_in_count=300, byte_in_count=30000,
        band_stats=[SimpleNamespace(packet_band_count=120)],
    )
    controller = MitigationController()
    controller.datapaths[1] = StatsDatapath(controller, 1, {"OFPMeterStatsRequest": [[meter]]})

    assert controller.get_meter_stats(1) == [
        {"meter_id": 1000, "flow_count": 1, "packet_in_count": 300, "byte_in_count": 30000, "packets_dropped_by_band": 120}
    ]


class _NeverSetEvent:
    """Stands in for `hub.Event` on the timeout path: eventlet's Timeout only
    fires inside a monkey-patched greenthread (as in the real bridge), not
    in a plain pytest process, so the real Event would block forever here.
    """

    def set(self):
        pass

    def wait(self, timeout=None):
        return False


def test_stats_query_returns_none_when_switch_never_answers(monkeypatch):
    monkeypatch.setattr(controller_module.hub, "Event", _NeverSetEvent)
    controller = MitigationController()
    controller.datapaths[1] = StatsDatapath(controller, 1, {})

    assert controller.get_flow_table(1) is None
    assert controller._pending_stats == {}


def test_stats_queries_return_none_for_unknown_datapath():
    controller = MitigationController()
    assert controller.get_flow_table(7) is None
    assert controller.get_meter_stats(7) is None
    assert controller.get_port_status(7) is None
