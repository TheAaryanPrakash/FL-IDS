"""Custom OpenFlow 1.3 controller (component 10).

CLAUDE.md's tech stack section calls for "an existing custom OpenFlow 1.3
controller (provided separately... it was built as a substitute after the
Ryu project turned out to be broken/unmaintained, so don't default to
installing Ryu)". No existing controller was available to provide (see
project history) — the user explicitly asked for a comprehensive one to
be built from scratch instead. This still respects the "don't default to
Ryu" instruction: it's built on `os-ken`, the actively-maintained
OpenStack fork of Ryu created for exactly the reason CLAUDE.md cites (Ryu
going unmaintained), API-compatible with Ryu but not the abandoned
project itself. Verified against real Open vSwitch in this environment —
real HELLO/FEATURES handshake, real FLOW_MOD installation confirmed via
`ovs-ofctl dump-flows` — not just a library that imports cleanly.

Provides the mitigation primitives component 9 (the SDN bridge) needs:
block a device outright, rate-limit it via an OpenFlow meter, or clear
any mitigation and let its traffic flow normally.
"""

from __future__ import annotations

import logging
import threading

from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.lib import hub
from os_ken.ofproto import ofproto_v1_3

logger = logging.getLogger(__name__)

# Priorities: higher wins. Default NORMAL is the fallback; block/rate-limit
# rules must outrank it so they actually take effect for matched traffic.
PRIORITY_DEFAULT = 0
PRIORITY_MITIGATION = 100

METER_ID_BASE = 1000  # meter IDs are allocated per-device starting here

STATS_REPLY_TIMEOUT_S = 3.0  # how long a stats query waits for the switch to answer


def describe_instructions(instructions: list, ofproto) -> str:
    """Render a flow entry's OpenFlow 1.3 instructions as a short human-readable string.

    An empty instruction list is OpenFlow's drop, so it renders as `"drop"`
    -- exactly what a block mitigation installs.
    """
    parts = []
    for inst in instructions:
        if hasattr(inst, "meter_id"):
            parts.append(f"meter:{inst.meter_id}")
        for action in getattr(inst, "actions", None) or []:
            port = getattr(action, "port", None)
            if port == ofproto.OFPP_NORMAL:
                parts.append("output:NORMAL")
            elif port is not None:
                parts.append(f"output:{port}")
            else:
                parts.append(type(action).__name__)
    return ",".join(parts) if parts else "drop"


class MitigationController(app_manager.OSKenApp):
    """OpenFlow 1.3 controller: default-allow, with per-device block/rate-limit
    rules installable on demand by the SDN mitigation bridge (component 9).
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.datapaths: dict[int, object] = {}
        self._meter_ids: dict[tuple[int, str], int] = {}
        self._next_meter_id = METER_ID_BASE
        self._lock = threading.Lock()
        # xid -> (event set on the final reply part, accumulated reply bodies).
        self._pending_stats: dict[int, tuple[hub.Event, list]] = {}

    # --- Switch connection lifecycle -------------------------------------

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def _on_switch_features(self, ev) -> None:
        """On a switch connecting: register it and install the default allow-all rule."""
        datapath = ev.msg.datapath
        with self._lock:
            self.datapaths[datapath.id] = datapath
        logger.info("Switch connected: datapath_id=%s", datapath.id)
        self._install_default_flow(datapath)

    def _install_default_flow(self, datapath) -> None:
        """Priority-0 catch-all: behave as an ordinary learning switch (OFPP_NORMAL)."""
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
        instructions = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath, priority=PRIORITY_DEFAULT, match=match, instructions=instructions
        )
        datapath.send_msg(mod)
        logger.info("Installed default NORMAL flow on datapath_id=%s", datapath.id)

    # --- Mitigation API (called by the SDN bridge, component 9) ----------

    def get_datapath_ids(self) -> list[int]:
        """Currently connected switch datapath IDs."""
        with self._lock:
            return list(self.datapaths.keys())

    def block_device(self, datapath_id: int, device_ip: str) -> bool:
        """Install a high-priority drop rule for all IPv4 traffic from `device_ip`.

        Args:
            datapath_id: Target switch.
            device_ip: The malicious device's IPv4 address.

        Returns:
            True if the rule was sent, False if the datapath isn't connected.
        """
        datapath = self._get_datapath(datapath_id)
        if datapath is None:
            return False
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=device_ip)
        # Empty instruction list = drop (no actions to forward the packet).
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=PRIORITY_MITIGATION,
            match=match,
            instructions=[],
            command=ofproto.OFPFC_ADD,
        )
        datapath.send_msg(mod)
        logger.info("Installed BLOCK rule: datapath_id=%s device_ip=%s", datapath_id, device_ip)
        return True

    def rate_limit_device(self, datapath_id: int, device_ip: str, rate_kbps: int) -> bool:
        """Install an OpenFlow meter + flow so `device_ip`'s traffic is rate-limited, not dropped.

        Args:
            datapath_id: Target switch.
            device_ip: The suspicious device's IPv4 address.
            rate_kbps: Meter rate limit, in kbps.

        Returns:
            True if the rules were sent, False if the datapath isn't connected.
        """
        datapath = self._get_datapath(datapath_id)
        if datapath is None:
            return False
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        meter_id = self._allocate_meter_id(datapath_id, device_ip)
        band = parser.OFPMeterBandDrop(rate=rate_kbps, burst_size=rate_kbps // 2 or 1)
        meter_mod = parser.OFPMeterMod(
            datapath=datapath,
            command=ofproto.OFPMC_ADD,
            flags=ofproto.OFPMF_KBPS,
            meter_id=meter_id,
            bands=[band],
        )
        datapath.send_msg(meter_mod)

        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=device_ip)
        instructions = [
            parser.OFPInstructionMeter(meter_id, ofproto.OFPIT_METER),
            parser.OFPInstructionActions(
                ofproto.OFPIT_APPLY_ACTIONS, [parser.OFPActionOutput(ofproto.OFPP_NORMAL)]
            ),
        ]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=PRIORITY_MITIGATION,
            match=match,
            instructions=instructions,
            command=ofproto.OFPFC_ADD,
        )
        datapath.send_msg(mod)
        logger.info(
            "Installed RATE-LIMIT rule: datapath_id=%s device_ip=%s rate_kbps=%d meter_id=%d",
            datapath_id, device_ip, rate_kbps, meter_id,
        )
        return True

    def clear_mitigation(self, datapath_id: int, device_ip: str) -> bool:
        """Remove any block/rate-limit flow for `device_ip`, reverting to default-allow.

        Args:
            datapath_id: Target switch.
            device_ip: The device's IPv4 address.

        Returns:
            True if the delete was sent, False if the datapath isn't connected.
        """
        datapath = self._get_datapath(datapath_id)
        if datapath is None:
            return False
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto

        match = parser.OFPMatch(eth_type=0x0800, ipv4_src=device_ip)
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=PRIORITY_MITIGATION,
            match=match,
            command=ofproto.OFPFC_DELETE,
            out_port=ofproto.OFPP_ANY,
            out_group=ofproto.OFPG_ANY,
        )
        datapath.send_msg(mod)

        key = (datapath_id, device_ip)
        if key in self._meter_ids:
            meter_id = self._meter_ids.pop(key)
            meter_mod = parser.OFPMeterMod(
                datapath=datapath, command=ofproto.OFPMC_DELETE, meter_id=meter_id
            )
            datapath.send_msg(meter_mod)

        logger.info("Cleared mitigation: datapath_id=%s device_ip=%s", datapath_id, device_ip)
        return True

    # --- Live switch state (read by the dashboard, component 13) --------
    #
    # Every value below is queried from the switch itself via OpenFlow
    # multipart requests at call time -- not a record of what this
    # controller *thinks* it installed -- so the dashboard shows the
    # actual flow table, the same thing `ovs-ofctl dump-flows` shows.

    def get_flow_table(self, datapath_id: int) -> list[dict] | None:
        """Query the switch's current flow table.

        Returns:
            One dict per flow entry (priority, match, actions, packet/byte
            counters, age), highest priority first; None if the datapath
            isn't connected or doesn't answer in time.
        """
        datapath = self._get_datapath(datapath_id)
        if datapath is None:
            return None
        body = self._request_stats(datapath, datapath.ofproto_parser.OFPFlowStatsRequest(datapath))
        if body is None:
            return None
        flows = [
            {
                "priority": stat.priority,
                "match": dict(stat.match.items()),
                "actions": describe_instructions(stat.instructions, datapath.ofproto),
                "packet_count": stat.packet_count,
                "byte_count": stat.byte_count,
                "duration_sec": stat.duration_sec,
            }
            for stat in body
        ]
        return sorted(flows, key=lambda f: -f["priority"])

    def get_meter_stats(self, datapath_id: int) -> list[dict] | None:
        """Query the switch's meter counters -- how much traffic each rate-limit meter saw and dropped.

        Returns:
            One dict per meter; None if the datapath isn't connected or
            doesn't answer in time.
        """
        datapath = self._get_datapath(datapath_id)
        if datapath is None:
            return None
        request = datapath.ofproto_parser.OFPMeterStatsRequest(datapath, 0, datapath.ofproto.OFPM_ALL)
        body = self._request_stats(datapath, request)
        if body is None:
            return None
        return [
            {
                "meter_id": stat.meter_id,
                "flow_count": stat.flow_count,
                "packet_in_count": stat.packet_in_count,
                "byte_in_count": stat.byte_in_count,
                "packets_dropped_by_band": sum(band.packet_band_count for band in stat.band_stats),
            }
            for stat in body
        ]

    def get_port_status(self, datapath_id: int) -> list[dict] | None:
        """Query the switch's ports: name, link state, and live traffic counters.

        Combines a port-description request (names, link up/down) with a
        port-stats request (rx/tx packets, bytes, drops).

        Returns:
            One dict per port, ordered by port number; None if the datapath
            isn't connected or doesn't answer in time.
        """
        datapath = self._get_datapath(datapath_id)
        if datapath is None:
            return None
        parser = datapath.ofproto_parser
        ofproto = datapath.ofproto
        desc_body = self._request_stats(datapath, parser.OFPPortDescStatsRequest(datapath, 0))
        stats_body = self._request_stats(datapath, parser.OFPPortStatsRequest(datapath, 0, ofproto.OFPP_ANY))
        if desc_body is None or stats_body is None:
            return None

        stats_by_port = {stat.port_no: stat for stat in stats_body}
        ports = []
        for desc in desc_body:
            if desc.port_no > ofproto.OFPP_MAX:  # the switch's LOCAL port, not a link
                continue
            name = desc.name.decode(errors="replace") if isinstance(desc.name, bytes) else str(desc.name)
            link_up = not (desc.state & ofproto.OFPPS_LINK_DOWN) and not (desc.config & ofproto.OFPPC_PORT_DOWN)
            stat = stats_by_port.get(desc.port_no)
            ports.append(
                {
                    "port_no": desc.port_no,
                    "name": name.rstrip("\x00"),
                    "link_up": bool(link_up),
                    **{
                        counter: getattr(stat, counter, 0)
                        for counter in ("rx_packets", "tx_packets", "rx_bytes", "tx_bytes", "rx_dropped", "tx_dropped")
                    },
                }
            )
        return sorted(ports, key=lambda p: p["port_no"])

    def _request_stats(self, datapath, request) -> list | None:
        """Send a multipart stats request and wait (cooperatively) for its full reply.

        The reply handlers below run on the same eventlet hub, so waiting
        on a `hub.Event` yields to them rather than deadlocking. A reply
        can arrive split across several messages (OFPMPF_REPLY_MORE);
        bodies accumulate until the last part.

        Returns:
            The concatenated reply body, or None on timeout.
        """
        datapath.set_xid(request)
        event = hub.Event()
        with self._lock:
            self._pending_stats[request.xid] = (event, [])
        datapath.send_msg(request)
        replied = event.wait(timeout=STATS_REPLY_TIMEOUT_S)
        with self._lock:
            _, body = self._pending_stats.pop(request.xid, (None, []))
        if not replied:
            logger.warning("Stats request %s to datapath_id=%s timed out", type(request).__name__, datapath.id)
            return None
        return body

    def _on_stats_reply(self, ev) -> None:
        """Shared handler for every multipart reply type: route the body to its waiting request by xid."""
        msg = ev.msg
        with self._lock:
            pending = self._pending_stats.get(msg.xid)
            if pending is None:
                return
            event, body = pending
            body.extend(msg.body)
        if not (msg.flags & msg.datapath.ofproto.OFPMPF_REPLY_MORE):
            event.set()

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def _on_flow_stats_reply(self, ev) -> None:
        self._on_stats_reply(ev)

    @set_ev_cls(ofp_event.EventOFPMeterStatsReply, MAIN_DISPATCHER)
    def _on_meter_stats_reply(self, ev) -> None:
        self._on_stats_reply(ev)

    @set_ev_cls(ofp_event.EventOFPPortStatsReply, MAIN_DISPATCHER)
    def _on_port_stats_reply(self, ev) -> None:
        self._on_stats_reply(ev)

    @set_ev_cls(ofp_event.EventOFPPortDescStatsReply, MAIN_DISPATCHER)
    def _on_port_desc_stats_reply(self, ev) -> None:
        self._on_stats_reply(ev)

    def _get_datapath(self, datapath_id: int):
        with self._lock:
            return self.datapaths.get(datapath_id)

    def _allocate_meter_id(self, datapath_id: int, device_ip: str) -> int:
        key = (datapath_id, device_ip)
        with self._lock:
            if key not in self._meter_ids:
                self._meter_ids[key] = self._next_meter_id
                self._next_meter_id += 1
            return self._meter_ids[key]


def start_controller(listen_port: int = 6653) -> MitigationController:
    """Start the controller's event loop and return the running app instance.

    Must be called from a process that has already run
    `eventlet.monkey_patch()` before any other imports (standard os-ken
    requirement). Blocks the caller's greenthread scheduling only in the
    sense that the returned services should be joined by the caller (see
    `fl_ids.sdn.bridge` for the combined controller+REST entrypoint) —
    this function itself returns as soon as the app is instantiated and
    its listener is live.

    Args:
        listen_port: TCP port to listen for switch connections on
            (OpenFlow's IANA-assigned default is 6653).

    Returns:
        The running `MitigationController` app instance.
    """
    from os_ken import cfg

    CONF = cfg.CONF
    CONF(["--ofp-tcp-listen-port", str(listen_port)], project="os_ken", version="os_ken")

    app_mgr = app_manager.AppManager.get_instance()
    app_mgr.load_apps(["os_ken.controller.ofp_handler", __name__])
    contexts = app_mgr.create_contexts()
    app_mgr.instantiate_apps(**contexts)

    controller = app_mgr.applications[MitigationController.__name__]
    logger.info("OpenFlow 1.3 controller listening on port %d", listen_port)
    return controller
