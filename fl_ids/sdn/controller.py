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
from os_ken.controller.handler import CONFIG_DISPATCHER, set_ev_cls
from os_ken.ofproto import ofproto_v1_3

logger = logging.getLogger(__name__)

# Priorities: higher wins. Default NORMAL is the fallback; block/rate-limit
# rules must outrank it so they actually take effect for matched traffic.
PRIORITY_DEFAULT = 0
PRIORITY_MITIGATION = 100

METER_ID_BASE = 1000  # meter IDs are allocated per-device starting here


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
