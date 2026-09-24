"""Mininet topology (component 10, network half).

One host per simulated client, on a single OpenFlow-controlled switch,
pointed at the custom controller (component 10's other half,
`fl_ids.sdn.controller`) — not Mininet's own built-in controller.

**Source-agnostic by design:** Phase B's default traffic source is
Mininet-hosted replay (`tcpreplay` against real Edge-IIoTset `.pcap`
captures), but the topology doesn't hard-code that assumption — a switch
port can optionally bridge onto an existing physical/VM NIC
(`--bridge-iface`), so traffic from an actual external device (another
machine on the LAN, a Raspberry Pi, a VM) can be routed through the same
OpenFlow-controlled switch alongside the Mininet-hosted hosts, if wanted
later. Not built further than that until it's actually needed, per
CLAUDE.md.

Must be run as root (Mininet creates network namespaces/veth pairs) —
`sudo python3 -m fl_ids.sdn.topology`, or via the project's scoped
sudoers rule (see the venv's `bin/python3` invocation in the deployment
notes).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# The sudoers NOPASSWD rule this script runs under (see module docstring)
# requires an exact argv match, so extra flags like `-u` aren't available
# for unbuffered output -- reconfigured here instead, so `sudo python3
# topology.py` output shows up promptly rather than sitting in a buffer.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# This script runs under a narrowly-scoped sudoers NOPASSWD rule pointing
# at the system python3 (Mininet needs a real root process; a venv's
# python3 is usually just a symlink to it anyway) -- which means the
# project's pip-installed packages (numpy, torch, flask, os-ken, ...)
# aren't on sys.path unless added explicitly, since `sudo python3` here
# never goes through the venv's activation machinery. Injected before any
# of those get imported, elsewhere in this file or its lazy imports.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_VENV_SITE_PACKAGES = _PROJECT_ROOT / "venv" / "lib" / "python3.14" / "site-packages"
for _extra_path in (_PROJECT_ROOT, _VENV_SITE_PACKAGES):
    if _extra_path.exists() and str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))

from mininet.cli import CLI
from mininet.link import Intf
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController

from fl_ids.sdn.demo_io import DEMO_CONFIG_PATH, DEMO_RESULT_PATH

logger = logging.getLogger(__name__)


def build_topology(
    num_hosts: int,
    controller_ip: str = "127.0.0.1",
    controller_port: int = 6653,
    bridge_physical_iface: str | None = None,
) -> Mininet:
    """Build and start the topology: N hosts on one OVS switch, real controller.

    Args:
        num_hosts: Number of simulated client hosts (one per Edge-IIoTset device).
        controller_ip: Where the OpenFlow controller (component 10) is listening.
        controller_port: OpenFlow controller's TCP port.
        bridge_physical_iface: If given, an existing physical/VM NIC name
            to bridge onto the switch — see module docstring.

    Returns:
        The running `Mininet` network. Caller is responsible for
        `.stop()` when done.
    """
    net = Mininet(switch=OVSSwitch, controller=None, autoSetMacs=True)

    net.addController("c0", controller=RemoteController, ip=controller_ip, port=controller_port)
    switch = net.addSwitch("s1", protocols="OpenFlow13")

    hosts = []
    for i in range(num_hosts):
        host = net.addHost(f"h{i + 1}", ip=f"10.0.0.{i + 1}/24")
        net.addLink(host, switch)
        hosts.append(host)

    net.start()

    if bridge_physical_iface:
        Intf(bridge_physical_iface, node=switch)
        logger.info("Bridged physical interface %s onto switch %s", bridge_physical_iface, switch.name)

    logger.info(
        "Topology up: %d hosts on switch %s, controller at %s:%d",
        num_hosts, switch.name, controller_ip, controller_port,
    )
    return net


def host_ip_map(net: Mininet, num_hosts: int) -> dict[str, str]:
    """Return `{host_name: ip_address}` for the topology's simulated-client hosts."""
    return {f"h{i + 1}": f"10.0.0.{i + 1}" for i in range(num_hosts)}


def replay_pcap(
    net: Mininet,
    pcap_path: str,
    replay_host_name: str = "h1",
    switch_name: str = "s1",
) -> tuple[str, int, int]:
    """Replay a real `.pcap` capture from one host through the topology.

    This is the "real traffic, not a stand-in" half of Phase 7/9's live
    demo: `tcpreplay` actually sends the captured packets out the host's
    interface, and they traverse the real OpenFlow-controlled switch —
    verified here via the switch's own packet counters (read before and
    after replay), rather than a second concurrent capture racing the
    replay (a `tcpdump`-while-`tcpreplay` setup proved fragile to
    synchronize reliably; the switch's own flow-table byte/packet
    counters are a simpler, equally real way to confirm transit).

    Args:
        net: The running `Mininet` network.
        pcap_path: Path to the real `.pcap` file to replay (e.g. one of
            Edge-IIoTset's per-device/per-attack captures).
        replay_host_name: Which host replays the traffic.
        switch_name: Which switch to read packet counters from.

    Returns:
        (tcpreplay's own output, packets_before, packets_after) — the
        counter delta confirms real transit.
    """
    host = net.get(replay_host_name)
    switch = net.get(switch_name)

    packets_before = _switch_packet_count(switch)
    replay_cmd = f"tcpreplay --intf1={host.defaultIntf().name} --topspeed '{pcap_path}'"
    logger.info("Replaying: %s", replay_cmd)
    output = host.cmd(replay_cmd)
    logger.info("tcpreplay output: %s", output.strip())
    packets_after = _switch_packet_count(switch)

    logger.info("Switch packet count: %d -> %d (delta=%d)", packets_before, packets_after, packets_after - packets_before)
    return output, packets_before, packets_after


def _switch_packet_count(switch) -> int:
    """Sum `n_packets` across all of a switch's installed flows (via `ovs-ofctl dump-flows`)."""
    import re

    output = switch.cmd(f"ovs-ofctl -O OpenFlow13 dump-flows {switch.name}")
    return sum(int(m) for m in re.findall(r"n_packets=(\d+)", output))


def run_live_pcap_demo(
    net: Mininet,
    pcap_path: str,
    artifact_dir: str | Path,
    bridge_url: str,
    replay_host_name: str = "h1",
    result_path: str | Path = DEMO_RESULT_PATH,
) -> dict:
    """Phase B: replay real traffic, classify it with the Phase A models, mitigate it for real.

    Loads the Phase A bundle (never trains anything here), replays
    `pcap_path` from `replay_host_name` through the topology, extracts
    features from the capture, classifies the device the way its FL client
    would (`fl_ids.sdn.live_inference`), and POSTs the verdict to the
    running SDN bridge — which installs a real OpenFlow rule, verifiable
    via `ovs-ofctl dump-flows`.

    Args:
        net: The running `Mininet` network.
        pcap_path: Real `.pcap` capture to replay.
        artifact_dir: Phase A bundle directory.
        bridge_url: Base URL of the running SDN mitigation bridge (component 9).
        replay_host_name: Which host replays the traffic (h<i+1> is FL client i).
        result_path: Where to write the JSON result for the Phase B orchestrator.

    Returns:
        The result: the device classification, the bridge's `/mitigate`
        response, the switch's packet counters around the replay, and the
        flow table afterwards.
    """
    import requests

    from fl_ids.orchestration.artifacts import load_phase_a_artifacts
    from fl_ids.sdn.feature_extraction import extract_features_for_inference
    from fl_ids.sdn.live_inference import classify_device_traffic
    from fl_ids.utils.config import load_config

    artifacts = load_phase_a_artifacts(artifact_dir, load_config().boosting)
    logger.info(
        "Loaded Phase A bundle from %s (%d clients, trained %s)",
        artifact_dir, len(artifacts.client_ids), artifacts.manifest.get("provenance", {}).get("created_at", "?"),
    )

    # Register every host with the bridge up front (not just the replaying
    # one, and before any traffic flows), so the dashboard's device and
    # topology panels are populated for the whole run.
    health = requests.get(f"{bridge_url}/health", timeout=5).json()
    datapath_ids = health["connected_datapaths"]
    if not datapath_ids:
        raise RuntimeError("No datapath connected to the controller yet")
    for host_name, host_ip in host_ip_map(net, len(net.hosts)).items():
        link = net.get(host_name).defaultIntf().link
        switch_intf = link.intf2 if link.intf1.node.name == host_name else link.intf1
        requests.post(
            f"{bridge_url}/devices/register",
            json={
                "device_id": host_name,
                "datapath_id": datapath_ids[0],
                "ip_address": host_ip,
                "port_name": switch_intf.name,
            },
            timeout=5,
        )

    _tcpreplay_output, packets_before, packets_after = replay_pcap(net, pcap_path, replay_host_name=replay_host_name)
    if packets_after <= packets_before:
        logger.warning(
            "Switch packet counters didn't increase during replay (before=%d, after=%d) -- "
            "traffic may not have actually transited the switch",
            packets_before, packets_after,
        )
    live_X_raw = extract_features_for_inference(pcap_path, artifacts.feature_names)
    verdict = classify_device_traffic(artifacts, replay_host_name, live_X_raw)
    logger.info(
        "Device-level classification for %s (client %d): %s (confidence=%.3f, stage=%s, %d/%d packets non-benign)",
        verdict.device_id, verdict.client_id, verdict.classification, verdict.confidence, verdict.stage,
        verdict.num_non_benign, verdict.num_packets,
    )

    response = requests.post(
        f"{bridge_url}/mitigate",
        json={
            "device_id": verdict.device_id,
            "classification": verdict.classification,
            "confidence": verdict.confidence,
            "stage": verdict.stage,
        },
        timeout=5,
    )
    mitigation = response.json()
    logger.info("Mitigation result: %s", mitigation)

    # Capture the real flow-table proof *now*, from inside this process --
    # the topology (and with it, the switch) gets torn down right after
    # this returns, so an external `ovs-ofctl` check racing that teardown
    # isn't reliable.
    switch = net.get("s1")
    flow_dump = switch.cmd("ovs-ofctl -O OpenFlow13 dump-flows s1")
    logger.info("Flow table after mitigation (ovs-ofctl dump-flows s1):\n%s", flow_dump)

    result = {
        "pcap": str(pcap_path),
        "artifact_dir": str(artifact_dir),
        "device_ip": host_ip_map(net, len(net.hosts))[replay_host_name],
        "classification": verdict.to_dict(),
        "mitigation": mitigation,
        "switch_packets_before": packets_before,
        "switch_packets_after": packets_after,
        "flow_table": flow_dump,
    }
    import json

    Path(result_path).write_text(json.dumps(result, indent=2))
    print(f"=== LIVE DEMO RESULT ===\n{json.dumps(result, indent=2)}", flush=True)
    return result


# This script's sudoers NOPASSWD rule requires an exact command match with
# no arguments (this system's sudo policy doesn't extend the usual
# "no args listed = any args allowed" convention) -- so it must also be
# runnable with a bare `sudo python3 topology.py`, zero argv. Optional
# demo parameters are read from this fixed path instead when present;
# CLI flags still work fine for direct (non-sudo-constrained) use and
# override the file's values.
DEFAULT_DEMO_CONFIG_PATH = DEMO_CONFIG_PATH


def _load_demo_config_defaults(path: Path) -> dict:
    """Read optional demo parameters from a JSON file, if present."""
    if not path.exists():
        return {}
    import json

    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    from fl_ids.utils.logging_setup import setup_logging

    setup_logging(level="INFO")
    setLogLevel("info")

    file_defaults = _load_demo_config_defaults(DEFAULT_DEMO_CONFIG_PATH)

    parser = argparse.ArgumentParser(description="Bring up the FL-IDS Mininet topology (component 10)")
    parser.add_argument("--num-hosts", type=int, default=10)
    parser.add_argument("--controller-ip", default="127.0.0.1")
    parser.add_argument("--controller-port", type=int, default=6653)
    parser.add_argument("--bridge-iface", default=None)
    parser.add_argument("--cli", action="store_true", help="Drop into the Mininet CLI after startup")
    parser.add_argument("--replay-pcap", default=None, help="Run the live classification+mitigation demo with this .pcap")
    parser.add_argument(
        "--artifact-dir", default=str(_PROJECT_ROOT / "saved_models" / "phase_a"),
        help="Phase A model bundle to classify with (see fl_ids.orchestration.phase_a)",
    )
    parser.add_argument("--replay-host", default="h1", help="Host that replays the capture (h<i+1> is FL client i)")
    parser.add_argument("--result-path", default=str(DEMO_RESULT_PATH))
    parser.add_argument("--bridge-url", default="http://127.0.0.1:8080")
    parser.add_argument(
        "--hold-seconds", type=float, default=0.0,
        help="Keep the topology up this long after the demo, so the dashboard can observe the mitigated state",
    )
    parser.set_defaults(**file_defaults)
    args = parser.parse_args()

    mininet_net = build_topology(args.num_hosts, args.controller_ip, args.controller_port, args.bridge_iface)
    try:
        if args.replay_pcap:
            run_live_pcap_demo(
                mininet_net, args.replay_pcap, args.artifact_dir, args.bridge_url,
                replay_host_name=args.replay_host, result_path=args.result_path,
            )
        if args.hold_seconds > 0:
            import time

            logger.info("Holding topology up for %.0fs", args.hold_seconds)
            time.sleep(args.hold_seconds)
        if args.cli:
            CLI(mininet_net)
    finally:
        mininet_net.stop()
