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
    real_csv_path: str,
    bridge_url: str,
    replay_host_name: str = "h1",
) -> dict:
    """Phase 7/9 milestone flow: replay real traffic, classify it live, mitigate it for real.

    Bootstraps a boosting classifier + autoencoder on the real
    Edge-IIoTset calibration set (self-contained for this demo — Phase 9's
    full orchestration loads persisted Phase A models instead), replays
    `pcap_path` through the topology, extracts features from what
    actually transited the switch, runs the full cascade, and POSTs the
    result to the running SDN bridge — which installs a real OpenFlow
    rule, verifiable via `ovs-ofctl dump-flows`.

    Args:
        net: The running `Mininet` network.
        pcap_path: Real `.pcap` capture to replay.
        real_csv_path: Path to `DNN-EdgeIIoT-dataset.csv`, for bootstrapping.
        bridge_url: Base URL of the running SDN mitigation bridge (component 9).
        replay_host_name: Which host replays the traffic / gets registered as the device.

    Returns:
        The bridge's `/mitigate` response payload.
    """
    import requests
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    from fl_ids.data.pipeline import load_and_encode
    from fl_ids.models.autoencoder import Autoencoder, compute_anomaly_threshold, reconstruction_error, train_autoencoder
    from fl_ids.models.boosting import BoostingClassifier
    from fl_ids.models.cascade import cascade_predict
    from fl_ids.sdn.feature_extraction import extract_features_for_inference
    from fl_ids.utils.config import AutoencoderConfig, BoostingConfig, CascadeConfig

    logger.info("Bootstrapping cascade models from %s", real_csv_path)
    X, y, label_encoder, feature_names, benign_class = load_and_encode(real_csv_path)
    X_calib, _, y_calib, _ = train_test_split(X, y, train_size=0.05, random_state=42, stratify=y)

    boosting_config = BoostingConfig(
        label_source="server_held_calibration_set", calibration_fraction=0.05,
        num_boost_round=300, learning_rate=0.1, num_leaves=127,
        broadcast_every_n_rounds=1, update_every_n_rounds=3,
    )
    cascade_config = CascadeConfig(confidence_threshold=0.7, anomaly_confidence_clip=(0.0, 1.0))
    boosting_model = BoostingClassifier(
        boosting_config, len(label_encoder.classes_), benign_class, seed=42,
        confidence_threshold=cascade_config.confidence_threshold,
    )
    boosting_model.train(X_calib, y_calib)

    mask = boosting_model.passes_to_autoencoder(X_calib)
    scaler = StandardScaler().fit(X_calib)
    X_calib_norm = scaler.transform(X_calib).astype("float32")
    ae_config = AutoencoderConfig(
        bottleneck_dim=8, hidden_dims=[32, 16], learning_rate=0.01, local_epochs=5, batch_size=64,
        anomaly_percentile=97, reconstruction_error_bins=20, reconstruction_error_range=(0.0, 5.0),
    )
    autoencoder = Autoencoder(X.shape[1], ae_config.hidden_dims, ae_config.bottleneck_dim)
    train_autoencoder(autoencoder, X_calib_norm[mask], ae_config, seed=42)
    benign_mask = mask & (y_calib == benign_class)
    threshold = compute_anomaly_threshold(
        reconstruction_error(autoencoder, X_calib_norm[benign_mask]), ae_config.anomaly_percentile
    )
    logger.info("Cascade models ready (anomaly threshold=%.4f)", threshold)

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
    live_X_raw = extract_features_for_inference(pcap_path, feature_names)
    live_X_norm = scaler.transform(live_X_raw).astype("float32")

    output = cascade_predict(
        boosting_model, autoencoder, threshold, live_X_raw, live_X_norm,
        list(label_encoder.classes_), cascade_config,
    )

    # Aggregate per-packet cascade output to one device-level classification:
    # the most frequent non-benign label, if any packet triggered one,
    # else "benign". Confidence is that label's mean confidence, and the
    # reported cascade stage is whichever stage made that label's calls.
    non_benign_mask = output.predicted_label != "benign"
    if non_benign_mask.any():
        labels, counts = np.unique(output.predicted_label[non_benign_mask], return_counts=True)
        device_classification = labels[np.argmax(counts)]
        label_mask = non_benign_mask & (output.predicted_label == device_classification)
        device_confidence = float(output.confidence[label_mask].mean())
    else:
        device_classification = "benign"
        label_mask = np.ones(len(output.predicted_label), dtype=bool)
        device_confidence = float(output.confidence.mean())
    stages, stage_counts = np.unique(output.stage[label_mask], return_counts=True)
    device_stage = str(stages[np.argmax(stage_counts)])

    logger.info(
        "Device-level classification for %s: %s (confidence=%.3f, stage=%s, %d/%d packets non-benign)",
        replay_host_name, device_classification, device_confidence, device_stage,
        non_benign_mask.sum(), len(output.predicted_label),
    )

    response = requests.post(
        f"{bridge_url}/mitigate",
        json={
            "device_id": replay_host_name,
            "classification": device_classification,
            "confidence": device_confidence,
            "stage": device_stage,
        },
        timeout=5,
    )
    result = response.json()
    logger.info("Mitigation result: %s", result)

    # Capture the real flow-table proof *now*, from inside this process --
    # the topology (and with it, the switch) gets torn down immediately
    # after this function returns, so an external `ovs-ofctl` check
    # racing that teardown isn't reliable.
    switch = net.get("s1")
    flow_dump = switch.cmd("ovs-ofctl -O OpenFlow13 dump-flows s1")
    logger.info("Flow table after mitigation (ovs-ofctl dump-flows s1):\n%s", flow_dump)
    summary = f"=== LIVE DEMO RESULT ===\n{result}\n\n=== FLOW TABLE (ovs-ofctl dump-flows s1) ===\n{flow_dump}"
    print(summary, flush=True)
    Path("/tmp/fl_ids_live_demo_result.txt").write_text(summary)

    return result


# This script's sudoers NOPASSWD rule requires an exact command match with
# no arguments (this system's sudo policy doesn't extend the usual
# "no args listed = any args allowed" convention) -- so it must also be
# runnable with a bare `sudo python3 topology.py`, zero argv. Optional
# demo parameters are read from this fixed path instead when present;
# CLI flags still work fine for direct (non-sudo-constrained) use and
# override the file's values.
DEFAULT_DEMO_CONFIG_PATH = Path("/tmp/fl_ids_topology_demo_config.json")


def _load_demo_config_defaults(path: Path) -> dict:
    """Read optional demo parameters from a JSON file, if present."""
    if not path.exists():
        return {}
    import json

    with open(path) as f:
        return json.load(f)


if __name__ == "__main__":
    import argparse

    import numpy as np

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
    parser.add_argument("--real-csv-path", default="data/raw/DNN-EdgeIIoT-dataset.csv")
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
            run_live_pcap_demo(mininet_net, args.replay_pcap, args.real_csv_path, args.bridge_url)
        if args.hold_seconds > 0:
            import time

            logger.info("Holding topology up for %.0fs", args.hold_seconds)
            time.sleep(args.hold_seconds)
        if args.cli:
            CLI(mininet_net)
    finally:
        mininet_net.stop()
