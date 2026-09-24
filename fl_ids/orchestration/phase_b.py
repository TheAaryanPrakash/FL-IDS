"""Phase B entrypoint (component 11): live mitigation with the Phase A models.

    python -m fl_ids.orchestration.phase_b --pcap "data/raw/pcaps/Backdoor_attack.pcap" [--replay-host h1]

1. Load the Phase A bundle up front, so a missing or inconsistent bundle
   fails here rather than midway through a root-run network setup.
2. Start the OpenFlow controller + SDN mitigation bridge (components 9, 10).
3. Hand the topology script its parameters through the fixed demo-config
   file and run it with the one command the sudoers rule allows
   (`sudo -n /usr/bin/python3 <repo>/fl_ids/sdn/topology.py`, no
   arguments). It brings up Mininet, replays the capture from the chosen
   host, classifies that host's traffic with the bundle
   (`fl_ids.sdn.live_inference`: host h<i+1> is FL client i) and posts
   the verdict to the bridge, which installs the flow rule.
4. Read the topology's JSON result and check the flow table it captured:
   a block or rate-limit verdict must show up as a rule matching the
   device's IP, the same thing `ovs-ofctl dump-flows` shows.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

from fl_ids.orchestration.artifacts import load_phase_a_artifacts
from fl_ids.sdn.demo_io import DEMO_CONFIG_PATH, DEMO_RESULT_PATH
from fl_ids.utils.config import Config, write_config_yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
TOPOLOGY_SCRIPT = REPO_ROOT / "fl_ids" / "sdn" / "topology.py"
# The exact command the sudoers rule allows (see fl_ids.sdn.demo_io).
SYSTEM_PYTHON = "/usr/bin/python3"
BRIDGE_STARTUP_TIMEOUT_SECONDS = 30.0
TOPOLOGY_TIMEOUT_SECONDS = 900.0


def flow_rule_for_ip(flow_table: str, ip_address: str) -> str | None:
    """The first `ovs-ofctl dump-flows` line whose match is on this source IP, if any.

    Matches the exact address, so 10.0.0.1 doesn't match a rule for 10.0.0.10.
    """
    pattern = re.compile(rf"nw_src={re.escape(ip_address)}(?![\d])")
    for line in flow_table.splitlines():
        if pattern.search(line):
            return line.strip()
    return None


def check_mitigation(result: dict) -> dict:
    """Whether the switch's flow table reflects the bridge's decision.

    Args:
        result: The topology script's JSON result.

    Returns:
        `{"action", "expected_rule", "observed_rule", "consistent"}` —
        a block/rate-limit decision needs a rule on the device's IP,
        and an allow decision must not leave one.
    """
    action = result["mitigation"].get("action")
    observed = flow_rule_for_ip(result["flow_table"], result["device_ip"])
    expected = action in ("block", "rate_limit")
    return {
        "action": action,
        "expected_rule": expected,
        "observed_rule": observed,
        "consistent": (observed is not None) == expected,
    }


def _start_bridge(config: Config, config_path: Path, log_path: Path) -> subprocess.Popen:
    bridge_url = f"http://{config.sdn.bridge_api_host}:{config.sdn.bridge_api_port}"
    try:
        requests.get(f"{bridge_url}/health", timeout=2)
    except requests.ConnectionError:
        pass
    else:
        raise RuntimeError(f"Something is already serving {bridge_url}; stop it before running Phase B")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        [
            sys.executable, "-m", "fl_ids.sdn.bridge",
            "--config-path", str(config_path),
            "--listen-host", config.sdn.bridge_api_host,
            "--listen-port", str(config.sdn.bridge_api_port),
            "--of-listen-port", str(config.sdn.controller_port),
        ],
        cwd=REPO_ROOT, stdout=log_file, stderr=subprocess.STDOUT,
    )
    proc._log_file = log_file  # closed in _stop_bridge
    deadline = time.monotonic() + BRIDGE_STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"Bridge exited with code {proc.returncode}; see {log_path}")
        try:
            requests.get(f"{bridge_url}/health", timeout=2).raise_for_status()
            logger.info("Controller + bridge up at %s (OpenFlow port %d)", bridge_url, config.sdn.controller_port)
            return proc
        except requests.RequestException:
            time.sleep(0.5)
    _stop_bridge(proc)
    raise RuntimeError(f"Bridge didn't answer on {bridge_url} within {BRIDGE_STARTUP_TIMEOUT_SECONDS}s; see {log_path}")


def _stop_bridge(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    proc._log_file.close()


def run_phase_b(
    config: Config,
    pcap_path: str | Path,
    artifact_dir: str | Path | None = None,
    replay_host: str = "h1",
    hold_seconds: float = 0.0,
    run_dir: str | Path = "runs/phase_b",
) -> dict:
    """Run Phase B end to end and return the live demo's result.

    Args:
        config: Full project config (`sdn.*` for ports and thresholds).
        pcap_path: Real capture to replay.
        artifact_dir: Phase A bundle; defaults to `orchestration.artifact_dir`.
        replay_host: Host that replays the capture (h<i+1> is FL client i).
        hold_seconds: Keep the topology (and bridge) up this long after the
            verdict, so the dashboard's live view can watch the mitigated state.
        run_dir: Where to write logs and the result.

    Returns:
        The topology's result plus `"check"` (`check_mitigation`).

    Raises:
        FileNotFoundError: Missing capture or bundle.
        RuntimeError: If the bridge or topology fails, or the flow table
            doesn't reflect the bridge's decision.
    """
    artifact_dir = Path(artifact_dir or config.orchestration.artifact_dir).resolve()
    pcap_path = Path(pcap_path).resolve()
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not pcap_path.exists():
        raise FileNotFoundError(pcap_path)

    artifacts = load_phase_a_artifacts(artifact_dir, config.boosting)
    num_hosts = len(artifacts.client_ids)
    logger.info("Phase A bundle %s: %d clients -> %d Mininet hosts", artifact_dir, num_hosts, num_hosts)

    config_path = write_config_yaml(config, run_dir / "config.yaml")
    bridge = _start_bridge(config, config_path, run_dir / "bridge.log")
    try:
        DEMO_RESULT_PATH.unlink(missing_ok=True)
        DEMO_CONFIG_PATH.write_text(json.dumps({
            "num_hosts": num_hosts,
            "controller_ip": config.sdn.controller_host,
            "controller_port": config.sdn.controller_port,
            "replay_pcap": str(pcap_path),
            "artifact_dir": str(artifact_dir),
            "replay_host": replay_host,
            "bridge_url": f"http://{config.sdn.bridge_api_host}:{config.sdn.bridge_api_port}",
            "result_path": str(DEMO_RESULT_PATH),
            "hold_seconds": hold_seconds,
        }))
        logger.info("Running the topology as root: sudo -n %s %s", SYSTEM_PYTHON, TOPOLOGY_SCRIPT)
        with open(run_dir / "topology.log", "w") as topology_log:
            completed = subprocess.run(
                ["sudo", "-n", SYSTEM_PYTHON, str(TOPOLOGY_SCRIPT)],
                cwd=REPO_ROOT, stdout=topology_log, stderr=subprocess.STDOUT,
                timeout=TOPOLOGY_TIMEOUT_SECONDS + hold_seconds,
            )
        if completed.returncode != 0:
            raise RuntimeError(f"Topology exited with code {completed.returncode}; see {run_dir / 'topology.log'}")
        if not DEMO_RESULT_PATH.exists():
            raise RuntimeError(f"Topology finished without writing {DEMO_RESULT_PATH}; see {run_dir / 'topology.log'}")
        result = json.loads(DEMO_RESULT_PATH.read_text())
    finally:
        _stop_bridge(bridge)

    result["check"] = check_mitigation(result)
    (run_dir / "phase_b_result.json").write_text(json.dumps(result, indent=2))

    verdict = result["classification"]
    logger.info(
        "Phase B: %s (client %d) -> %s (confidence %.3f, %s stage, %d/%d packets non-benign) -> %s; flow rule: %s",
        verdict["device_id"], verdict["client_id"], verdict["classification"], verdict["confidence"],
        verdict["stage"], verdict["num_non_benign"], verdict["num_packets"], result["check"]["action"],
        result["check"]["observed_rule"] or "none",
    )
    if not result["check"]["consistent"]:
        raise RuntimeError(f"Flow table doesn't reflect the bridge's decision: {result['check']}")
    return result


if __name__ == "__main__":
    import argparse

    from fl_ids.utils.config import load_config
    from fl_ids.utils.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Phase B: live mitigation with the Phase A model bundle")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--pcap", required=True, help="Real .pcap capture to replay")
    parser.add_argument("--artifact-dir", default=None, help="Defaults to orchestration.artifact_dir")
    parser.add_argument("--replay-host", default="h1", help="Host replaying the capture (h<i+1> is FL client i)")
    parser.add_argument("--hold-seconds", type=float, default=0.0,
                        help="Keep the network up after the verdict, for the dashboard's live view")
    parser.add_argument("--run-dir", default="runs/phase_b")
    args = parser.parse_args()

    run_config = load_config(args.config_path)
    setup_logging(run_config.logging.level, run_config.logging.log_dir, run_config.logging.log_file)
    run_phase_b(run_config, args.pcap, args.artifact_dir, args.replay_host, args.hold_seconds, args.run_dir)
