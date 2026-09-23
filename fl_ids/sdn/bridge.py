"""SDN mitigation bridge (component 9).

A REST API receiving `{device_id, classification, confidence}` — the
*combined* cascade output (boosting's label when confident, else the
autoencoder's anomaly flag) — and calling into the OpenFlow controller
(component 10) to install the appropriate flow rule based on confidence
thresholds: >= `block_confidence_threshold` -> block, >=
`rate_limit_confidence_threshold` -> rate-limit, else allow/clear any
existing mitigation. Benign classifications are never mitigated
regardless of confidence.

The pure decision logic (`classify_mitigation_action`) and device
registry are importable and testable without starting any server or
touching `os-ken`/`eventlet` — only `run_bridge` (the actual entrypoint)
needs the real controller and its eventlet-based event loop.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

from fl_ids.models.cascade import BENIGN_LABEL
from fl_ids.utils.config import Config, SDNConfig

if TYPE_CHECKING:
    from flask import Flask

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_KBPS = 1000

CASCADE_STAGES = ("boosting", "autoencoder")


class DeviceRegistry:
    """Maps device_id -> (datapath_id, ip_address, port_name).

    `port_name` (the switch-side interface the device hangs off, e.g.
    `s1-eth1`) is optional -- mitigation only needs the IP -- but lets the
    dashboard line up a device with its port's link status and counters.

    Populated by the Mininet topology script (component 10) or a test
    harness at startup — not persisted; a real deployment would source
    this from the topology/DHCP registry instead.
    """

    def __init__(self) -> None:
        self._devices: dict[str, tuple[int, str, str | None]] = {}

    def register(self, device_id: str, datapath_id: int, ip_address: str, port_name: str | None = None) -> None:
        self._devices[device_id] = (datapath_id, ip_address, port_name)

    def lookup(self, device_id: str) -> tuple[int, str] | None:
        """Return `(datapath_id, ip_address)` for a device, or None if unregistered."""
        entry = self._devices.get(device_id)
        return entry[:2] if entry else None

    def as_dict(self) -> dict[str, dict[str, str | int | None]]:
        return {
            device_id: {"datapath_id": dp_id, "ip_address": ip, "port_name": port_name}
            for device_id, (dp_id, ip, port_name) in self._devices.items()
        }


def classify_mitigation_action(classification: str, confidence: float, sdn_config: SDNConfig) -> str:
    """Map a cascade classification + confidence to a mitigation action.

    Args:
        classification: The cascade's final label — a specific attack
            type name, `"anomalous"`, or `"benign"`.
        confidence: The cascade's confidence for that classification, in `[0, 1]`.
        sdn_config: `block_confidence_threshold`, `rate_limit_confidence_threshold`.

    Returns:
        `"block"`, `"rate_limit"`, or `"allow"`.
    """
    if classification == BENIGN_LABEL:
        return "allow"
    if confidence >= sdn_config.block_confidence_threshold:
        return "block"
    if confidence >= sdn_config.rate_limit_confidence_threshold:
        return "rate_limit"
    return "allow"


def create_bridge_app(
    controller,
    registry: DeviceRegistry,
    sdn_config: SDNConfig,
    rate_limit_kbps: int = DEFAULT_RATE_LIMIT_KBPS,
) -> Flask:
    """Build the Flask app wiring HTTP requests to controller calls.

    Args:
        controller: A running `fl_ids.sdn.controller.MitigationController`.
        registry: Maps device_id to (datapath_id, ip_address).
        sdn_config: Mitigation confidence thresholds.
        rate_limit_kbps: Rate-limit meter rate applied for `"rate_limit"` actions.

    Returns:
        A configured Flask app, not yet serving.
    """
    from flask import Flask, jsonify, request

    app = Flask(__name__)
    mitigation_history: list[dict] = []

    @app.route("/mitigate", methods=["POST"])
    def mitigate():
        """Apply the mitigation for one cascade classification.

        Body: `{device_id, classification, confidence, stage?}` -- `stage`
        (`"boosting"` or `"autoencoder"`, which cascade stage made the
        call) is optional for the mitigation decision itself but recorded
        in the history the dashboard reads.
        """
        payload = request.get_json(force=True)
        device_id = payload["device_id"]
        classification = payload["classification"]
        confidence = float(payload["confidence"])
        stage = payload.get("stage")
        if stage is not None and stage not in CASCADE_STAGES:
            return jsonify({"error": f"stage must be one of {CASCADE_STAGES}, got {stage!r}"}), 400

        entry = registry.lookup(device_id)
        if entry is None:
            return jsonify({"error": f"unknown device_id: {device_id}"}), 404
        datapath_id, ip_address = entry

        action = classify_mitigation_action(classification, confidence, sdn_config)
        if action == "block":
            applied = controller.block_device(datapath_id, ip_address)
        elif action == "rate_limit":
            applied = controller.rate_limit_device(datapath_id, ip_address, rate_limit_kbps)
        else:
            applied = controller.clear_mitigation(datapath_id, ip_address)

        logger.info(
            "Mitigation: device_id=%s ip=%s classification=%s confidence=%.3f -> action=%s applied=%s",
            device_id, ip_address, classification, confidence, action, applied,
        )
        result = {
            "timestamp": time.time(),
            "device_id": device_id,
            "ip_address": ip_address,
            "classification": classification,
            "confidence": confidence,
            "stage": stage,
            "action": action,
            "applied": applied,
        }
        mitigation_history.append(result)
        return (
            jsonify(result),
            200,
        )

    @app.route("/devices", methods=["GET"])
    def list_devices():
        return jsonify(registry.as_dict())

    @app.route("/devices/register", methods=["POST"])
    def register_device():
        payload = request.get_json(force=True)
        device_id = payload["device_id"]
        datapath_id = int(payload["datapath_id"])
        ip_address = payload["ip_address"]
        registry.register(device_id, datapath_id, ip_address, port_name=payload.get("port_name"))
        logger.info("Registered device: device_id=%s datapath_id=%d ip=%s", device_id, datapath_id, ip_address)
        return jsonify({"registered": device_id}), 200

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "connected_datapaths": controller.get_datapath_ids()})

    @app.route("/mitigation_history", methods=["GET"])
    def get_mitigation_history():
        """Every `/mitigate` call this process has handled — the dashboard's
        (component 13) live classifications feed for the Phase B view.
        """
        return jsonify(mitigation_history)

    @app.route("/switch_state", methods=["GET"])
    def switch_state():
        """Live flow table, meter counters, and port/link status for every connected switch.

        Queried from the switches themselves on each request (see
        `MitigationController.get_flow_table` and friends) -- the
        dashboard's view of real flow-table state, not a record of what
        the bridge believes it installed. A datapath that doesn't answer
        in time reports `null` for that section.
        """
        state = {}
        for dp_id in controller.get_datapath_ids():
            state[str(dp_id)] = {
                "flows": controller.get_flow_table(dp_id),
                "meters": controller.get_meter_stats(dp_id),
                "ports": controller.get_port_status(dp_id),
            }
        return jsonify(state)

    return app


def run_bridge(
    config: Config,
    registry: DeviceRegistry,
    listen_host: str = "0.0.0.0",
    listen_port: int = 8080,
    of_listen_port: int = 6653,
) -> None:
    """Entrypoint: starts the OpenFlow controller and REST bridge in one process.

    Both run on the same eventlet event loop, so the REST handlers can
    call the controller's methods directly (safe cooperative scheduling,
    no IPC needed between them).

    Args:
        config: Full project config.
        registry: Pre-populated device registry (topology script fills
            this in before calling `run_bridge`).
        listen_host: REST API bind address.
        listen_port: REST API bind port.
        of_listen_port: OpenFlow controller bind port (switches connect here).
    """
    import eventlet
    import eventlet.wsgi

    eventlet.monkey_patch()

    from fl_ids.sdn.controller import start_controller

    controller = start_controller(listen_port=of_listen_port)
    app = create_bridge_app(controller, registry, config.sdn)

    logger.info("SDN mitigation bridge listening on %s:%d", listen_host, listen_port)
    eventlet.wsgi.server(eventlet.listen((listen_host, listen_port)), app)


if __name__ == "__main__":
    import eventlet

    eventlet.monkey_patch()

    import argparse

    from fl_ids.utils.config import load_config
    from fl_ids.utils.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Run the SDN mitigation bridge (component 9)")
    parser.add_argument("--config-path", default=None)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8080)
    parser.add_argument("--of-listen-port", type=int, default=6653)
    args = parser.parse_args()

    setup_logging()
    run_config = load_config(args.config_path)
    empty_registry = DeviceRegistry()
    run_bridge(run_config, empty_registry, args.listen_host, args.listen_port, args.of_listen_port)
