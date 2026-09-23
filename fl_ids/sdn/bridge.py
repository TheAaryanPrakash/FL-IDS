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
from typing import TYPE_CHECKING

from fl_ids.models.cascade import BENIGN_LABEL
from fl_ids.utils.config import Config, SDNConfig

if TYPE_CHECKING:
    from flask import Flask

logger = logging.getLogger(__name__)

DEFAULT_RATE_LIMIT_KBPS = 1000


class DeviceRegistry:
    """Maps device_id -> (datapath_id, ip_address).

    Populated by the Mininet topology script (component 10) or a test
    harness at startup — not persisted; a real deployment would source
    this from the topology/DHCP registry instead.
    """

    def __init__(self) -> None:
        self._devices: dict[str, tuple[int, str]] = {}

    def register(self, device_id: str, datapath_id: int, ip_address: str) -> None:
        self._devices[device_id] = (datapath_id, ip_address)

    def lookup(self, device_id: str) -> tuple[int, str] | None:
        return self._devices.get(device_id)

    def as_dict(self) -> dict[str, dict[str, str | int]]:
        return {
            device_id: {"datapath_id": dp_id, "ip_address": ip}
            for device_id, (dp_id, ip) in self._devices.items()
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

    @app.route("/mitigate", methods=["POST"])
    def mitigate():
        payload = request.get_json(force=True)
        device_id = payload["device_id"]
        classification = payload["classification"]
        confidence = float(payload["confidence"])

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
        return (
            jsonify(
                {
                    "device_id": device_id,
                    "ip_address": ip_address,
                    "classification": classification,
                    "confidence": confidence,
                    "action": action,
                    "applied": applied,
                }
            ),
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
        registry.register(device_id, datapath_id, ip_address)
        logger.info("Registered device: device_id=%s datapath_id=%d ip=%s", device_id, datapath_id, ip_address)
        return jsonify({"registered": device_id}), 200

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "connected_datapaths": controller.get_datapath_ids()})

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
