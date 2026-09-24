"""Live cascade inference for Phase B: one device's replayed traffic -> one mitigation decision.

Separated from `fl_ids.sdn.topology` so it's testable without root or
Mininet. Uses only a loaded Phase A bundle (`fl_ids.orchestration.artifacts`),
never a model trained on the spot.

Mininet host `h<i+1>` is FL client `i` (component 10: one host per
simulated client), so its traffic is normalized with client `i`'s scaler
and scored against client `i`'s anomaly threshold — the way that client
scores its own traffic.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import numpy as np

from fl_ids.models.cascade import BENIGN_LABEL, cascade_predict
from fl_ids.orchestration.artifacts import PhaseAArtifacts

_HOST_NAME = re.compile(r"h(\d+)")


@dataclass
class DeviceClassification:
    """One device's cascade verdict, in the shape the SDN bridge's `/mitigate` takes (plus diagnostics)."""

    device_id: str
    classification: str
    confidence: float
    stage: str
    client_id: int
    num_packets: int
    num_non_benign: int

    def to_dict(self) -> dict:
        return asdict(self)


def client_id_for_host(host_name: str) -> int:
    """Map Mininet host `h<i+1>` to FL client `i`.

    Raises:
        ValueError: If the name isn't of the form `h<positive int>`.
    """
    match = _HOST_NAME.fullmatch(host_name)
    if not match or int(match.group(1)) < 1:
        raise ValueError(f"Host {host_name!r} isn't a per-client Mininet host (h1, h2, ...)")
    return int(match.group(1)) - 1


def classify_device_traffic(artifacts: PhaseAArtifacts, host_name: str, X_raw: np.ndarray) -> DeviceClassification:
    """Run the cascade over a device's packets and reduce them to one device-level verdict.

    Per-packet outputs are reduced the way the mitigation needs: the most
    frequent non-benign label if any packet got one (its mean confidence,
    and whichever cascade stage made most of those calls), else "benign".

    Args:
        artifacts: The loaded Phase A bundle.
        host_name: The Mininet host the traffic came from.
        X_raw: Raw-scale feature rows, aligned to `artifacts.feature_names`.

    Returns:
        The device's classification.

    Raises:
        ValueError: If there are no packets, or the host maps to a client
            the bundle has no scaler/threshold for.
    """
    if len(X_raw) == 0:
        raise ValueError(f"No packets to classify for {host_name}")
    client_id = client_id_for_host(host_name)
    if client_id not in artifacts.client_thresholds:
        raise ValueError(
            f"{host_name} maps to client {client_id}, but the Phase A bundle only has clients {artifacts.client_ids}"
        )

    output = cascade_predict(
        artifacts.boosting_model,
        artifacts.autoencoder,
        artifacts.client_thresholds[client_id],
        X_raw,
        artifacts.normalize(client_id, X_raw),
        artifacts.class_names,
        artifacts.cascade_config,
    )

    non_benign = output.predicted_label != BENIGN_LABEL
    if non_benign.any():
        labels, counts = np.unique(output.predicted_label[non_benign], return_counts=True)
        classification = str(labels[np.argmax(counts)])
        chosen = non_benign & (output.predicted_label == classification)
    else:
        classification = BENIGN_LABEL
        chosen = np.ones(len(output.predicted_label), dtype=bool)
    stages, stage_counts = np.unique(output.stage[chosen], return_counts=True)

    return DeviceClassification(
        device_id=host_name,
        classification=classification,
        confidence=float(output.confidence[chosen].mean()),
        stage=str(stages[np.argmax(stage_counts)]),
        client_id=client_id,
        num_packets=len(X_raw),
        num_non_benign=int(non_benign.sum()),
    )
