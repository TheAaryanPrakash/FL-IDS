"""The Phase A model bundle: what Phase A saves and Phase B loads (component 11).

A bundle is a directory:

- `boosting_model.txt` — the LightGBM model string clients filtered with
  (`BoostingClassifier.to_bytes`).
- `autoencoder_weights.npz` — the final global autoencoder weights.
- `client_normalization.npz` — each client's scaler (`mean_<cid>`,
  `scale_<cid>`); the z-score clip is in the manifest.
- `manifest.json` — everything needed to use the above without the
  training-time config: class/feature names, autoencoder architecture,
  cascade decision-rule settings, each client's anomaly threshold, the
  held-out test metrics, and provenance (seed, rounds, git commit, time).

**Why per-client scalers and thresholds.** Every client normalizes its
own traffic with a scaler fit on its own data (component 1) and
calibrates its own anomaly threshold on its own benign traffic
(component 3); there's no global scaler or threshold to save. Phase B
maps Mininet host `h<i+1>` to FL client `i` (one host per simulated
client, component 10), so a host's live traffic is scored the way its
client scores its own traffic.

The bundle is self-describing on purpose: Phase B builds the autoencoder
and the cascade rule from the manifest, not from whatever
`configs/config.yaml` says at load time, so a later config edit can't
silently pair the saved weights with a different architecture or
threshold.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fl_ids.data.pipeline import normalize_with_scaler
from fl_ids.models.autoencoder import Autoencoder, set_weights
from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import BoostingConfig, CascadeConfig

logger = logging.getLogger(__name__)

BOOSTING_FILE = "boosting_model.txt"
AUTOENCODER_FILE = "autoencoder_weights.npz"
NORMALIZATION_FILE = "client_normalization.npz"
MANIFEST_FILE = "manifest.json"


@dataclass
class PhaseAArtifacts:
    """A loaded (or about-to-be-saved) Phase A model bundle."""

    boosting_model: BoostingClassifier
    autoencoder: Autoencoder
    autoencoder_weights: list[np.ndarray]
    class_names: list[str]
    benign_class: int
    feature_names: list[str]
    cascade_config: CascadeConfig
    # {client_id: (mean, scale)} -- StandardScaler's parameters.
    client_scalers: dict[int, tuple[np.ndarray, np.ndarray]]
    client_thresholds: dict[int, float]
    manifest: dict = field(default_factory=dict)
    # z-score clip the clients trained with (data.normalized_clip).
    normalized_clip: float | None = None

    @property
    def client_ids(self) -> list[int]:
        """Client IDs the bundle has a scaler and threshold for."""
        return sorted(self.client_thresholds)

    def normalize(self, client_id: int, X_raw: np.ndarray) -> np.ndarray:
        """Normalize raw rows the way `client_id` normalizes its own traffic.

        Raises:
            KeyError: If the bundle has no scaler for `client_id`.
        """
        mean, scale = self.client_scalers[client_id]
        return normalize_with_scaler(X_raw, mean, scale, self.normalized_clip)


def save_phase_a_artifacts(
    artifact_dir: str | Path,
    boosting_model: BoostingClassifier,
    autoencoder_weights: list[np.ndarray],
    class_names: list[str],
    benign_class: int,
    feature_names: list[str],
    hidden_dims: list[int],
    bottleneck_dim: int,
    cascade_config: CascadeConfig,
    client_scalers: dict[int, tuple[np.ndarray, np.ndarray]],
    client_thresholds: dict[int, float],
    normalized_clip: float | None,
    extra_manifest: dict | None = None,
) -> Path:
    """Write a Phase A model bundle.

    Args:
        artifact_dir: Directory to write (created if missing; existing
            bundle files are overwritten).
        boosting_model: The trained boosting model.
        autoencoder_weights: Final global autoencoder weights.
        class_names: Class names in class-index order.
        benign_class: Integer class index corresponding to "Normal".
        feature_names: Feature column names, in model input order.
        hidden_dims: Autoencoder hidden layer widths.
        bottleneck_dim: Autoencoder bottleneck width.
        cascade_config: The cascade decision rule the models were evaluated with.
        client_scalers: `{client_id: (mean, scale)}`.
        client_thresholds: `{client_id: anomaly threshold}`.
        normalized_clip: The z-score clip clients normalized with (None: unclipped).
        extra_manifest: Merged into `manifest.json` (metrics, provenance).

    Returns:
        The bundle directory.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    (artifact_dir / BOOSTING_FILE).write_bytes(boosting_model.to_bytes())
    np.savez(artifact_dir / AUTOENCODER_FILE, *autoencoder_weights)
    normalization = {}
    for cid, (mean, scale) in client_scalers.items():
        normalization[f"mean_{cid}"] = mean
        normalization[f"scale_{cid}"] = scale
    np.savez(artifact_dir / NORMALIZATION_FILE, **normalization)

    manifest = {
        "class_names": list(class_names),
        "benign_class": int(benign_class),
        "feature_names": list(feature_names),
        "input_dim": len(feature_names),
        "autoencoder": {"hidden_dims": list(hidden_dims), "bottleneck_dim": int(bottleneck_dim)},
        "cascade": {
            "confidence_threshold": cascade_config.confidence_threshold,
            "anomaly_confidence_clip": list(cascade_config.anomaly_confidence_clip),
        },
        "normalized_clip": normalized_clip,
        # JSON keys are strings; load_phase_a_artifacts converts back.
        "client_thresholds": {str(cid): float(t) for cid, t in client_thresholds.items()},
        **(extra_manifest or {}),
    }
    (artifact_dir / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2))
    logger.info("Saved Phase A bundle to %s (%d clients)", artifact_dir, len(client_thresholds))
    return artifact_dir


def load_phase_a_artifacts(artifact_dir: str | Path, boosting_config: BoostingConfig) -> PhaseAArtifacts:
    """Load a Phase A model bundle.

    Args:
        artifact_dir: Directory written by `save_phase_a_artifacts`.
        boosting_config: Only stored on the loaded classifier for interface
            symmetry (training hyperparameters play no part in inference).

    Returns:
        The bundle, with the autoencoder rebuilt from the manifest's architecture.

    Raises:
        FileNotFoundError: If a bundle file is missing.
        ValueError: If the files disagree (weights vs architecture, or
            scalers vs thresholds vs feature count).
    """
    artifact_dir = Path(artifact_dir)
    missing = [f for f in (BOOSTING_FILE, AUTOENCODER_FILE, NORMALIZATION_FILE, MANIFEST_FILE)
               if not (artifact_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Phase A bundle {artifact_dir} is missing {missing} -- run Phase A first")

    manifest = json.loads((artifact_dir / MANIFEST_FILE).read_text())
    cascade_config = CascadeConfig(
        confidence_threshold=manifest["cascade"]["confidence_threshold"],
        anomaly_confidence_clip=tuple(manifest["cascade"]["anomaly_confidence_clip"]),
    )
    class_names = manifest["class_names"]
    benign_class = manifest["benign_class"]

    boosting_model = BoostingClassifier.from_bytes(
        (artifact_dir / BOOSTING_FILE).read_bytes(),
        boosting_config,
        num_classes=len(class_names),
        benign_class=benign_class,
        seed=manifest.get("seed", 0),
        confidence_threshold=cascade_config.confidence_threshold,
    )

    with np.load(artifact_dir / AUTOENCODER_FILE) as npz:
        # np.savez names positional arrays arr_0, arr_1, ...; read them in
        # numeric order (a sorted listing would put arr_10 before arr_2).
        weights = [npz[f"arr_{i}"] for i in range(len(npz.files))]
    architecture = manifest["autoencoder"]
    autoencoder = Autoencoder(manifest["input_dim"], architecture["hidden_dims"], architecture["bottleneck_dim"])
    expected = len(autoencoder.state_dict())
    if len(weights) != expected:
        raise ValueError(f"Saved autoencoder has {len(weights)} weight arrays, the manifest's architecture {expected}")
    try:
        set_weights(autoencoder, weights)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Saved autoencoder weights don't match the manifest's architecture: {exc}") from exc
    autoencoder.eval()

    thresholds = {int(cid): float(t) for cid, t in manifest["client_thresholds"].items()}
    with np.load(artifact_dir / NORMALIZATION_FILE) as npz:
        scalers = {cid: (npz[f"mean_{cid}"], npz[f"scale_{cid}"]) for cid in thresholds if f"mean_{cid}" in npz.files}
    if set(scalers) != set(thresholds):
        raise ValueError(f"Clients with thresholds {sorted(thresholds)} but scalers {sorted(scalers)}")
    for cid, (mean, _scale) in scalers.items():
        if mean.shape != (manifest["input_dim"],):
            raise ValueError(f"Client {cid}'s scaler has {mean.shape} features, expected {manifest['input_dim']}")

    return PhaseAArtifacts(
        boosting_model=boosting_model,
        autoencoder=autoencoder,
        autoencoder_weights=weights,
        class_names=class_names,
        benign_class=benign_class,
        feature_names=manifest["feature_names"],
        cascade_config=cascade_config,
        client_scalers=scalers,
        client_thresholds=thresholds,
        manifest=manifest,
        normalized_clip=manifest["normalized_clip"],
    )
