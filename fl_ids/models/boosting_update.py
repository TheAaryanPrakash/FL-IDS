"""Incremental boosting update from analyst-confirmed alerts (component 2).

The bootstrap boosting model is trained once on the server-held
calibration set. After that, the only way it learns anything new is from
labeled traffic reaching the server, and by design raw traffic doesn't
leave clients. This module relaxes that for one narrow case, standing in
for a security analyst triaging alerts:

1. **Client** (`select_alerts`): on an update round, the client runs the
   global autoencoder it just received (with its own threshold) over its
   local traffic that boosting passed as "normal". Rows it flags are
   alerts. Up to `boosting.alert_budget_per_client` of them are surfaced
   with the label an analyst would confirm: the attack type, or Normal
   for a false alarm. In simulation that label is the row's true label.
2. **Server** (`BoostingUpdater`): takes alerts only from clients that
   survived the trust filter that round (the spec's "surviving clients"),
   adds them to everything surfaced so far, and continues training the
   current model on calibration set + alerts for
   `boosting.update_num_boost_round` more trees (LightGBM `init_model`).
   The new version is broadcast from the next round on.

**One exception: a class the model has never been trained on.** The
bootstrap model gives such a class a probability of ~1e-14 everywhere,
so its softmax hessian p(1-p) is ~0 for every row, and with
`min_sum_hessian_in_leaf` (needed to stop training diverging) no tree for
that class can ever split: `init_model` continuation can't learn it. When
alerts bring a never-trained class, the update retrains from scratch on
calibration set + alerts instead, and records that it did.

What this relaxes, stated plainly: feature rows of flagged traffic leave
the client, capped per round and only on update rounds. Label poisoning
by a malicious client is not defended against beyond the trust filter
(which judges its autoencoder update, not its alerts).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import BoostingConfig

logger = logging.getLogger(__name__)

ALERT_FEATURES_KEY = "alert_features"
ALERT_LABELS_KEY = "alert_labels"
ALERT_COUNT_KEY = "alert_count"
SURFACE_ALERTS_KEY = "surface_alerts"


@dataclass
class SurfacedAlerts:
    """Alert rows one client surfaced: raw features and confirmed labels."""

    X_raw: np.ndarray
    y: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def select_alerts(
    errors: np.ndarray,
    threshold: float,
    X_raw: np.ndarray,
    y: np.ndarray,
    budget: int,
    seed: int,
) -> SurfacedAlerts:
    """Pick the rows a client surfaces: flagged by its autoencoder, capped at `budget`.

    Args:
        errors: Reconstruction errors for the rows boosting passed as "normal".
        threshold: The client's anomaly threshold.
        X_raw: Those rows' raw features.
        y: Their confirmed labels.
        budget: Maximum rows to surface.
        seed: Random seed for sampling when more rows are flagged than the budget.

    Returns:
        The surfaced alerts (possibly empty).
    """
    flagged = np.where(errors > threshold)[0]
    if len(flagged) > budget:
        flagged = np.sort(np.random.default_rng(seed).choice(flagged, budget, replace=False))
    return SurfacedAlerts(X_raw[flagged].astype(np.float32), y[flagged].astype(np.int64))


def encode_alerts(alerts: SurfacedAlerts) -> dict:
    """Pack alerts into Flower fit metrics (bytes and ints only)."""
    return {
        ALERT_FEATURES_KEY: alerts.X_raw.astype(np.float32).tobytes(),
        ALERT_LABELS_KEY: alerts.y.astype(np.int64).tobytes(),
        ALERT_COUNT_KEY: len(alerts),
    }


def decode_alerts(metrics: dict, num_features: int) -> SurfacedAlerts | None:
    """Unpack alerts from fit metrics; None if the client surfaced none this round.

    Raises:
        ValueError: If the payload's size doesn't match `num_features`.
    """
    if int(metrics.get(ALERT_COUNT_KEY, 0)) == 0:
        return None
    X = np.frombuffer(metrics[ALERT_FEATURES_KEY], dtype=np.float32)
    y = np.frombuffer(metrics[ALERT_LABELS_KEY], dtype=np.int64)
    if X.size != len(y) * num_features:
        raise ValueError(f"Alert payload has {X.size} values for {len(y)} rows of {num_features} features")
    return SurfacedAlerts(X.reshape(len(y), num_features).copy(), y.copy())


@dataclass
class BoostingUpdateRecord:
    """What one update did, for the round history and dashboard."""

    round: int
    version: int
    alerts_used: int
    alerts_by_class: dict[str, int]
    alerts_from_excluded_clients: int
    total_alerts_collected: int
    # "continue" (init_model) or "retrain" (alerts brought a class the model
    # had never been trained on; see the module docstring).
    mode: str = "continue"
    new_classes: list[str] = field(default_factory=list)
    metrics: dict | None = None


@dataclass
class BoostingUpdater:
    """Server-side incremental update: calibration set + accumulated alerts, continued with `init_model`.

    Attributes:
        config: Boosting config (`update_every_n_rounds`, `update_num_boost_round`).
        X_calib, y_calib: The server-held calibration set the model was bootstrapped on.
        class_names: Class names in class-index order (for per-class alert counts).
        evaluate: Optional `model -> metrics dict` scoring each new version
            on server-held data (Phase A passes its calibration eval split).
    """

    config: BoostingConfig
    X_calib: np.ndarray
    y_calib: np.ndarray
    class_names: list[str]
    evaluate: object = None
    version: int = 1
    _alerts_X: list[np.ndarray] = field(default_factory=list)
    _alerts_y: list[np.ndarray] = field(default_factory=list)
    _trained_classes: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        self._trained_classes = {int(c) for c in np.unique(self.y_calib)}

    def is_update_round(self, server_round: int) -> bool:
        """Whether clients should surface alerts this round (and the server update after it)."""
        every = self.config.update_every_n_rounds
        return every > 0 and server_round % every == 0

    @property
    def total_alerts(self) -> int:
        return int(sum(len(y) for y in self._alerts_y))

    def update(
        self,
        model: BoostingClassifier,
        server_round: int,
        alerts_by_client: dict[int, SurfacedAlerts | None],
        surviving_clients: set[int],
    ) -> tuple[BoostingClassifier, BoostingUpdateRecord | None]:
        """Fold this round's surviving clients' alerts in and continue training.

        Args:
            model: The boosting model broadcast this round.
            server_round: Current round.
            alerts_by_client: Every client's surfaced alerts this round.
            surviving_clients: Clients that passed the trust filter.

        Returns:
            (model to broadcast next, record) — the same model and None if
            no surviving client surfaced anything.
        """
        used = {cid: a for cid, a in alerts_by_client.items() if a is not None and cid in surviving_clients}
        excluded = sum(len(a) for cid, a in alerts_by_client.items() if a is not None and cid not in surviving_clients)
        if not used:
            logger.info(
                "Round %d: no alerts from surviving clients (%d from excluded clients ignored); boosting unchanged",
                server_round, excluded,
            )
            return model, None

        for alerts in used.values():
            self._alerts_X.append(alerts.X_raw)
            self._alerts_y.append(alerts.y)
        round_y = np.concatenate([a.y for a in used.values()])
        X = np.concatenate([self.X_calib, *self._alerts_X])
        y = np.concatenate([self.y_calib, *self._alerts_y])

        new_classes = sorted({int(c) for c in np.unique(round_y)} - self._trained_classes)
        if new_classes:
            updated = BoostingClassifier(
                model.config, model.num_classes, model.benign_class, model.seed, model.confidence_threshold
            )
            updated.train(X, y)
            self._trained_classes |= set(new_classes)
        else:
            updated = model.continue_training(X, y, self.config.update_num_boost_round)
        self.version += 1
        labels, counts = np.unique(round_y, return_counts=True)
        record = BoostingUpdateRecord(
            round=server_round,
            version=self.version,
            alerts_used=len(round_y),
            alerts_by_class={self.class_names[c]: int(n) for c, n in zip(labels, counts)},
            alerts_from_excluded_clients=excluded,
            total_alerts_collected=self.total_alerts,
            mode="retrain" if new_classes else "continue",
            new_classes=[self.class_names[c] for c in new_classes],
            metrics=self.evaluate(updated) if self.evaluate else None,
        )
        logger.info(
            "Round %d: boosting v%d (%s%s) from %d alerts (%s); %d from excluded clients ignored",
            server_round, self.version, record.mode,
            f", new classes {record.new_classes}" if new_classes else "",
            record.alerts_used, record.alerts_by_class, excluded,
        )
        return updated, record
