"""Shared boosting setup for the real-data FL integration tests.

Both real-data FL integration tests (Phase 3's and Phase 4's milestones,
re-verified on real data in Phase 5) need a bootstrap boosting model
whose client-side filter actually works -- otherwise the autoencoder
trains on unfiltered (or wrongly filtered) traffic and the milestone
being re-verified isn't the real pipeline's. These tests previously each
hard-coded their own `num_leaves=63`/`num_boost_round=150`/threshold 0.6,
which on real data turned out degenerate in seed-dependent ways: at one
seed it passed 56% of attack traffic through to the autoencoder, at
another it rejected 74% of *benign* traffic. So the hyperparameters and
cascade threshold come from `configs/config.yaml` here -- the same
values the real pipeline and the Phase 6 evaluation use -- and
`assert_boosting_filter_is_real` guards against a degenerate filter
silently slipping back in.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from fl_ids.models.boosting import BoostingClassifier
from fl_ids.utils.config import load_config

PRODUCTION_CONFIG = load_config()

# Bounds on the bootstrap filter, set on these tests' real-data pools
# (seeds 555 and 777). The pass-through rates originally measured here
# (1-18% of attack rows, >=99% of benign) predate the leaf-regularization
# fix to boosting's training divergence; the bounds still hold after it.
MAX_ATTACK_PASS_THROUGH = 0.3
MIN_BENIGN_PASS_THROUGH = 0.9


def boosting_config_section() -> dict:
    """The `boosting:` section of a test config YAML, from the production config."""
    return asdict(PRODUCTION_CONFIG.boosting)


def cascade_config_section() -> dict:
    """The `cascade:` section of a test config YAML, from the production config.

    Clients read `confidence_threshold` from here to decide what passes
    their boosting filter, so it must match the server-side model's.
    """
    cascade = PRODUCTION_CONFIG.cascade
    return {
        "confidence_threshold": cascade.confidence_threshold,
        "anomaly_confidence_clip": list(cascade.anomaly_confidence_clip),
    }


def train_bootstrap_boosting(
    X_calib: np.ndarray, y_calib: np.ndarray, num_classes: int, benign_class: int, seed: int
) -> BoostingClassifier:
    """Train the server-side bootstrap boosting model with production hyperparameters."""
    model = BoostingClassifier(
        PRODUCTION_CONFIG.boosting,
        num_classes=num_classes,
        benign_class=benign_class,
        seed=seed,
        confidence_threshold=PRODUCTION_CONFIG.cascade.confidence_threshold,
    )
    model.train(X_calib, y_calib)
    return model


def assert_boosting_filter_is_real(
    model: BoostingClassifier, X_raw: np.ndarray, y: np.ndarray, benign_class: int
) -> None:
    """Assert the boosting filter keeps attacks away from, and lets benign traffic through to, the autoencoder.

    Args:
        model: The bootstrap boosting model clients will filter with.
        X_raw: Raw-scale rows disjoint from the model's training data
            (e.g. the federated pool).
        y: True labels for `X_raw`.
        benign_class: Integer class index corresponding to "Normal".
    """
    passes = model.passes_to_autoencoder(X_raw)
    attack_pass_through = float(passes[y != benign_class].mean())
    benign_pass_through = float(passes[y == benign_class].mean())
    assert attack_pass_through <= MAX_ATTACK_PASS_THROUGH, (
        f"Boosting filter passes {attack_pass_through:.1%} of attack traffic to the autoencoder "
        f"(max {MAX_ATTACK_PASS_THROUGH:.0%}) -- the autoencoder would train on attacks"
    )
    assert benign_pass_through >= MIN_BENIGN_PASS_THROUGH, (
        f"Boosting filter passes only {benign_pass_through:.1%} of benign traffic to the autoencoder "
        f"(min {MIN_BENIGN_PASS_THROUGH:.0%}) -- the autoencoder would be starved of benign data"
    )
