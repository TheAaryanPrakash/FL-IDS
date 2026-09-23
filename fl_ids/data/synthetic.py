"""Synthetic labeled attack-type data generator.

Used to bootstrap and unit-test the boosting classifier (Phase 2) and the
core FL loop (Phase 3) before real Edge-IIoTset data is wired in (Phase 5),
per CLAUDE.md's phased plan: "Component 2, on synthetic labeled data
initially" and "Use a small synthetic dataset for [Phase 3] specifically
so you're debugging the FL plumbing separate from data pipeline ...
issues." Class imbalance mirrors the real dataset's heavy benign-traffic
skew (~73% Normal) so cold-start bootstrap testing sees a realistic shape.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification

# Mirrors the real Edge-IIoTset Attack_type classes, with "Normal" as the
# benign class at index 0.
DEFAULT_CLASS_NAMES: list[str] = [
    "Normal",
    "DDoS_UDP",
    "DDoS_ICMP",
    "DDoS_TCP",
    "DDoS_HTTP",
    "SQL_injection",
    "Password",
    "Vulnerability_scanner",
    "Uploading",
    "Backdoor",
    "Port_Scanning",
    "XSS",
    "Ransomware",
    "MITM",
    "Fingerprinting",
]

# Roughly mirrors the real dataset's per-class share (Normal ~73%, long
# tail of attack types); sums to 1.0.
DEFAULT_CLASS_WEIGHTS: list[float] = [
    0.55,
    0.06,
    0.06,
    0.03,
    0.03,
    0.03,
    0.03,
    0.03,
    0.02,
    0.02,
    0.02,
    0.02,
    0.02,
    0.02,
    0.01,
]


def make_synthetic_attack_dataset(
    n_samples: int,
    n_features: int = 20,
    class_names: list[str] | None = None,
    class_weights: list[float] | None = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate a synthetic, class-imbalanced multi-class tabular dataset.

    Args:
        n_samples: Total number of samples to generate.
        n_features: Number of numeric features (must be >= 4).
        class_names: Attack-type class names, in class-index order. Index 0
            must be the benign class. Defaults to `DEFAULT_CLASS_NAMES`.
        class_weights: Per-class sample proportions, matching `class_names`
            in length and summing to ~1.0. Defaults to
            `DEFAULT_CLASS_WEIGHTS`.
        seed: Random seed, for reproducibility.

    Returns:
        (X, y, class_names) — `y` is integer-encoded in `class_names`
        order, with class 0 ("Normal" by default) as the benign class.
    """
    if n_features < 4:
        raise ValueError("n_features must be >= 4 for make_classification")

    class_names = class_names if class_names is not None else DEFAULT_CLASS_NAMES
    class_weights = class_weights if class_weights is not None else DEFAULT_CLASS_WEIGHTS
    if len(class_weights) != len(class_names):
        raise ValueError("class_weights and class_names must be the same length")

    n_informative = min(n_features, 15)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=len(class_names),
        n_clusters_per_class=1,
        weights=class_weights,
        flip_y=0.01,
        random_state=seed,
    )
    return X.astype(np.float32), y.astype(np.int64), class_names
