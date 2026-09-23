"""Boosting classifier — cascade stage 1 (component 2).

**Design decision — label source (documented per CLAUDE.md's explicit
requirement not to default into this silently):** this classifier is
trained on a small **server-held labeled calibration set**, not on labels
held by clients. Clients never need to reveal their local attack-type
annotations to the server; the server only needs a modest, independently
curated slice of labeled attack samples (e.g. from a lab testbed or threat
intelligence feed) to bootstrap a first-pass classifier before FL begins.
The tradeoff: the server must have *some* representative labeled data
upfront, and the model's coverage is bounded by what that calibration set
contains — clients' local traffic never directly supervises this stage,
only the client-side autoencoder (component 3) does. This is configured
via `BoostingConfig.label_source` (see configs/config.yaml).

**Cold start:** the model is trained once on the calibration set and
must be reasonably decent *before* round 1 of FL training, since early
rounds' clients filter their local traffic through it to build
autoencoder training data (component 3/4) — a useless bootstrap model
would mean the autoencoder trains on effectively unfiltered garbage.

**Distribution mechanism:** the model is broadcast to clients (not
aggregated via Flower's weight-averaging), so it must be serializable to
bytes and reconstructable client-side — see `to_bytes`/`from_bytes`.

**Design decision — feature scale: this model always trains and predicts
on RAW (unnormalized) features, never per-client-normalized ones.**
Component 1 normalizes per-client (never globally — see
`fl_ids.data.pipeline`), which is correct for the autoencoder (a neural
net that benefits from normalized input) but wrong for this model: a
tree-based classifier doesn't need normalization at all (per-feature
decision splits are invariant to monotonic scaling), and applying a
*different* per-client scaler to input from a shared, centrally-trained
model actively breaks it. An earlier version of this module tried
training the boosting model on a calibration set standardized to
"roughly the same shape" every client's own scaler would also produce,
reasoning that tree splits only need comparable ranges, not identical
scaling — Phase 5's real-data validation showed this reasoning was wrong
in practice: under real non-IID skew, a client whose local shard is
e.g. 94% benign traffic ends up with per-feature statistics dominated by
benign traffic's characteristic value range, nowhere close to the
calibration set's population-level statistics. The same held-out rows
scored ~0.94 accuracy in raw features vs. ~0.06 through that client's own
scaler — not a minor approximation error, a broken model. `X_raw` /
`X_test_raw` (see `fl_ids.data.pipeline.partition_and_normalize_clients`)
exist specifically so this model — both server-side training and
client-side inference via `passes_to_autoencoder` — always sees the same
raw-scale feature representation everywhere.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np

from fl_ids.utils.config import BoostingConfig

logger = logging.getLogger(__name__)


@dataclass
class CascadeStage1Result:
    """Per-sample output of the boosting stage's half of the cascade rule.

    Attributes:
        predicted_class: Boosting's argmax predicted class index per sample.
        confidence: The predicted class's softmax probability.
        is_confident_attack: True where the cascade's final output is
            already decided at this stage (predicted class is a known
            attack type and confidence >= the configured threshold) — see
            CLAUDE.md's "Cascade decision rule".
    """

    predicted_class: np.ndarray
    confidence: np.ndarray
    is_confident_attack: np.ndarray


class BoostingClassifier:
    """LightGBM multi-class attack-type classifier — the cascade's first pass.

    Wraps a `lightgbm.Booster` with the cascade decision rule, and with
    serialization for the server -> client broadcast mechanism (this model
    is not FL-aggregated; it's distributed as a config payload each round
    or every few rounds, per component 2).
    """

    def __init__(
        self,
        config: BoostingConfig,
        num_classes: int,
        benign_class: int,
        seed: int,
    ) -> None:
        """Initialize an untrained classifier.

        Args:
            config: Boosting hyperparameters and cascade confidence threshold.
            num_classes: Number of attack-type classes (including benign).
            benign_class: The integer class index corresponding to "Normal".
            seed: Random seed for training reproducibility.
        """
        self.config = config
        self.num_classes = num_classes
        self.benign_class = benign_class
        self.seed = seed
        self._booster: lgb.Booster | None = None

    @property
    def is_trained(self) -> bool:
        """Whether `train` or `from_bytes` has populated an underlying model."""
        return self._booster is not None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train (or retrain) on labeled attack-type data.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Integer-encoded attack-type labels, shape (n_samples,).
        """
        train_set = lgb.Dataset(X, label=y)
        params = {
            "objective": "multiclass",
            "num_class": self.num_classes,
            "learning_rate": self.config.learning_rate,
            "num_leaves": self.config.num_leaves,
            "seed": self.seed,
            "verbosity": -1,
            # LightGBM's default multi-threaded histogram building is NOT
            # bit-reproducible across runs even with a fixed seed (parallel
            # reduction order varies with OS thread scheduling) -- this
            # combination is LightGBM's own documented recipe for genuine
            # run-to-run determinism. Discovered in Phase 5: without these,
            # the exact same call produced wildly different real-data
            # per-class recall from run to run (e.g. one minority class's
            # recall swung between 0.0 and 0.88 on identical inputs/seed),
            # violating CLAUDE.md's "fixed seeds everywhere" requirement.
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
        }
        logger.info(
            "Training boosting classifier: %d samples, %d classes, %d rounds",
            len(y),
            self.num_classes,
            self.config.num_boost_round,
        )
        self._booster = lgb.train(params, train_set, num_boost_round=self.config.num_boost_round)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return per-class predicted probabilities, shape (n_samples, num_classes).

        Raises:
            RuntimeError: If called before the model is trained or loaded.
        """
        if self._booster is None:
            raise RuntimeError("BoostingClassifier must be trained (or loaded) before predicting")
        return self._booster.predict(X)

    def predict_cascade_stage1(self, X: np.ndarray) -> CascadeStage1Result:
        """Apply the shared cascade decision rule's boosting half.

        This is the single implementation of the boosting-side cascade
        check (see CLAUDE.md's "Cascade decision rule") — component 3/4's
        client-side traffic filtering and component 12's per-stage
        evaluation both call this rather than re-deriving the rule
        independently.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            A `CascadeStage1Result` with one entry per input sample.
        """
        proba = self.predict_proba(X)
        predicted_class = np.argmax(proba, axis=1)
        confidence = proba[np.arange(len(proba)), predicted_class]
        is_confident_attack = (predicted_class != self.benign_class) & (
            confidence >= self.config.confidence_threshold
        )
        return CascadeStage1Result(predicted_class, confidence, is_confident_attack)

    def passes_to_autoencoder(self, X: np.ndarray) -> np.ndarray:
        """Boolean mask of samples that fall through to the autoencoder stage.

        Per the cascade rule, this is everything boosting does *not* call a
        confident known attack: predicted-normal samples, and low-confidence
        attack predictions alike. Component 3/4 use this mask to filter
        local traffic before autoencoder training.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Boolean array, shape (n_samples,); True where the sample passes
            through to the autoencoder.
        """
        return ~self.predict_cascade_stage1(X).is_confident_attack

    def to_bytes(self) -> bytes:
        """Serialize the trained model — the payload broadcast to clients each round.

        Raises:
            RuntimeError: If the model has not been trained yet.
        """
        if self._booster is None:
            raise RuntimeError("Cannot serialize an untrained BoostingClassifier")
        return self._booster.model_to_string().encode("utf-8")

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        config: BoostingConfig,
        num_classes: int,
        benign_class: int,
        seed: int,
    ) -> "BoostingClassifier":
        """Reconstruct a classifier client-side from a broadcast model payload.

        Args:
            data: Bytes produced by `to_bytes` on the server.
            config: Same boosting config as the server (cascade threshold
                must match for consistent client-side filtering).
            num_classes: Number of attack-type classes.
            benign_class: Integer class index corresponding to "Normal".
            seed: Seed to store on the reconstructed instance (unused for
                inference, kept for interface symmetry).

        Returns:
            A `BoostingClassifier` ready for `predict_proba`/`predict_cascade_stage1`.
        """
        instance = cls(config, num_classes, benign_class, seed)
        instance._booster = lgb.Booster(model_str=data.decode("utf-8"))
        return instance
