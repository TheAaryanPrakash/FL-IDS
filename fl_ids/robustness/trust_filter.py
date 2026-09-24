"""Cosine-similarity trust filter (component 5) — server-side defense
against poisoned client autoencoder weight updates.

Three requirements carried over from hard lessons in an earlier prototype
(per CLAUDE.md — built in from day one, not discovered the hard way):

1. **Operates on weight deltas (new - old), never raw weights.** Raw
   weights are dominated by the shared global-model component, so cosine
   similarity between any two clients' raw weights is ~0.99 regardless of
   malicious status — the filter would do nothing.
2. **Uses a relative, per-round outlier threshold (median absolute
   deviation), not a fixed cosine cutoff.** A fixed threshold breaks the
   moment non-IID heterogeneity shifts — honest clients can naturally sit
   at 0.3-0.4 similarity under heavy heterogeneity.
3. **Pairs cosine similarity with a norm-clipping check.** Cosine
   similarity is direction-only and misses scaling attacks (same
   direction, blown-up magnitude).

**Exclusion is three checks, not MAD alone.** A client is excluded if
any holds (each recorded as a reason in `TrustFilterResult`):

- `mad_outlier`: its similarity is a MAD outlier within this round.
- `opposes_consensus`: its similarity is below `min_cosine_similarity`
  (default 0): its update points away from the robust consensus
  direction. This is the sign boundary, not a tuned cutoff like the 0.5
  requirement 2 warns about. FLTrust (Cao et al., NDSS 2021) gives such
  updates zero weight via ReLU(cos) for the same reason.
- `low_trust`: its EMA trust score (`TrustTracker`) is below
  `min_trust_score` (default 0.5, i.e. its similarity has averaged below
  0 over its history), so its history counts, not just this round.

Why MAD alone wasn't enough, measured on real data at 20% sign-flip
attackers: honest clients' similarities spread from ~0.2 (heavily
skewed shards) to ~0.8, so the MAD band widened until its cutoff sat at
-0.6 to -0.8, and attackers at -0.2 to -0.3 passed ~78% of rounds.
Every attacker round sat below 0 except two near-zero ones (+0.09,
+0.12), which the trust check catches once history accumulates.

What remains, measured with all three checks on (same setup, 12
rounds): attacker exclusion rose from ~22% to ~62% of attacker-rounds
(one attacker 10/12, the other 5/12). The misses are late rounds: once
the model has converged, heavily-skewed honest clients' deltas are
mostly noise (cosine 0.00-0.07), and an attacker whose true delta is
also noise sends a sign-flipped noise vector, still near-orthogonal
(+0.02 to +0.11). No direction test can separate those, and such an
update barely moves the model; its magnitude is bounded by the norm
clip and the trimmed mean. The cost: honest near-zero clients are
excluded in ~9% of client-rounds (was ~2%) when their noise lands just
below 0.

**Documented limitation (per CLAUDE.md, stated plainly, not hidden):**
this defense targets malicious *update-sending* behavior (e.g.
sign-flipping), not malicious local *data*. A client whose local data is
simply unusual looks, to an unsupervised autoencoder that never sees
labels, just like another differently-distributed honest client under
non-IID heterogeneity — cosine similarity alone cannot reliably
distinguish "malicious data" from "honest non-IID." The sign-flip /
gradient-ascent attacker this module is validated against (see
`fl_ids.robustness.attackers`) corrupts the update itself, which is the
right test case for what this component can actually defend against.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from fl_ids.utils.config import RobustnessConfig

logger = logging.getLogger(__name__)


def flatten_weights(weights: list[np.ndarray]) -> np.ndarray:
    """Flatten a list of weight arrays (one model's full parameter set) into one 1D vector."""
    return np.concatenate([w.ravel() for w in weights])


def compute_delta(new_weights: list[np.ndarray], old_weights: list[np.ndarray]) -> np.ndarray:
    """A client's weight delta (new - old), flattened to a single vector.

    Operating on deltas — never raw weights — is requirement 1 above.
    """
    return flatten_weights(new_weights) - flatten_weights(old_weights)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors; 0.0 if either is exactly zero."""
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


@dataclass
class TrustFilterResult:
    """Per-round trust-filtering output for one set of client deltas.

    Attributes:
        client_ids: Client IDs, in the same order as `similarities`/`norms`.
        similarities: Each client's cosine similarity to the round's
            reference direction (the coordinate-wise median delta).
        norms: Each client's raw (pre-clip) delta L2 norm.
        is_outlier: True where a client's similarity is a MAD-based
            statistical outlier this round (one exclusion reason of three).
        clipped_deltas: {client_id: delta}, norm-clipped to
            `norm_clip_multiplier` times the round's median norm — only
            for surviving clients.
        survivors: Client IDs that passed every exclusion check.
        exclusion_reasons: {client_id: reasons} for excluded clients only
            (`mad_outlier`, `opposes_consensus`, `low_trust`).
        trust_scores: {client_id: EMA trust after this round}; empty when
            no tracker was given.
    """

    client_ids: list[int]
    similarities: np.ndarray
    norms: np.ndarray
    is_outlier: np.ndarray
    clipped_deltas: dict[int, np.ndarray]
    survivors: list[int]
    exclusion_reasons: dict[int, list[str]] = field(default_factory=dict)
    trust_scores: dict[int, float] = field(default_factory=dict)

    @property
    def excluded(self) -> list[int]:
        """Client IDs excluded from aggregation this round, for any reason."""
        return list(self.exclusion_reasons)


def filter_client_deltas(
    client_deltas: dict[int, np.ndarray],
    config: RobustnessConfig,
    trust_tracker: TrustTracker | None = None,
) -> TrustFilterResult:
    """Screen client weight deltas for poisoning: direction and trust checks, then norm clip.

    The reference direction is the coordinate-wise *median* delta across
    all of this round's clients — robust to a minority of amplified
    malicious updates, unlike a plain mean (which a sufficiently amplified
    attacker could skew, ironically making honest clients look like the
    outliers).

    Only unusually *low* similarity is flagged (one-sided): cosine
    similarity is capped at 1.0, so there's no meaningful "too similar to
    the crowd" outlier case — only "direction diverges from the crowd."

    A client is excluded if its similarity is a MAD outlier this round,
    is below `min_cosine_similarity`, or (with a tracker) its trust score
    after folding in this round is below `min_trust_score` — see the
    module docstring for why all three.

    Args:
        client_deltas: {client_id: flattened weight delta} for this round.
        config: `norm_clip_multiplier`, `mad_outlier_threshold`,
            `min_cosine_similarity`, `min_trust_score` (None disables
            either of the last two).
        trust_tracker: Updated in place with this round's similarities;
            its scores drive the `low_trust` check. None skips that check.

    Returns:
        A `TrustFilterResult` with per-client diagnostics and the set of
        surviving client IDs, with their deltas norm-clipped.
    """
    client_ids = list(client_deltas.keys())
    deltas = np.stack([client_deltas[cid] for cid in client_ids])

    reference = np.median(deltas, axis=0)
    similarities = np.array([cosine_similarity(deltas[i], reference) for i in range(len(client_ids))])
    norms = np.linalg.norm(deltas, axis=1)

    median_sim = np.median(similarities)
    mad = np.median(np.abs(similarities - median_sim))
    if mad == 0.0:
        # Degenerate round (e.g. all clients ~identical, or too few of
        # them) -- fall back to a small epsilon so a single genuine
        # outlier still gets flagged rather than everyone passing/failing
        # together on a zero-width band.
        mad = 1e-6
    is_outlier = similarities < (median_sim - config.mad_outlier_threshold * mad)

    trust_scores = trust_tracker.update(client_ids, similarities) if trust_tracker is not None else {}

    exclusion_reasons: dict[int, list[str]] = {}
    for i, cid in enumerate(client_ids):
        reasons = []
        if is_outlier[i]:
            reasons.append("mad_outlier")
        if config.min_cosine_similarity is not None and similarities[i] < config.min_cosine_similarity:
            reasons.append("opposes_consensus")
        if config.min_trust_score is not None and cid in trust_scores and trust_scores[cid] < config.min_trust_score:
            reasons.append("low_trust")
        if reasons:
            exclusion_reasons[cid] = reasons

    median_norm = np.median(norms)
    clip_norm = config.norm_clip_multiplier * median_norm

    clipped_deltas: dict[int, np.ndarray] = {}
    survivors: list[int] = []
    for i, cid in enumerate(client_ids):
        if cid in exclusion_reasons:
            continue
        delta = deltas[i]
        if clip_norm > 0 and norms[i] > clip_norm:
            delta = delta * (clip_norm / norms[i])
        clipped_deltas[cid] = delta
        survivors.append(cid)

    return TrustFilterResult(
        client_ids, similarities, norms, is_outlier, clipped_deltas, survivors, exclusion_reasons, trust_scores
    )


class TrustTracker:
    """Running per-client trust score (EMA across rounds), not just a one-round judgment.

    Each round's raw cosine similarity (range [-1, 1]) is rescaled to
    [0, 1] for a friendlier "trust" semantic (0 = max distrust, 1 = max
    trust) before being folded into the EMA. `ema_alpha` is the weight on
    the new round's observation vs. running history (config default 0.3:
    30% new, 70% history).

    New clients start at full trust (1.0) — "innocent until proven
    guilty" — rather than jumping straight to their first raw round
    score. This matters for the visible-decay behavior this component is
    meant to show: a client attacking from round 1 has real trust to lose
    across rounds instead of landing at its steady-state score
    immediately, and an honest client's score converging from 1.0 down to
    its natural ~0.7-0.9 level still reads as "stays high" throughout.
    """

    INITIAL_TRUST = 1.0

    def __init__(self, ema_alpha: float) -> None:
        self.ema_alpha = ema_alpha
        self.scores: dict[int, float] = {}

    def update(self, client_ids: list[int], similarities: np.ndarray) -> dict[int, float]:
        """Update (or initialize) and return each given client's trust score.

        Args:
            client_ids: Client IDs observed this round.
            similarities: Matching cosine similarities to the round's
                reference direction.

        Returns:
            {client_id: current trust score}, for exactly the clients
            passed in.
        """
        for cid, sim in zip(client_ids, similarities):
            round_score = (float(sim) + 1.0) / 2.0
            prior = self.scores.get(cid, self.INITIAL_TRUST)
            self.scores[cid] = self.ema_alpha * round_score + (1 - self.ema_alpha) * prior
        return {cid: self.scores[cid] for cid in client_ids}
