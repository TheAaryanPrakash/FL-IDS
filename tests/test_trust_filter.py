"""Tests for the cosine-similarity trust filter (component 5)."""

from __future__ import annotations

import numpy as np
import pytest

from fl_ids.robustness.trust_filter import (
    TrustTracker,
    compute_delta,
    cosine_similarity,
    filter_client_deltas,
)
from fl_ids.utils.config import RobustnessConfig


def _robustness_config(**overrides) -> RobustnessConfig:
    defaults = dict(trim_fraction=0.15, norm_clip_multiplier=2.0, mad_outlier_threshold=3.0, trust_ema_alpha=0.3)
    defaults.update(overrides)
    return RobustnessConfig(**defaults)


def test_compute_delta_flattens_and_subtracts():
    old = [np.array([1.0, 2.0]), np.array([[3.0, 4.0]])]
    new = [np.array([1.5, 2.5]), np.array([[3.5, 4.5]])]
    delta = compute_delta(new, old)
    assert np.allclose(delta, [0.5, 0.5, 0.5, 0.5])


def test_cosine_similarity_basic_cases():
    a = np.array([1.0, 0.0])
    assert cosine_similarity(a, np.array([1.0, 0.0])) == pytest.approx(1.0)
    assert cosine_similarity(a, np.array([-1.0, 0.0])) == pytest.approx(-1.0)
    assert cosine_similarity(a, np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero():
    assert cosine_similarity(np.array([0.0, 0.0]), np.array([1.0, 1.0])) == 0.0
    assert cosine_similarity(np.array([1.0, 1.0]), np.array([0.0, 0.0])) == 0.0


def test_filter_honest_similar_deltas_mostly_survive_across_many_rounds():
    """MAD-based detection on tightly-concentrated high-dimensional cosine
    similarities has a real, small false-positive rate at small n (a
    known property of the method, not a bug) — a single round can
    occasionally flag a genuinely honest client. The milestone's own
    wording ("excludes the attacker in *most* rounds") reflects this;
    the meaningful claim is that honest clients survive the *large
    majority* of rounds, checked here over many independent trials.
    """
    config = _robustness_config()
    survival_counts = {i: 0 for i in range(10)}
    n_trials = 30

    for trial in range(n_trials):
        rng = np.random.default_rng(trial)
        base = rng.normal(0, 1, size=50)
        deltas = {i: base + rng.normal(0, 0.3, size=50) for i in range(10)}
        result = filter_client_deltas(deltas, config)
        for cid in result.survivors:
            survival_counts[cid] += 1

    survival_rates = np.array([count / n_trials for count in survival_counts.values()])
    assert survival_rates.mean() > 0.85, f"honest clients should survive most rounds, got rates={survival_rates}"


def test_filter_flags_sign_flipped_amplified_attacker():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, size=50)
    deltas = {i: base + rng.normal(0, 0.05, size=50) for i in range(5)}
    deltas[99] = -base * 5.0  # sign-flipped, amplified attacker

    result = filter_client_deltas(deltas, _robustness_config())

    idx_99 = result.client_ids.index(99)
    assert result.is_outlier[idx_99]
    assert 99 not in result.survivors
    assert set(result.survivors) == {0, 1, 2, 3, 4}


def test_reference_direction_is_median_not_mean_robust_to_amplified_outlier():
    # A single extreme outlier should not corrupt the reference direction
    # (and thus flag the honest majority instead) if the reference is a
    # median, unlike a plain mean.
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, size=30)
    deltas = {i: base + rng.normal(0, 0.02, size=30) for i in range(7)}
    deltas[999] = -base * 20.0  # extreme amplified outlier

    result = filter_client_deltas(deltas, _robustness_config())

    # The honest majority should still survive despite one extreme outlier.
    honest_ids = set(range(7))
    assert honest_ids.issubset(set(result.survivors))
    assert 999 not in result.survivors


def test_norm_clipping_scales_down_large_but_similar_direction_update():
    rng = np.random.default_rng(3)
    base = rng.normal(0, 1, size=40)
    deltas = {i: base + rng.normal(0, 0.02, size=40) for i in range(5)}
    deltas[5] = base * 10.0  # same direction, much larger magnitude

    config = _robustness_config(norm_clip_multiplier=2.0, mad_outlier_threshold=10.0)  # avoid MAD-excluding it
    result = filter_client_deltas(deltas, config)

    assert 5 in result.survivors
    median_norm = np.median(result.norms)
    clipped_norm = np.linalg.norm(result.clipped_deltas[5])
    assert clipped_norm == pytest.approx(config.norm_clip_multiplier * median_norm, rel=1e-3)
    assert clipped_norm < result.norms[result.client_ids.index(5)]


def test_trust_tracker_ema_update_formula():
    tracker = TrustTracker(ema_alpha=0.3)
    scores_round1 = tracker.update([0], np.array([1.0]))  # similarity 1.0 -> round_score 1.0
    assert scores_round1[0] == pytest.approx(1.0)  # first observation: no history yet

    scores_round2 = tracker.update([0], np.array([-1.0]))  # similarity -1.0 -> round_score 0.0
    expected = 0.3 * 0.0 + 0.7 * 1.0
    assert scores_round2[0] == pytest.approx(expected)


def test_trust_tracker_new_client_starts_at_full_trust_and_decays_under_attack():
    tracker = TrustTracker(ema_alpha=0.3)
    scores_round1 = tracker.update([0], np.array([-1.0]))  # attacks from round 1
    assert scores_round1[0] == pytest.approx(0.3 * 0.0 + 0.7 * 1.0)  # EMA from a prior of 1.0, not straight to 0.0

    prev = scores_round1[0]
    for _ in range(5):
        scores = tracker.update([0], np.array([-1.0]))
        assert scores[0] < prev  # visibly decaying round over round
        prev = scores[0]


def test_trust_tracker_honest_stays_high_attacker_decays():
    tracker = TrustTracker(ema_alpha=0.3)
    honest_id, attacker_id = 0, 1

    for _ in range(8):
        tracker.update([honest_id, attacker_id], np.array([0.8, -0.7]))

    assert tracker.scores[honest_id] > 0.7
    assert tracker.scores[attacker_id] < 0.3
    assert tracker.scores[honest_id] - tracker.scores[attacker_id] > 0.4


def test_filter_client_deltas_empty_raises_on_stack():
    with pytest.raises(ValueError):
        filter_client_deltas({}, _robustness_config())
