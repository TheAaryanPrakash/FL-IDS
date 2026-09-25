"""Tests for row-level checkpointing of long evaluation runs (component 12)."""

from __future__ import annotations

import pytest

import fl_ids.eval.zero_day as zero_day_module
from fl_ids.data.synthetic import make_synthetic_attack_dataset
from fl_ids.eval.checkpoint import RowCheckpoint, run_fingerprint
from fl_ids.eval.variants import FULL_PIPELINE
from fl_ids.eval.zero_day import run_zero_day_experiment, zero_day_detection_by_run
from tests.test_eval_sweeps import _config

BENIGN = 0
KEY = ("run", "holdout_class")


def test_rows_survive_a_restart(tmp_path):
    path = tmp_path / "ck.csv"
    first = RowCheckpoint(path, "abc", KEY)
    first.append({"run": "a", "holdout_class": "XSS", "detection_rate": 0.5})

    reopened = RowCheckpoint(path, "abc", KEY)
    assert reopened.get("a", "XSS") == {"run": "a", "holdout_class": "XSS", "detection_rate": 0.5}
    assert reopened.get("a", "MITM") is None


def test_rows_from_a_different_run_are_refused(tmp_path):
    path = tmp_path / "ck.csv"
    RowCheckpoint(path, "old-code", KEY).append({"run": "a", "holdout_class": "XSS", "detection_rate": 0.5})
    with pytest.raises(ValueError, match="different run"):
        RowCheckpoint(path, "new-code", KEY)


def test_fingerprint_changes_with_anything_that_changes_results():
    config = _config(num_clients=5, poisoning_fractions=[0.0])
    base = run_fingerprint(config, 20, 42, poisoning_fraction=0.3)
    assert base == run_fingerprint(config, 20, 42, poisoning_fraction=0.3)
    assert base != run_fingerprint(config, 10, 42, poisoning_fraction=0.3)
    assert base != run_fingerprint(config, 20, 7, poisoning_fraction=0.3)
    assert base != run_fingerprint(config, 20, 42, poisoning_fraction=0.2)
    config.robustness.trim_fraction = 0.2
    assert base != run_fingerprint(config, 20, 42, poisoning_fraction=0.3)


def _count_trainings(monkeypatch) -> list:
    calls = []
    real = zero_day_module.train_variant

    def counting(*args, **kwargs):
        calls.append(args[0].name)
        return real(*args, **kwargs)

    monkeypatch.setattr(zero_day_module, "train_variant", counting)
    return calls


def test_a_killed_zero_day_run_resumes_without_retraining_finished_rows(tmp_path, monkeypatch):
    X, y, class_names = make_synthetic_attack_dataset(n_samples=4000, n_features=12, seed=0)
    config = _config(num_clients=4, poisoning_fractions=[0.0])
    config.evaluation.zero_day_holdout_classes = ["DDoS_TCP", "Uploading"]
    config.evaluation.zero_day_max_holdout_rows = 100
    runs = [("clean", FULL_PIPELINE, 0.0), ("poisoned", FULL_PIPELINE, 0.25)]
    path = tmp_path / "ck.csv"
    args = (runs, X, y, class_names, BENIGN, config, 2, 0)

    full = zero_day_detection_by_run(*args, checkpoint=RowCheckpoint(path, "fp", KEY))

    # Simulate a kill after 3 of the 4 rows: keep the header + first 3 rows.
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines[:4]) + "\n")
    calls = _count_trainings(monkeypatch)
    resumed = zero_day_detection_by_run(*args, checkpoint=RowCheckpoint(path, "fp", KEY))

    assert len(calls) == 1  # only the missing row was retrained
    assert resumed.equals(full)


def test_standalone_zero_day_resumes_per_holdout_class(tmp_path, monkeypatch):
    X, y, class_names = make_synthetic_attack_dataset(n_samples=4000, n_features=12, seed=0)
    config = _config(num_clients=4, poisoning_fractions=[0.0])
    config.evaluation.zero_day_holdout_classes = ["DDoS_TCP", "Uploading"]
    config.evaluation.zero_day_max_holdout_rows = 100
    path = tmp_path / "ck.csv"

    full = run_zero_day_experiment(
        X, y, class_names, BENIGN, config, num_rounds=2, seed=0,
        checkpoint=RowCheckpoint(path, "fp", ("holdout_class",)),
    )
    calls = _count_trainings(monkeypatch)
    resumed = run_zero_day_experiment(
        X, y, class_names, BENIGN, config, num_rounds=2, seed=0,
        checkpoint=RowCheckpoint(path, "fp", ("holdout_class",)),
    )

    assert calls == []
    assert resumed[["holdout_class", "cascade_detection_rate"]].equals(full[["holdout_class", "cascade_detection_rate"]])
