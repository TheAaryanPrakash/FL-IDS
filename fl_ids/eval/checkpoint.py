"""Row-level checkpointing for long evaluation runs (component 12).

The zero-day loops retrain once per (run, held-out class): 70 trainings of
~6-9 minutes each for the ablation or the poisoning sweep. Their results
used to live only in memory until the end, so a run killed at 53 of 70
(as happened when the machine ran low on memory) lost ~8 hours.
`RowCheckpoint` appends each finished row to a CSV as soon as it exists,
and a restarted run skips the rows it already has.

**Why a fingerprint.** A checkpoint is only valid for the exact code,
config, round count and seed that produced it. Every row carries a
fingerprint of those, and a checkpoint holding rows with a different
fingerprint is refused outright rather than silently mixed into new
results. The fingerprint includes the git commit, plus a hash of
uncommitted changes under `fl_ids/` and `configs/`.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import subprocess
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fl_ids.utils.config import Config

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
FINGERPRINT_COLUMN = "checkpoint_fingerprint"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return ""


def run_fingerprint(config: Config, num_rounds: int, seed: int, **run_parameters) -> str:
    """Short hash identifying the code, config, round count, seed and other parameters of a run.

    Args:
        config: The full config the run uses (after any CLI overrides).
        num_rounds: FL rounds per training.
        seed: Random seed.
        **run_parameters: Anything else that changes results (e.g. the
            ablation's poisoning fraction).

    Returns:
        A 16-hex-character fingerprint.
    """
    parts = {
        "config": asdict(config),
        "num_rounds": num_rounds,
        "seed": seed,
        "run_parameters": run_parameters,
        "commit": _git("rev-parse", "HEAD").strip(),
        "uncommitted": hashlib.sha256(_git("diff", "HEAD", "--", "fl_ids", "configs").encode()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]


class RowCheckpoint:
    """Append-only CSV of finished result rows, keyed by some of their columns.

    Args:
        path: Checkpoint CSV (created on the first `append`).
        fingerprint: `run_fingerprint` of the current run.
        key_columns: Columns that identify a row, e.g. ("run", "holdout_class").

    Raises:
        ValueError: If `path` holds rows from a run with a different fingerprint.
    """

    def __init__(self, path: str | Path, fingerprint: str, key_columns: tuple[str, ...]) -> None:
        self.path = Path(path)
        self.fingerprint = fingerprint
        self.key_columns = key_columns
        self._rows: dict[tuple, dict] = {}
        if self.path.exists():
            # round_trip: resumed floats must equal the ones written, bit for bit.
            existing = pd.read_csv(self.path, dtype={FINGERPRINT_COLUMN: str}, float_precision="round_trip")
            stale = existing[existing[FINGERPRINT_COLUMN] != fingerprint]
            if len(stale):
                raise ValueError(
                    f"Checkpoint {self.path} has {len(stale)} rows from a different run "
                    f"(code, config, rounds or seed changed). Delete it to start over."
                )
            for row in existing.drop(columns=FINGERPRINT_COLUMN).to_dict(orient="records"):
                self._rows[self._key(row)] = row
            logger.info("Resuming from %s: %d finished rows", self.path, len(self._rows))

    def _key(self, row: dict) -> tuple:
        return tuple(row[c] for c in self.key_columns)

    def get(self, *key) -> dict | None:
        """The finished row with this key, or None."""
        return self._rows.get(tuple(key))

    def append(self, row: dict) -> None:
        """Record a finished row, writing it to disk immediately."""
        new_file = not self.path.exists()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[*row, FINGERPRINT_COLUMN])
            if new_file:
                writer.writeheader()
            writer.writerow({**row, FINGERPRINT_COLUMN: self.fingerprint})
        self._rows[self._key(row)] = row

    def remove(self) -> None:
        """Delete the checkpoint once the run's final outputs are written."""
        self.path.unlink(missing_ok=True)
