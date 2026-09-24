"""Tests for the targeted repair of column-shifted dataset rows (fl_ids.data.repair)."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fl_ids.data.pipeline import TARGET_COLUMN, load_and_encode, load_raw_csv
from fl_ids.data.repair import apply_capture_repairs, reextract_capture_rows
from tests.test_data_pipeline import _make_synthetic_dataframe

MITM_PCAP = Path("data/raw/pcaps/MITM (ARP spoofing + DNS) Attack.pcap")
DATASET = Path("data/raw/DNN-EdgeIIoT-dataset.csv")


def _loaded(tmp_path, n_per_class: int, seed: int) -> pd.DataFrame:
    """A synthetic dataset as the pipeline hands it to the repair: after load_raw_csv."""
    path = tmp_path / f"dataset_{seed}.csv"
    _make_synthetic_dataframe(n_per_class=n_per_class, seed=seed).to_csv(path, index=False)
    return load_raw_csv(path)


def test_repair_replaces_only_the_repaired_attack_type(tmp_path):
    df = _loaded(tmp_path, 50, 0)
    replacement = df[df[TARGET_COLUMN] == "XSS"].head(7).copy()
    replacement["feature_a"] = 999.0
    path = tmp_path / "xss.csv"
    replacement.to_csv(path, index=False)

    repaired = apply_capture_repairs(df, {"XSS": str(path)})

    assert (repaired[TARGET_COLUMN] == "XSS").sum() == 7
    assert (repaired.loc[repaired[TARGET_COLUMN] == "XSS", "feature_a"] == 999.0).all()
    for cls in set(df[TARGET_COLUMN]) - {"XSS"}:
        assert (repaired[TARGET_COLUMN] == cls).sum() == (df[TARGET_COLUMN] == cls).sum()


def test_missing_repair_file_fails_with_the_command_to_generate_it(tmp_path):
    df = _loaded(tmp_path, 5, 0)
    with pytest.raises(FileNotFoundError, match="python -m fl_ids.data.repair"):
        apply_capture_repairs(df, {"XSS": str(tmp_path / "missing.csv")})


def test_load_and_encode_applies_repairs(tmp_path):
    csv_path = tmp_path / "dataset.csv"
    df = _make_synthetic_dataframe(n_per_class=40, seed=1)
    df.to_csv(csv_path, index=False)
    replacement = df[df[TARGET_COLUMN] == "Password"].head(3).copy()
    replacement["feature_b"] = [101.0, 102.0, 103.0]
    replacement.to_csv(tmp_path / "password.csv", index=False)

    X, y, encoder, feature_names, _ = load_and_encode(csv_path, {"Password": str(tmp_path / "password.csv")})

    password = list(encoder.classes_).index("Password")
    assert (y == password).sum() == 3
    assert sorted(X[y == password, feature_names.index("feature_b")]) == [101.0, 102.0, 103.0]


@pytest.mark.skipif(
    not (MITM_PCAP.exists() and DATASET.exists() and shutil.which("tshark")), reason="real MITM capture not present"
)
def test_reextracted_mitm_rows_are_in_the_dataset_format_and_have_arp_fields():
    columns = pd.read_csv(DATASET, nrows=0).columns.tolist()
    rows = reextract_capture_rows(MITM_PCAP, "MITM", columns)

    assert rows.columns.tolist() == columns and len(rows) == 1229
    assert (rows[TARGET_COLUMN] == "MITM").all() and (rows["Attack_label"] == 1).all()
    # The authors' CSV has no ARP fields at all for this ARP-spoofing capture
    # (its values are shifted into other columns); the real packets do.
    assert (rows["arp.opcode"] != 0).mean() > 0.2
    numeric = rows.drop(columns=[TARGET_COLUMN]).select_dtypes(include="number")
    assert np.isfinite(numeric.to_numpy()).all()
