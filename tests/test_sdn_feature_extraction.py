"""Tests for live traffic feature extraction (Phase 7/9 support).

Fast tests exercise the cleaning/encoding/alignment logic directly on
synthetic DataFrames (no tshark needed). The real end-to-end extraction
(`extract_raw_fields` against an actual `.pcap`) is tested separately,
skipped if tshark or a real capture isn't present.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fl_ids.sdn.feature_extraction import (
    align_to_feature_schema,
    clean_and_encode_live,
    extract_raw_fields,
)

REAL_PCAP_PATH = Path("data/raw/pcaps/OS Fingerprinting attack.pcap")
TSHARK_AVAILABLE = shutil.which("tshark") is not None


def test_clean_and_encode_live_converts_hex_and_numeric_fields():
    raw_df = pd.DataFrame(
        {
            "icmp.checksum": ["0x2ea2", "100", ""],
            "tcp.len": ["50", "", "20"],
            "http.request.method": ["GET", "", "POST"],
        }
    )
    encoded = clean_and_encode_live(raw_df)

    assert encoded["icmp.checksum"].tolist() == [float(0x2ea2), 100.0, 0.0]
    assert encoded["tcp.len"].tolist() == [50.0, 0.0, 20.0]
    # http.request.method got one-hot encoded away.
    assert "http.request.method" not in encoded.columns
    assert "http.request.method_GET" in encoded.columns
    assert "http.request.method_POST" in encoded.columns


def test_clean_and_encode_live_marks_missing_categoricals_with_sentinel():
    raw_df = pd.DataFrame({"mqtt.protoname": ["MQTT", None, ""]})
    encoded = clean_and_encode_live(raw_df)

    # Two of three rows are placeholder (missing/empty) -> sentinel column.
    sentinel_cols = [c for c in encoded.columns if "PLACEHOLDER" in c]
    assert len(sentinel_cols) == 1
    assert encoded[sentinel_cols[0]].sum() == 2
    assert encoded["mqtt.protoname_MQTT"].sum() == 1


def test_align_to_feature_schema_resolves_placeholder_to_both_known_variants():
    raw_df = pd.DataFrame({"mqtt.protoname": [None, None]})
    encoded = clean_and_encode_live(raw_df)

    # Trained schema has both "_0" and "_0.0" variants for this field.
    feature_names = ["mqtt.protoname_0", "mqtt.protoname_0.0", "mqtt.protoname_MQTT"]
    aligned = align_to_feature_schema(encoded, feature_names)

    assert aligned.shape == (2, 3)
    assert np.array_equal(aligned[:, 0], [1.0, 1.0])  # _0 activated
    assert np.array_equal(aligned[:, 1], [1.0, 1.0])  # _0.0 also activated
    assert np.array_equal(aligned[:, 2], [0.0, 0.0])  # MQTT not activated


def test_align_to_feature_schema_drops_unseen_columns_and_fills_missing():
    df = pd.DataFrame({"icmp.checksum": [1.0, 2.0], "some_unseen_category_col": [1, 0]})
    feature_names = ["icmp.checksum", "tcp.len"]  # tcp.len absent from df; unseen col not in schema

    aligned = align_to_feature_schema(df, feature_names)

    assert aligned.shape == (2, 2)
    assert np.array_equal(aligned[:, 0], [1.0, 2.0])
    assert np.array_equal(aligned[:, 1], [0.0, 0.0])  # filled with 0, not observed


@pytest.mark.skipif(not TSHARK_AVAILABLE, reason="tshark not installed")
@pytest.mark.skipif(not REAL_PCAP_PATH.exists(), reason="real pcap not present")
def test_extract_raw_fields_against_real_pcap():
    df = extract_raw_fields(REAL_PCAP_PATH)
    assert len(df) > 0
    assert "icmp.checksum" in df.columns
    # Real ICMP echo request/reply traffic in this capture should show
    # non-null checksums for at least some packets.
    assert df["icmp.checksum"].notna().any()
