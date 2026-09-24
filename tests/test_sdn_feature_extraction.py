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


def test_tshark_4_boolean_fields_become_ones_and_zeros(caplog):
    # tshark 4.x prints booleans as True/False where the dataset authors'
    # version wrote 1/0; a silent fallback once zeroed tcp.flags.ack on
    # every live packet.
    raw_df = pd.DataFrame({"tcp.flags.ack": ["True", "False", "", "1"], "tcp.len": ["10", "x", "0x10", "3"]})
    with caplog.at_level("WARNING"):
        encoded = clean_and_encode_live(raw_df)

    assert encoded["tcp.flags.ack"].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert encoded["tcp.len"].tolist() == [10.0, 0.0, 16.0, 3.0]
    # The one genuinely unparseable value is reported, not silently zeroed.
    assert "tcp.len: 1 values couldn't be parsed" in caplog.text


def test_hex_categoricals_use_the_dataset_authors_spelling():
    # tshark 4.x: "0x00"; the authors' tshark (and so the training schema): "0x00000000".
    encoded = clean_and_encode_live(pd.DataFrame({"mqtt.conack.flags": ["0x00", "0x00000000", ""]}))
    assert encoded["mqtt.conack.flags_0x00000000"].tolist() == [1, 1, 0]
    assert "mqtt.conack.flags_0x00" not in encoded.columns


def test_clean_and_encode_live_maps_missing_categoricals_to_canonical_placeholder():
    # tshark reports a field that doesn't apply as empty; literal placeholder
    # spellings must land on the same canonical column training uses.
    raw_df = pd.DataFrame({"mqtt.protoname": ["MQTT", None, "", "0.0", "0"]})
    encoded = clean_and_encode_live(raw_df)

    assert encoded["mqtt.protoname_0"].sum() == 4
    assert encoded["mqtt.protoname_MQTT"].sum() == 1
    assert "mqtt.protoname_0.0" not in encoded.columns


def test_align_to_feature_schema_uses_the_single_trained_placeholder_column():
    raw_df = pd.DataFrame({"mqtt.protoname": [None, "MQTT"]})
    encoded = clean_and_encode_live(raw_df)

    # The trained schema has one placeholder column per field (see
    # fl_ids.data.pipeline.canonicalize_placeholders).
    feature_names = ["mqtt.protoname_0", "mqtt.protoname_MQTT"]
    aligned = align_to_feature_schema(encoded, feature_names)

    assert np.array_equal(aligned, [[1.0, 0.0], [0.0, 1.0]])


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


BACKDOOR_PCAP_PATH = Path("data/raw/pcaps/Backdoor_attack.pcap")
BACKDOOR_CSV_PATH = Path("data/raw/capture_csvs/Backdoor_attack.csv")


@pytest.mark.skipif(not TSHARK_AVAILABLE, reason="tshark not installed")
@pytest.mark.skipif(
    not (BACKDOOR_PCAP_PATH.exists() and BACKDOOR_CSV_PATH.exists()), reason="real capture + authors' CSV not present"
)
def test_live_extraction_reproduces_the_dataset_authors_rows():
    """Our tshark extraction of Backdoor_attack.pcap must match the authors' own CSV of that capture.

    Both go through the same live encoder, so this isolates extraction.
    Before the tshark 4.x boolean fix, only 6.6% of rows matched.
    """
    from fl_ids.sdn.feature_extraction import FEATURE_FIELDS

    authors_raw = pd.read_csv(BACKDOOR_CSV_PATH, dtype=str, low_memory=False)[FEATURE_FIELDS]
    authors = clean_and_encode_live(authors_raw)
    ours = clean_and_encode_live(extract_raw_fields(BACKDOOR_PCAP_PATH))
    columns = sorted(set(authors.columns) | set(ours.columns))
    authors_rows = {row.tobytes() for row in np.round(align_to_feature_schema(authors, columns), 3)}
    ours_aligned = np.round(align_to_feature_schema(ours, columns), 3)

    matched = np.mean([row.tobytes() in authors_rows for row in ours_aligned])
    assert matched > 0.99, f"only {matched:.3f} of live-extracted rows match the authors' rows"
