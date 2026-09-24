"""Live traffic feature extraction (Phase B support for components 9/11).

Extracts the same Wireshark fields the Edge-IIoTset DNN dataset's authors
used — see `fl_ids.data.pipeline`'s `DEFAULT_DROP_COLUMNS`/
`CATEGORICAL_COLUMNS`, taken from the dataset's own documented
preprocessing recipe — directly from a `.pcap` capture via `tshark`,
producing feature rows that can be aligned to a trained model's feature
schema for live cascade inference. This is what makes Phase 7/9's live
demo classify *real* replayed traffic rather than a synthetic or
manually-triggered stand-in.

**AppArmor note:** this system's `tshark` profile only permits offline
file reads (`-r`) from `user-tmp`-covered locations (e.g. `/tmp`), not
arbitrary project paths — the profile's `file r /**.pcap` rule lives only
in the separate `dumpcap` (live-capture) sub-profile, which doesn't apply
to `-r`. `extract_raw_fields` stages a copy into `/tmp` before invoking
tshark to work within that constraint.
"""

from __future__ import annotations

import io
import logging
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from fl_ids.data.pipeline import (
    CANONICAL_PLACEHOLDER,
    CATEGORICAL_COLUMNS,
    DEFAULT_DROP_COLUMNS,
    PLACEHOLDER_SPELLINGS,
)

logger = logging.getLogger(__name__)

# All 61 Wireshark fields in the DNN-EdgeIIoT-dataset.csv schema, minus
# the two post-hoc label columns (Attack_label/Attack_type) -- those
# aren't extractable packet fields, they're what we're predicting.
ALL_WIRESHARK_FIELDS: list[str] = [
    "frame.time", "ip.src_host", "ip.dst_host",
    "arp.dst.proto_ipv4", "arp.opcode", "arp.hw.size", "arp.src.proto_ipv4",
    "icmp.checksum", "icmp.seq_le", "icmp.transmit_timestamp", "icmp.unused",
    "http.file_data", "http.content_length", "http.request.uri.query",
    "http.request.method", "http.referer", "http.request.full_uri",
    "http.request.version", "http.response", "http.tls_port",
    "tcp.ack", "tcp.ack_raw", "tcp.checksum", "tcp.connection.fin",
    "tcp.connection.rst", "tcp.connection.syn", "tcp.connection.synack",
    "tcp.dstport", "tcp.flags", "tcp.flags.ack", "tcp.len", "tcp.options",
    "tcp.payload", "tcp.seq", "tcp.srcport",
    "udp.port", "udp.stream", "udp.time_delta",
    "dns.qry.name", "dns.qry.name.len", "dns.qry.qu", "dns.qry.type",
    "dns.retransmission", "dns.retransmit_request", "dns.retransmit_request_in",
    "mqtt.conack.flags", "mqtt.conflag.cleansess", "mqtt.conflags",
    "mqtt.hdrflags", "mqtt.len", "mqtt.msg_decoded_as", "mqtt.msg",
    "mqtt.msgtype", "mqtt.proto_len", "mqtt.protoname", "mqtt.topic",
    "mqtt.topic_len", "mqtt.ver",
    "mbtcp.len", "mbtcp.trans_id", "mbtcp.unit_id",
]

# Fields actually needed as model input: the raw field set minus the
# ID/timestamp/raw-payload columns component 1 also drops (identical
# recipe, applied at extraction time here rather than after the fact).
FEATURE_FIELDS: list[str] = [f for f in ALL_WIRESHARK_FIELDS if f not in DEFAULT_DROP_COLUMNS]

_HEX_LIKE_PREFIX = "0x"
_BOOLEAN_VALUES = {"True": 1.0, "False": 0.0}
_HEX_VALUE = re.compile(r"0x[0-9a-fA-F]+")
# Text fields the dataset stores as 0 (its authors dropped the query name
# text; the column survives in the schema as ~all zeros), so a name here is
# expected and not worth a warning.
_TEXT_FIELDS_STORED_AS_ZERO = {"dns.qry.name"}


def _stage_in_tmp(pcap_path: str | Path) -> Path:
    """Copy a pcap into /tmp so tshark's AppArmor profile allows reading it."""
    staged = Path(tempfile.gettempdir()) / f"fl_ids_extract_{Path(pcap_path).name}"
    shutil.copyfile(pcap_path, staged)
    return staged


def extract_raw_fields(pcap_path: str | Path) -> pd.DataFrame:
    """Run tshark to extract `FEATURE_FIELDS` from a `.pcap` capture.

    Args:
        pcap_path: Path to a `.pcap` file (e.g. one of Edge-IIoTset's raw
            per-device captures, or a live-captured replay).

    Returns:
        A DataFrame with one row per packet, columns = `FEATURE_FIELDS`
        (raw string/hex values, not yet cleaned or encoded).
    """
    staged_path = _stage_in_tmp(pcap_path)
    try:
        cmd = ["tshark", "-r", str(staged_path), "-T", "fields"]
        for field in FEATURE_FIELDS:
            cmd += ["-e", field]
        cmd += ["-E", "header=y", "-E", "separator=,", "-E", "quote=d", "-E", "occurrence=f"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    finally:
        staged_path.unlink(missing_ok=True)

    df = pd.read_csv(io.StringIO(result.stdout), dtype=str)
    logger.info("Extracted %d packets, %d fields from %s", len(df), len(df.columns), pcap_path)
    return df


def _to_numeric(series: pd.Series) -> pd.Series:
    """Convert a raw tshark field column to floats, handling hex and boolean renderings.

    Some fields (e.g. checksums) tshark renders as hex strings (`"0x2ea2"`)
    rather than the plain decimal the source CSV used, and newer tshark
    (4.x) renders boolean fields such as `tcp.flags.ack` as `True`/`False`
    where the dataset authors' version wrote `1`/`0` — both converted here
    so downstream numeric handling matches component 1's pipeline. Missing
    values become 0, matching the dataset's own convention for fields
    that don't apply to a given packet (e.g. `tcp.*` fields on a
    non-TCP packet).

    Anything else unparseable also becomes 0, but is logged: a silent
    fallback here once zeroed `tcp.flags.ack` on every packet (tshark 4.6's
    `True`), and live rows stopped matching the training rows.
    """
    unparseable: list[str] = []

    def convert(value):
        if pd.isna(value) or value == "":
            return 0.0
        text = str(value)
        if text in _BOOLEAN_VALUES:
            return _BOOLEAN_VALUES[text]
        if text.startswith(_HEX_LIKE_PREFIX):
            return float(int(text, 16))
        try:
            return float(text)
        except ValueError:
            unparseable.append(text)
            return 0.0

    converted = series.map(convert)
    if unparseable and series.name not in _TEXT_FIELDS_STORED_AS_ZERO:
        logger.warning(
            "%s: %d values couldn't be parsed as numbers and were set to 0 (e.g. %r)",
            series.name, len(unparseable), unparseable[0],
        )
    return converted


# A field that doesn't apply to a packet comes out of tshark empty. The
# training pipeline writes that case as `CANONICAL_PLACEHOLDER` for every
# row (fl_ids.data.pipeline.canonicalize_placeholders collapses the source
# CSVs' label-correlated "0"/"0.0" spellings), so live traffic maps empty
# fields, and any literal placeholder spelling, to that same value.


def _authors_hex_spelling(value: str) -> str:
    """Rewrite a hex categorical value in the dataset authors' 8-digit spelling.

    One-hot categories are matched by exact string, and tshark 4.x prints
    e.g. `mqtt.conack.flags` as `0x00` where the authors' version wrote
    `0x00000000` -- same value, but a category the trained model has never
    seen, so live MQTT CONNACK packets lost the feature entirely.
    """
    if _HEX_VALUE.fullmatch(value):
        return f"0x{int(value, 16):08x}"
    return value


def clean_and_encode_live(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Clean and one-hot encode freshly-extracted live fields.

    Args:
        raw_df: Output of `extract_raw_fields`.

    Returns:
        A DataFrame with numeric fields converted and the same nominal
        columns one-hot encoded as component 1's training pipeline
        (`fl_ids.data.pipeline.one_hot_encode_categoricals`), with empty
        or placeholder categorical values written as the training
        pipeline's `CANONICAL_PLACEHOLDER`.
    """
    df = raw_df.copy()
    for col in df.columns:
        if col in CATEGORICAL_COLUMNS:
            values = df[col].fillna("").astype(str).map(_authors_hex_spelling)
            is_placeholder = (values == "") | values.isin(PLACEHOLDER_SPELLINGS)
            df[col] = values.where(~is_placeholder, CANONICAL_PLACEHOLDER)
        else:
            df[col] = _to_numeric(df[col])

    present_categoricals = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    return pd.get_dummies(df, columns=present_categoricals)


def align_to_feature_schema(df: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Align a cleaned/encoded live batch to a trained model's exact feature columns.

    Live traffic in a short capture window won't necessarily exercise
    every one-hot category the training data did (e.g. no HTTP POST
    requests in this batch) — those columns are filled with 0, matching
    "this category wasn't observed." Any column produced live but absent
    from the trained schema (an unseen category value) is dropped, since
    the model has no corresponding weight for it.

    Args:
        df: Output of `clean_and_encode_live`.
        feature_names: The trained model's feature column names, in order
            (component 1's `build_federated_dataset` return value).

    Returns:
        A float32 array, shape (n_packets, len(feature_names)), column
        order matching `feature_names` exactly.
    """
    aligned = df.reindex(columns=feature_names, fill_value=0.0)
    return aligned.to_numpy(dtype=np.float32)


def extract_features_for_inference(pcap_path: str | Path, feature_names: list[str]) -> np.ndarray:
    """End-to-end: pcap -> raw fields -> cleaned/encoded -> aligned feature matrix.

    Args:
        pcap_path: Path to a `.pcap` capture.
        feature_names: The trained model's feature column names, in order.

    Returns:
        A float32 array ready for `BoostingClassifier`/`Autoencoder` input.
    """
    raw_df = extract_raw_fields(pcap_path)
    encoded_df = clean_and_encode_live(raw_df)
    return align_to_feature_schema(encoded_df, feature_names)
