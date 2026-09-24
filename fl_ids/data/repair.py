"""Targeted repair of column-shifted rows in the dataset (component 1).

The authors' per-capture CSVs for a couple of captures have values under
the wrong column names. Checked row by row against our own tshark
extraction of the same `.pcap` (which reproduces the authors' rows at
97-100% for the unaffected captures): in MITM's CSV, `tcp.seq` holds what
is really `udp.time_delta`, `udp.time_delta` holds `dns.qry.type`, and so
on, and 98.8% of those rows are in `DNN-EdgeIIoT-dataset.csv`. The model
would learn an "ARP spoofing" attack with no ARP fields, and real MITM
traffic replayed in Phase B wouldn't look like its training rows.

This re-extracts an affected capture from the authors' own `.pcap` with
the same field list, writes the rows in `DNN-EdgeIIoT-dataset.csv`'s own
format, and `fl_ids.data.pipeline.load_and_encode` swaps them in for the
corrupted rows (`data.capture_repairs`). Labeling follows the authors'
convention for attack captures: every packet in the capture gets the
capture's attack type. It's a data repair, not new feature engineering:
same fields, same encoding, same labels.

    python -m fl_ids.data.repair --pcap "data/raw/pcaps/MITM (ARP spoofing + DNS) Attack.pcap" \\
        --attack-type MITM --output data/processed/MITM_reextracted.csv
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from fl_ids.data.pipeline import BENIGN_LABEL, CANONICAL_PLACEHOLDER, CATEGORICAL_COLUMNS, TARGET_COLUMN
from fl_ids.sdn.feature_extraction import FEATURE_FIELDS, _authors_hex_spelling, _to_numeric, extract_raw_fields

logger = logging.getLogger(__name__)

LABEL_COLUMN = "Attack_label"


def reextract_capture_rows(pcap_path: str | Path, attack_type: str, dataset_columns: list[str]) -> pd.DataFrame:
    """Extract a capture's packets as rows in `DNN-EdgeIIoT-dataset.csv`'s format.

    Numeric fields become the decimal numbers the dataset uses (hex and
    tshark 4.x booleans converted, empty -> 0), categorical fields keep
    their strings with an empty field written as the canonical placeholder,
    and the columns component 1 drops anyway are filled with the placeholder.

    Args:
        pcap_path: The authors' `.pcap` for the capture.
        attack_type: Label for every row (the capture's attack type).
        dataset_columns: `DNN-EdgeIIoT-dataset.csv`'s columns, in order.

    Returns:
        One row per packet, columns exactly `dataset_columns`.
    """
    raw = extract_raw_fields(pcap_path)
    rows = pd.DataFrame(index=raw.index)
    for column in dataset_columns:
        if column == TARGET_COLUMN:
            rows[column] = attack_type
        elif column == LABEL_COLUMN:
            rows[column] = 0 if attack_type == BENIGN_LABEL else 1
        elif column in CATEGORICAL_COLUMNS:
            values = raw[column].fillna("").astype(str).map(_authors_hex_spelling)
            rows[column] = values.where(values != "", CANONICAL_PLACEHOLDER)
        elif column in FEATURE_FIELDS:
            rows[column] = _to_numeric(raw[column])
        else:  # dropped by component 1 before any model sees it
            rows[column] = CANONICAL_PLACEHOLDER
    logger.info("Re-extracted %d %s rows from %s", len(rows), attack_type, pcap_path)
    return rows[dataset_columns]


def apply_capture_repairs(df: pd.DataFrame, capture_repairs: dict[str, str]) -> pd.DataFrame:
    """Replace every row of each repaired attack type with its re-extracted rows.

    Args:
        df: The loaded dataset (after `load_raw_csv`, before cleaning).
        capture_repairs: `{attack_type: path to re-extracted CSV}` (`data.capture_repairs`).

    Returns:
        The dataset with those attack types' rows swapped.

    Raises:
        FileNotFoundError: If a configured repair file hasn't been generated.
    """
    from fl_ids.data.pipeline import load_raw_csv

    for attack_type, path in capture_repairs.items():
        if not Path(path).exists():
            raise FileNotFoundError(
                f"data.capture_repairs lists {attack_type} -> {path}, which doesn't exist yet. Generate it with "
                f"`python -m fl_ids.data.repair --pcap <the {attack_type} .pcap> --attack-type {attack_type} "
                f"--output {path}`"
            )
        replacement = load_raw_csv(path)
        removed = int((df[TARGET_COLUMN] == attack_type).sum())
        df = pd.concat([df[df[TARGET_COLUMN] != attack_type], replacement[df.columns]], ignore_index=True)
        logger.info("Repaired %s: replaced %d dataset rows with %d re-extracted rows", attack_type, removed, len(replacement))
    return df


if __name__ == "__main__":
    import argparse

    from fl_ids.utils.logging_setup import setup_logging

    parser = argparse.ArgumentParser(description="Re-extract a column-shifted capture in the dataset's format")
    parser.add_argument("--pcap", required=True)
    parser.add_argument("--attack-type", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-csv", default="data/raw/DNN-EdgeIIoT-dataset.csv", help="For the column order")
    args = parser.parse_args()

    setup_logging()
    columns = pd.read_csv(args.dataset_csv, nrows=0).columns.tolist()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reextract_capture_rows(args.pcap, args.attack_type, columns).to_csv(output, index=False)
    logger.info("Wrote %s", output)
