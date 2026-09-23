"""Tests for the dashboard's data-shaping helpers (component 13)."""

from __future__ import annotations

import math

import pandas as pd

from fl_ids.dashboard.data import (
    active_mitigations_table,
    boosting_metrics_df,
    boosting_per_class_table,
    devices_table,
    filter_rate_df,
    flow_table_df,
    mitigation_history_table,
    per_client_reconstruction_error_df,
    port_packet_rates,
    ports_table,
    reconstruction_error_df,
    stage_counts,
    survivors_table,
    trust_score_df,
)

ROUND_HISTORY = [
    {
        "round": 1,
        "trust_scores": {"0": 0.9, "1": 0.85, "99": 0.6},
        "is_outlier": {"0": False, "1": False, "99": True},
        "survivors": [0, 1],
        "mean_filtered_fraction": 0.62,
        "mean_reconstruction_error": 0.5,
        "per_client_reconstruction_error": {"0": 0.4, "1": 0.45, "99": 1.2},
        "boosting_model_version": 1,
        "boosting_metrics": {
            "accuracy": 0.9, "weighted_f1": 0.88, "macro_f1": 0.7, "false_positive_rate": 0.02,
            "per_class": {
                "Normal": {"f1": 0.99, "recall": 0.99, "support": 1000},
                "Backdoor": {"f1": 0.4, "recall": 0.3, "support": 50},
            },
        },
    },
    {
        "round": 2,
        "trust_scores": {"0": 0.92, "1": 0.88, "99": 0.4},
        "is_outlier": {"0": False, "1": False, "99": True},
        "survivors": [0, 1],
        "mean_filtered_fraction": 0.65,
        "mean_reconstruction_error": 0.35,
        "per_client_reconstruction_error": {"0": 0.3, "1": 0.32, "99": 1.5},
        "boosting_model_version": 2,
        "boosting_metrics": {"accuracy": 0.93, "weighted_f1": 0.91, "macro_f1": 0.75, "false_positive_rate": 0.01},
    },
]


def test_reconstruction_error_df_shape_and_values():
    df = reconstruction_error_df(ROUND_HISTORY)
    assert list(df.index) == [1, 2]
    assert df["mean_reconstruction_error"].tolist() == [0.5, 0.35]


def test_filter_rate_df_shape_and_values():
    df = filter_rate_df(ROUND_HISTORY)
    assert df["mean_filtered_fraction"].tolist() == [0.62, 0.65]


def test_trust_score_df_has_one_column_per_client():
    df = trust_score_df(ROUND_HISTORY)
    assert set(df.columns) == {"client_0", "client_1", "client_99"}
    assert df["client_99"].tolist() == [0.6, 0.4]


def test_per_client_reconstruction_error_df_matches_source():
    df = per_client_reconstruction_error_df(ROUND_HISTORY)
    assert df["client_0"].tolist() == [0.4, 0.3]
    assert df["client_99"].tolist() == [1.2, 1.5]


def test_survivors_table_reports_survivors_and_excluded():
    table = survivors_table(ROUND_HISTORY)
    assert len(table) == 2
    assert table.iloc[0]["survivors"] == "0, 1"
    assert table.iloc[0]["excluded"] == "99"


def test_devices_table_flattens_registry():
    devices = {"h1": {"datapath_id": 1, "ip_address": "10.0.0.1"}}
    table = devices_table(devices)
    assert table.to_dict(orient="records") == [{"device_id": "h1", "datapath_id": 1, "ip_address": "10.0.0.1"}]


def test_devices_table_empty_registry():
    assert devices_table({}).empty


def test_mitigation_history_table_reverses_to_most_recent_first():
    history = [
        {"device_id": "h1", "action": "allow"},
        {"device_id": "h1", "action": "block"},
    ]
    table = mitigation_history_table(history)
    assert table.iloc[0]["action"] == "block"
    assert table.iloc[1]["action"] == "allow"


def test_mitigation_history_table_empty():
    assert mitigation_history_table([]).empty


DEVICES = {
    "h1": {"datapath_id": 1, "ip_address": "10.0.0.1", "port_name": "s1-eth1"},
    "h2": {"datapath_id": 1, "ip_address": "10.0.0.2", "port_name": "s1-eth2"},
}

FLOWS = [
    {"priority": 100, "match": {"eth_type": 2048, "ipv4_src": "10.0.0.1"}, "actions": "drop",
     "packet_count": 42, "byte_count": 4200, "duration_sec": 3},
    {"priority": 100, "match": {"eth_type": 2048, "ipv4_src": "10.0.0.2"}, "actions": "meter:1000,output:NORMAL",
     "packet_count": 7, "byte_count": 700, "duration_sec": 2},
    {"priority": 0, "match": {}, "actions": "output:NORMAL", "packet_count": 900, "byte_count": 90000, "duration_sec": 60},
]


def test_boosting_metrics_df_tracks_version_and_metrics_per_round():
    df = boosting_metrics_df(ROUND_HISTORY)
    assert df["model_version"].tolist() == [1, 2]
    assert df["accuracy"].tolist() == [0.9, 0.93]


def test_boosting_metrics_df_missing_metrics_become_nan_not_dropped():
    df = boosting_metrics_df([{"round": 1}, {"round": 2, "boosting_metrics": None}])
    assert list(df.index) == [1, 2]
    assert df["accuracy"].isna().all()


def test_boosting_per_class_table_sorts_worst_f1_first():
    table = boosting_per_class_table(ROUND_HISTORY[0]["boosting_metrics"])
    assert table["class"].tolist() == ["Backdoor", "Normal"]


def test_boosting_per_class_table_handles_missing_metrics():
    assert boosting_per_class_table(None).empty


def test_mitigation_history_table_orders_columns_and_formats_time():
    history = [
        {"timestamp": 0.0, "device_id": "h1", "ip_address": "10.0.0.1", "classification": "anomalous",
         "confidence": 0.6, "stage": "autoencoder", "action": "rate_limit", "applied": True},
    ]
    table = mitigation_history_table(history)
    assert list(table.columns) == [
        "time", "device_id", "ip_address", "classification", "stage", "confidence", "action", "applied",
    ]
    assert table.iloc[0]["time"] == "00:00:00"


def test_stage_counts_counts_each_cascade_stage():
    history = [{"stage": "boosting"}, {"stage": "boosting"}, {"stage": "autoencoder"}, {}]
    counts = stage_counts(history)["classifications"].to_dict()
    assert counts == {"boosting": 2, "autoencoder": 1, "unknown": 1}


def test_flow_table_df_resolves_source_ip_to_device():
    table = flow_table_df(FLOWS, DEVICES)
    assert table["device"].tolist() == ["h1", "h2", ""]
    assert table.iloc[2]["match"] == "*"
    assert table.iloc[0]["match"] == "eth_type=2048, ipv4_src=10.0.0.1"


def test_active_mitigations_come_from_flow_table_annotated_with_cause():
    history = [
        {"device_id": "h1", "classification": "benign", "stage": "autoencoder", "confidence": 0.9},
        {"device_id": "h1", "classification": "Backdoor", "stage": "boosting", "confidence": 0.97},
    ]
    table = active_mitigations_table(FLOWS, DEVICES, history)
    records = table.to_dict(orient="records")
    assert records[0]["device"] == "h1" and records[0]["enforced"] == "block"
    assert records[0]["classification"] == "Backdoor" and records[0]["stage"] == "boosting"
    assert records[1]["device"] == "h2" and records[1]["enforced"] == "rate_limit"
    assert pd.isna(records[1]["classification"])  # rule on the switch with no recorded cause still shows


def test_active_mitigations_ignore_history_without_a_flow_rule():
    history = [{"device_id": "h1", "classification": "Backdoor", "stage": "boosting", "confidence": 0.97}]
    default_only = [FLOWS[2]]
    assert active_mitigations_table(default_only, DEVICES, history).empty


def test_ports_table_attaches_devices_and_link_state():
    ports = [
        {"port_no": 1, "name": "s1-eth1", "link_up": True, "rx_packets": 10},
        {"port_no": 2, "name": "s1-eth2", "link_up": False, "rx_packets": 0},
    ]
    table = ports_table(ports, DEVICES)
    assert table["device"].tolist() == ["h1", "h2"]
    assert table["link"].tolist() == ["up", "DOWN"]


def test_port_packet_rates_between_samples():
    before = [{"name": "s1-eth1", "rx_packets": 100}, {"name": "s1-eth2", "rx_packets": 50}]
    after = [{"name": "s1-eth1", "rx_packets": 400}, {"name": "s1-eth2", "rx_packets": 10}, {"name": "s1-eth3", "rx_packets": 5}]
    rates = port_packet_rates(before, after, elapsed_s=3.0)
    assert math.isclose(rates["s1-eth1"], 100.0)
    assert "s1-eth2" not in rates  # counter went backwards (switch restart)
    assert "s1-eth3" not in rates  # no previous sample


def test_port_packet_rates_without_previous_sample():
    assert port_packet_rates(None, [{"name": "s1-eth1", "rx_packets": 1}], 3.0) == {}
