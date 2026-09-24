"""Rendering tests for the Streamlit dashboard (component 13).

`fl_ids.dashboard.data` has unit tests for each table/series; these run
the app itself (Streamlit's AppTest, in-process) against the state the
pipeline actually produces -- the strategy's round-history file for the
training view, the bridge's REST responses for the live view -- and check
that the panels show those values, not placeholders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "fl_ids" / "dashboard" / "app.py")

ROUND_HISTORY = [
    {
        "round": 1,
        "trust_scores": {"0": 0.93, "1": 0.9, "7": 0.62},
        "survivors": [0, 1],
        "exclusion_reasons": {"7": ["mad_outlier", "opposes_consensus"]},
        "filtered_fractions": {"0": 0.6, "1": 0.7, "7": 0.5},
        "mean_filtered_fraction": 0.6,
        "mean_reconstruction_error": 0.2448,
        "per_client_reconstruction_error": {"0": 0.2, "1": 0.3, "7": 0.25},
        "boosting_model_version": 1,
        "boosting_metrics": {"accuracy": 0.95, "weighted_f1": 0.94, "macro_f1": 0.8, "false_positive_rate": 0.001},
    },
    {
        "round": 2,
        "trust_scores": {"0": 0.9, "1": 0.88, "7": 0.41},
        "survivors": [0, 1],
        "exclusion_reasons": {"7": ["opposes_consensus", "low_trust"]},
        "filtered_fractions": {"0": 0.62, "1": 0.71, "7": 0.52},
        "mean_filtered_fraction": 0.62,
        "mean_reconstruction_error": 0.0614,
        "per_client_reconstruction_error": {"0": 0.05, "1": 0.07, "7": 0.06},
        "boosting_model_version": 2,
        "boosting_metrics": {"accuracy": 0.963, "weighted_f1": 0.96, "macro_f1": 0.88, "false_positive_rate": 0.0002},
    },
]


def _app(**session_state) -> AppTest:
    app = AppTest.from_file(APP_PATH, default_timeout=30)
    for key, value in session_state.items():
        app.session_state[key] = value
    return app


def _metrics(app: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in app.metric}


def test_training_view_shows_the_latest_round_from_the_live_state_file(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(ROUND_HISTORY))

    app = _app(training_state_path=str(state_path), training_auto_refresh=False).run()

    assert not app.exception
    metrics = _metrics(app)
    assert metrics["Round"] == "2"
    assert metrics["Clients surviving trust filter"] == "2/3"
    assert metrics["Mean benign reconstruction error"] == "0.0614"
    assert metrics["Model version"] == "2"
    assert metrics["Accuracy"] == "0.963"

    survivors = next(df.value for df in app.dataframe if "excluded" in df.value.columns)
    assert survivors["excluded"].tolist() == ["7 (mad_outlier+opposes_consensus)", "7 (opposes_consensus+low_trust)"]


def test_training_view_updates_when_a_new_round_is_written(tmp_path):
    state_path = tmp_path / "state.json"
    state_path.write_text(json.dumps(ROUND_HISTORY[:1]))
    app = _app(training_state_path=str(state_path), training_auto_refresh=False).run()
    assert _metrics(app)["Round"] == "1"

    state_path.write_text(json.dumps(ROUND_HISTORY))
    app.run()
    assert _metrics(app)["Round"] == "2"


def test_training_view_says_so_when_there_is_no_state_file(tmp_path):
    app = _app(training_state_path=str(tmp_path / "missing.json"), training_auto_refresh=False).run()
    assert not app.exception
    assert any("No live state file" in w.value for w in app.warning)
    assert not app.metric


BRIDGE = {
    "/health": {"status": "ok", "connected_datapaths": [1]},
    "/devices": {
        "h1": {"datapath_id": 1, "ip_address": "10.0.0.1", "port_name": "s1-eth1"},
        "h2": {"datapath_id": 1, "ip_address": "10.0.0.2", "port_name": "s1-eth2"},
    },
    "/mitigation_history": [
        {"device_id": "h1", "classification": "Backdoor", "confidence": 0.97, "stage": "boosting",
         "action": "block", "timestamp": 1_700_000_000.0},
    ],
    "/switch_state": {
        "1": {
            "flows": [
                {"priority": 100, "match": {"eth_type": 2048, "ipv4_src": "10.0.0.1"}, "actions": "drop",
                 "packet_count": 23788, "byte_count": 1_900_000, "duration_sec": 12},
                {"priority": 0, "match": {}, "actions": "NORMAL", "packet_count": 40, "byte_count": 4000, "duration_sec": 30},
            ],
            "meters": [],
            "ports": [
                {"port_no": 1, "name": "s1-eth1", "link_up": True, "rx_packets": 23800, "tx_packets": 10},
                {"port_no": 2, "name": "s1-eth2", "link_up": True, "rx_packets": 5, "tx_packets": 20},
            ],
        }
    },
}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_live_view_shows_the_switchs_mitigation_and_its_cause(monkeypatch):
    # The app runs in-process, so patching requests.get stands in for the bridge.
    monkeypatch.setattr(
        requests, "get", lambda url, timeout=None: _FakeResponse(BRIDGE["/" + url.rsplit("/", 1)[-1]])
    )

    app = _app(view="Live Simulation (Phase B)", bridge_url="http://bridge", simulation_auto_refresh=False).run()

    assert not app.exception
    metrics = _metrics(app)
    assert metrics["Connected switches"] == "1"
    assert metrics["Classifications received"] == "1"
    assert metrics["Devices under mitigation"] == "1"

    active = next(df.value for df in app.dataframe if "enforced" in df.value.columns)
    assert active.to_dict(orient="records")[0] == {
        "device": "h1", "ip_address": "10.0.0.1", "enforced": "block", "packets_matched": 23788,
        "classification": "Backdoor", "stage": "boosting", "confidence": 0.97,
    }


def test_live_view_reports_an_unreachable_bridge(monkeypatch):
    def refuse(url, timeout=None):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "get", refuse)
    app = _app(view="Live Simulation (Phase B)", bridge_url="http://bridge", simulation_auto_refresh=False).run()

    assert not app.exception
    assert any("Could not reach the SDN bridge" in e.value for e in app.error)
