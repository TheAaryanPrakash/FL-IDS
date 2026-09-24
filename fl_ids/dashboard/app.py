"""Live monitoring dashboard (component 13).

Two views, matching the two phases:
- **Training (Phase A):** reads `TrustFilteredStrategy`'s live per-round
  state file (`fl_ids.fl.strategy`'s `live_state_path`) — per-round
  reconstruction error (global and per-client), the boosting filter rate,
  the broadcast boosting model's metrics, per-client trust score
  trajectory, and which clients survived each round's trust filter.
- **Live simulation (Phase B):** reads the running SDN mitigation
  bridge's REST API directly — topology and link status, live
  classifications (with the cascade stage that made each call), the
  switch's actual flow table and active mitigations, and port traffic
  counters, all queried from the switch over OpenFlow on every refresh.

Reads real, live state from wherever the running pipeline already
persists it (component 7's strategy, component 9's bridge, and the switch
itself via component 10's controller) — this file holds no state of its
own, per CLAUDE.md's explicit instruction not to build a second, parallel
state-tracking system. (The one exception is the previous port-counter
sample kept in the viewer's session, purely to turn two counter readings
into a packets/s rate.)

Run with: `streamlit run fl_ids/dashboard/app.py`
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
import streamlit as st

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

st.set_page_config(page_title="FL-IDS Dashboard", layout="wide")
st.title("FL-IDS Live Monitoring Dashboard")

view = st.sidebar.radio("View", ["Training (Phase A)", "Live Simulation (Phase B)"], key="view")


def _render_boosting_section(round_history: list[dict]) -> None:
    st.subheader("Boosting Classifier (first-pass stage)")
    metrics_df = boosting_metrics_df(round_history)
    latest = round_history[-1]
    if metrics_df[["accuracy", "weighted_f1"]].isna().all().all():
        st.info(
            "No boosting metrics in the live state. Start the Flower server with "
            "`--boosting-metrics-path` to record the broadcast model's held-out metrics."
        )
        return

    version = latest.get("boosting_model_version")
    updated_rounds = metrics_df.index[metrics_df["model_version"].diff().fillna(0) != 0].tolist()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model version", version)
    col2.metric("Accuracy", f"{metrics_df['accuracy'].iloc[-1]:.3f}")
    col3.metric("Weighted F1", f"{metrics_df['weighted_f1'].iloc[-1]:.3f}")
    col4.metric("False-positive rate", f"{metrics_df['false_positive_rate'].iloc[-1]:.3f}")
    st.caption(
        f"Model updated in rounds {updated_rounds}." if updated_rounds
        else "Same boosting model broadcast every round so far (no update yet this run)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.line_chart(metrics_df[["accuracy", "weighted_f1", "macro_f1", "false_positive_rate"]])
    with col2:
        st.dataframe(boosting_per_class_table(latest.get("boosting_metrics")), width="stretch", height=300)


def _render_training_view() -> None:
    st.header("Phase A: Federated Training")
    state_path_str = st.sidebar.text_input("Live state file", value="/tmp/fl_ids_training_state.json", key="training_state_path")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (5s)", value=True, key="training_auto_refresh")

    path = Path(state_path_str)
    if not path.exists():
        st.warning(
            f"No live state file at {state_path_str} yet. Start a Flower server with "
            f"`--strategy custom_trust_filtered --live-state-path {state_path_str}` to see live data here."
        )
        round_history = []
    else:
        try:
            round_history = json.loads(path.read_text())
        except json.JSONDecodeError:
            st.info("State file unreadable, retrying...")
            round_history = []

    if path.exists() and not round_history:
        st.info("Training started but no rounds have completed yet...")

    if round_history:
        latest = round_history[-1]
        col1, col2, col3 = st.columns(3)
        col1.metric("Round", latest["round"])
        col2.metric("Clients surviving trust filter", f"{len(latest.get('survivors', []))}/{len(latest.get('trust_scores', {}))}")
        mean_error = latest.get("mean_reconstruction_error")
        col3.metric("Mean benign reconstruction error", f"{mean_error:.4f}" if mean_error is not None else "pending")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Reconstruction Error (global, benign validation)")
            st.line_chart(reconstruction_error_df(round_history))
        with col2:
            st.subheader("Boosting Filter Rate (fraction passed to autoencoder)")
            st.line_chart(filter_rate_df(round_history))

        _render_boosting_section(round_history)

        st.subheader("Per-Client Trust Score Trajectory")
        st.line_chart(trust_score_df(round_history))

        st.subheader("Per-Client Reconstruction Error")
        st.line_chart(per_client_reconstruction_error_df(round_history))

        st.subheader("Which Clients Survived Each Round's Trust Filter")
        st.dataframe(survivors_table(round_history), width="stretch")

        with st.expander(f"Raw state for round {latest['round']}"):
            st.json(latest)

    if auto_refresh:
        time.sleep(5)
        st.rerun()


def _topology_dot(switch_states: dict, devices: dict, mitigations_by_ip: dict[str, str]) -> str:
    """Graphviz DOT for the live topology: switches, their ports' devices, link and mitigation state."""
    colors = {"block": "#d62728", "rate_limit": "#ff7f0e"}
    lines = ["graph topology {", "rankdir=LR;", 'node [fontname="Helvetica", fontsize=10];']
    port_to_device = {info.get("port_name"): (device_id, info) for device_id, info in devices.items()}
    for dp_id, state in switch_states.items():
        switch_node = f"s{dp_id}"
        lines.append(f'"{switch_node}" [label="switch dpid={dp_id}", shape=box, style=filled, fillcolor="#dbe9f6"];')
        for port in state.get("ports") or []:
            device_id, info = port_to_device.get(port["name"], (port["name"], {}))
            action = mitigations_by_ip.get(info.get("ip_address"))
            fill = colors.get(action, "#e8f5e9")
            label = f"{device_id}\\n{info.get('ip_address', '')}" + (f"\\n[{action}]" if action else "")
            lines.append(f'"{device_id}" [label="{label}", shape=ellipse, style=filled, fillcolor="{fill}"];')
            edge_style = "solid" if port["link_up"] else "dashed"
            edge_color = "#555555" if port["link_up"] else "#d62728"
            lines.append(
                f'"{switch_node}" -- "{device_id}" [label="{port["name"]}", style={edge_style}, color="{edge_color}", fontsize=8];'
            )
    lines.append("}")
    return "\n".join(lines)


def _render_live_simulation_view() -> None:
    st.header("Phase B: Live SDN Mitigation Simulation")
    bridge_url = st.sidebar.text_input("SDN bridge URL", value="http://127.0.0.1:8080", key="bridge_url")
    auto_refresh = st.sidebar.checkbox("Auto-refresh (3s)", value=True, key="simulation_auto_refresh")

    try:
        health = requests.get(f"{bridge_url}/health", timeout=3).json()
        devices = requests.get(f"{bridge_url}/devices", timeout=3).json()
        history = requests.get(f"{bridge_url}/mitigation_history", timeout=3).json()
        switch_states = requests.get(f"{bridge_url}/switch_state", timeout=10).json()
    except requests.RequestException as exc:
        st.error(f"Could not reach the SDN bridge at {bridge_url}: {exc}")
        switch_states = None

    if switch_states is not None:
        all_flows = [flow for state in switch_states.values() for flow in state.get("flows") or []]
        active_df = active_mitigations_table(all_flows, devices, history)
        mitigations_by_ip = dict(zip(active_df["ip_address"], active_df["enforced"])) if not active_df.empty else {}

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Bridge status", health["status"])
        col2.metric("Connected switches", len(health["connected_datapaths"]))
        col3.metric("Classifications received", len(history))
        col4.metric("Devices under mitigation", len(active_df))

        st.subheader("Topology & Link Status")
        if switch_states:
            st.graphviz_chart(_topology_dot(switch_states, devices, mitigations_by_ip))
        else:
            st.info("No switch connected to the controller yet.")

        st.subheader("Live Classifications (cascade output -> mitigation)")
        history_df = mitigation_history_table(history)
        if not history_df.empty:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.dataframe(history_df, width="stretch")
            with col2:
                st.caption("Which cascade stage made the call")
                st.bar_chart(stage_counts(history))
        else:
            st.info("No classifications yet.")

        st.subheader("Active Mitigations (from the live flow table)")
        if not active_df.empty:
            st.dataframe(active_df, width="stretch")
        else:
            st.info("No per-device block or rate-limit rules on any switch.")

        now = time.time()
        previous = st.session_state.get("previous_port_sample")
        for dp_id, state in switch_states.items():
            st.subheader(f"Switch dpid={dp_id}")
            flows_col, traffic_col = st.columns(2)
            with flows_col:
                st.caption("Flow table (queried over OpenFlow, as `ovs-ofctl dump-flows` shows)")
                if state.get("flows") is None:
                    st.warning("Switch did not answer the flow-stats request.")
                else:
                    st.dataframe(flow_table_df(state["flows"], devices), width="stretch")
                if state.get("meters"):
                    st.caption("Rate-limit meters")
                    st.dataframe(state["meters"], width="stretch")
            with traffic_col:
                st.caption("Ports: link status and traffic counters")
                ports_df = ports_table(state.get("ports"), devices)
                if previous and dp_id in previous["ports"]:
                    rates = port_packet_rates(previous["ports"][dp_id], state.get("ports"), now - previous["time"])
                    if not ports_df.empty:
                        ports_df["rx_pkt_per_s"] = ports_df["interface"].map(rates).round(1)
                st.dataframe(ports_df, width="stretch")
        st.session_state["previous_port_sample"] = {
            "time": now,
            "ports": {dp_id: state.get("ports") for dp_id, state in switch_states.items()},
        }

        with st.expander("Registered devices"):
            st.dataframe(devices_table(devices), width="stretch")

    if auto_refresh:
        time.sleep(3)
        st.rerun()


if view == "Training (Phase A)":
    _render_training_view()
else:
    _render_live_simulation_view()
