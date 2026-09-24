"""Pure data-shaping helpers for the dashboard (component 13).

Kept separate from `app.py`'s Streamlit rendering so the actual
data-transformation logic (turning `TrustFilteredStrategy`'s round-history
JSON or the SDN bridge's REST responses into plottable tables) is
testable without needing a Streamlit runtime.
"""

from __future__ import annotations

import pandas as pd


def reconstruction_error_df(round_history: list[dict]) -> pd.DataFrame:
    """Global mean reconstruction error per round, indexed by round."""
    rounds = [r["round"] for r in round_history]
    values = [r.get("mean_reconstruction_error") for r in round_history]
    return pd.DataFrame({"round": rounds, "mean_reconstruction_error": values}).set_index("round")


def filter_rate_df(round_history: list[dict]) -> pd.DataFrame:
    """Boosting filter rate (mean fraction of local traffic passed to the autoencoder) per round."""
    rounds = [r["round"] for r in round_history]
    values = [r.get("mean_filtered_fraction") for r in round_history]
    return pd.DataFrame({"round": rounds, "mean_filtered_fraction": values}).set_index("round")


def _all_client_ids(round_history: list[dict]) -> list[int]:
    ids: set[int] = set()
    for entry in round_history:
        ids.update(int(cid) for cid in entry.get("trust_scores", {}).keys())
    return sorted(ids)


def trust_score_df(round_history: list[dict]) -> pd.DataFrame:
    """Per-client trust score trajectory, one column per client, indexed by round."""
    rounds = [r["round"] for r in round_history]
    data: dict[str, list] = {"round": rounds}
    for cid in _all_client_ids(round_history):
        data[f"client_{cid}"] = [r["trust_scores"].get(str(cid)) for r in round_history]
    return pd.DataFrame(data).set_index("round")


def per_client_reconstruction_error_df(round_history: list[dict]) -> pd.DataFrame:
    """Per-client reconstruction error, one column per client, indexed by round."""
    rounds = [r["round"] for r in round_history]
    data: dict[str, list] = {"round": rounds}
    for cid in _all_client_ids(round_history):
        data[f"client_{cid}"] = [
            r.get("per_client_reconstruction_error", {}).get(str(cid)) for r in round_history
        ]
    return pd.DataFrame(data).set_index("round")


def survivors_table(round_history: list[dict]) -> pd.DataFrame:
    """One row per round: which clients survived the trust filter, which were excluded and why."""
    rows = []
    for entry in round_history:
        reasons = entry.get("exclusion_reasons", {})
        rows.append(
            {
                "round": entry["round"],
                "survivors": ", ".join(str(c) for c in entry.get("survivors", [])),
                "excluded": ", ".join(f"{cid} ({'+'.join(why)})" for cid, why in reasons.items()),
            }
        )
    return pd.DataFrame(rows)


def boosting_metrics_df(round_history: list[dict]) -> pd.DataFrame:
    """Held-out metrics of the boosting model broadcast each round, indexed by round.

    Rounds without recorded metrics (a server started without
    `--boosting-metrics-path`) come through as NaN rather than being dropped,
    so the round axis stays aligned with the other training charts.
    """
    rows = []
    for entry in round_history:
        metrics = entry.get("boosting_metrics") or {}
        rows.append(
            {
                "round": entry["round"],
                "model_version": entry.get("boosting_model_version"),
                "accuracy": metrics.get("accuracy"),
                "weighted_f1": metrics.get("weighted_f1"),
                "macro_f1": metrics.get("macro_f1"),
                "false_positive_rate": metrics.get("false_positive_rate"),
            }
        )
    return pd.DataFrame(rows).set_index("round").astype(float)


def boosting_per_class_table(boosting_metrics: dict | None) -> pd.DataFrame:
    """Per-attack-type F1/recall/support of one boosting model, worst F1 first."""
    per_class = (boosting_metrics or {}).get("per_class", {})
    rows = [{"class": name, **values} for name, values in per_class.items()]
    df = pd.DataFrame(rows, columns=["class", "f1", "recall", "support"])
    return df.sort_values("f1").reset_index(drop=True)


def devices_table(devices: dict[str, dict]) -> pd.DataFrame:
    """The SDN bridge's device registry as a flat table."""
    return pd.DataFrame([{"device_id": device_id, **info} for device_id, info in devices.items()])


MITIGATION_HISTORY_COLUMNS = [
    "time", "device_id", "ip_address", "classification", "stage", "confidence", "action", "applied",
]


def mitigation_history_table(history: list[dict]) -> pd.DataFrame:
    """The SDN bridge's mitigation history, most recent first.

    Each row is one live cascade classification: the device, the label,
    which cascade stage made the call (`boosting` or `autoencoder`), its
    confidence, and the mitigation action the bridge took.
    """
    df = pd.DataFrame(history)
    if df.empty:
        return df
    if "timestamp" in df.columns:
        df["time"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%H:%M:%S")
    columns = [c for c in MITIGATION_HISTORY_COLUMNS if c in df.columns]
    return df[columns].iloc[::-1].reset_index(drop=True)


def stage_counts(history: list[dict]) -> pd.DataFrame:
    """How many live classifications each cascade stage made, indexed by stage."""
    stages = [entry.get("stage") or "unknown" for entry in history]
    counts = pd.Series(stages, dtype=object).value_counts()
    return counts.rename_axis("stage").to_frame("classifications")


def _ip_to_device(devices: dict[str, dict]) -> dict[str, str]:
    return {info["ip_address"]: device_id for device_id, info in devices.items()}


def flow_table_df(flows: list[dict] | None, devices: dict[str, dict]) -> pd.DataFrame:
    """A switch's live flow table as a flat table, with each match's source IP resolved to its device."""
    ip_to_device = _ip_to_device(devices)
    rows = []
    for flow in flows or []:
        match = flow.get("match", {})
        rows.append(
            {
                "priority": flow["priority"],
                "match": ", ".join(f"{k}={v}" for k, v in match.items()) or "*",
                "device": ip_to_device.get(match.get("ipv4_src"), ""),
                "actions": flow["actions"],
                "packets": flow.get("packet_count"),
                "bytes": flow.get("byte_count"),
                "age_s": flow.get("duration_sec"),
            }
        )
    return pd.DataFrame(rows)


def _enforced_action(actions: str) -> str:
    """Name the mitigation a flow's actions implement: a drop is a block, a meter is a rate limit."""
    if actions == "drop":
        return "block"
    if "meter" in actions:
        return "rate_limit"
    return actions


def active_mitigations_table(
    flows: list[dict] | None,
    devices: dict[str, dict],
    history: list[dict],
) -> pd.DataFrame:
    """Mitigations actually in force on the switch, one row per mitigated device.

    Derived from the *live flow table* -- a device appears here only if
    the switch really holds a per-source-IP rule for it -- then annotated
    with the most recent classification that caused it. So a mitigation
    the bridge thinks it applied but the switch doesn't hold won't show.
    """
    ip_to_device = _ip_to_device(devices)
    latest_by_device: dict[str, dict] = {}
    for entry in history:
        latest_by_device[entry["device_id"]] = entry

    rows = []
    for flow in flows or []:
        ip = flow.get("match", {}).get("ipv4_src")
        if ip is None:
            continue
        device_id = ip_to_device.get(ip, "")
        cause = latest_by_device.get(device_id, {})
        rows.append(
            {
                "device": device_id,
                "ip_address": ip,
                "enforced": _enforced_action(flow["actions"]),
                "packets_matched": flow.get("packet_count"),
                "classification": cause.get("classification"),
                "stage": cause.get("stage"),
                "confidence": cause.get("confidence"),
            }
        )
    return pd.DataFrame(rows)


def ports_table(ports: list[dict] | None, devices: dict[str, dict]) -> pd.DataFrame:
    """Switch ports with link status and counters, each attached to the device registered on it."""
    port_to_device = {info.get("port_name"): device_id for device_id, info in devices.items() if info.get("port_name")}
    rows = []
    for port in ports or []:
        rows.append(
            {
                "port": port["port_no"],
                "interface": port["name"],
                "device": port_to_device.get(port["name"], ""),
                "link": "up" if port["link_up"] else "DOWN",
                "rx_packets": port.get("rx_packets"),
                "tx_packets": port.get("tx_packets"),
                "rx_bytes": port.get("rx_bytes"),
                "tx_bytes": port.get("tx_bytes"),
                "rx_dropped": port.get("rx_dropped"),
            }
        )
    return pd.DataFrame(rows)


def port_packet_rates(
    previous_ports: list[dict] | None,
    current_ports: list[dict] | None,
    elapsed_s: float,
) -> dict[str, float]:
    """Per-interface received-packet rate (packets/s) between two port-stats samples.

    Ports missing from the previous sample (or a counter that went
    backwards, e.g. a switch restart) are left out rather than reported
    as a bogus rate.
    """
    if not previous_ports or not current_ports or elapsed_s <= 0:
        return {}
    previous = {p["name"]: p.get("rx_packets", 0) for p in previous_ports}
    rates = {}
    for port in current_ports:
        before = previous.get(port["name"])
        now = port.get("rx_packets", 0)
        if before is not None and now >= before:
            rates[port["name"]] = (now - before) / elapsed_s
    return rates
