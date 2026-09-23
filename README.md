# FL-IDS: Federated Learning-Based Intrusion Detection System

A federated learning intrusion detection system for IoT/IIoT traffic. See
`CLAUDE.md` for the full project specification, architecture, literature
grounding, and phased build plan this repository follows.

## Architecture at a glance

A two-stage detection cascade — a LightGBM/XGBoost boosting classifier as a
first-pass supervised filter, falling through to a client-side unsupervised
autoencoder for traffic the boosting stage calls "normal" — trained via
federated learning (Flower) across non-IID clients, protected by
server-side cosine-similarity trust filtering against poisoned updates, with
an SDN layer (Mininet + OpenFlow 1.3) enforcing the resulting
classifications as live flow-table mitigations.

Built in two phases: **Phase A** (offline FL training, producing evaluated
models) and **Phase B** (online mitigation demo replaying real traffic
through the trained cascade and SDN controller).

## Repository layout

```
fl_ids/
  data/           Component 1 — data pipeline (load, partition, normalize)
  models/         Components 2, 3 — boosting classifier, autoencoder
  fl/             Components 4, 7, 8 — Flower client, custom Strategy, server
  robustness/     Components 5, 6 — cosine trust filter, trimmed-mean aggregation
  sdn/            Components 9, 10 — mitigation bridge, controller/topology
  orchestration/  Component 11 — Phase A / Phase B entrypoints
  eval/           Component 12 — evaluation harness
  dashboard/      Component 13 — live monitoring dashboard
  utils/          Config loading, logging setup
configs/          config.yaml — all tunable parameters, single source of truth
tests/            pytest unit + integration tests, mirrors fl_ids/ structure
scripts/          One-off utility scripts (dataset download, etc.)
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Dataset (not committed — see `.gitignore`): download Edge-IIoTset from
Kaggle (`mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`)
and place `DNN-EdgeIIoT-dataset.csv` under `data/raw/`, and per-device
`.pcap` captures under `data/raw/pcaps/`.

## Running tests

```bash
pytest
```

## Status

Following the phased build plan in `CLAUDE.md`, starting from Phase 0
(scaffolding). Progress is tracked via git history — see commit messages
for which milestone each commit satisfies.
