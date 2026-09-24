# PAPER_NOTES — raw material for the FL-IDS paper

> **Maintenance rule (applies every session):** this file is updated at the
> end of every phase, in the same commit as that phase's code, and pushed to
> `origin` at once so the remote never falls behind. Update it whenever a
> run produces numbers, a design changes, or a limitation turns up.
>
> **Number rule:** every number here was produced by an actual run and is
> tagged with its source (commit, script or log, seed, rounds). Never add
> estimated, rounded-for-effect or not-yet-produced numbers; write
> "pending" instead. Superseded numbers move to §9 ("Do not cite") with the
> reason, and are never silently deleted.
>
> This is notes, not prose. The paper gets written from this.

Last updated: 2026-09-25. Phase 6 regeneration at commit `7d127a0`: ablation
complete; poisoning sweep and standalone zero-day killed for low memory
(§8.3), to be rerun sequentially.

---

## 0. Status by phase (from git log, not assumed)

| Phase | Milestone commit | State |
|---|---|---|
| 0 Scaffolding | `3d0b5d9` | done |
| 1 Data pipeline | `494cb4d` | done; later fixes `c1cc656` (label leak), `6bea978` (MITM repair), `6bb8317` (z-score clip) |
| 2 Boosting bootstrap | `b097c83` | done; later `2009d90` (leaf regularization), `54d8d18` (incremental update) |
| 3 Core FL loop | `af49e8f` | done |
| 4 Robustness layer | `a47d93f` | done; exclusion rule changed in `7d127a0` (§4.6) |
| 5 Real data | `680e1a7` | done; its numbers predate the leak fix (§9) |
| 6 Evaluation | `854cd3b` harness; `0224a6d`, `2341ed8`, `11103fe` | harness done; at `7d127a0` (20 rounds, seed 42): **ablation complete**; poisoning sweep and standalone zero-day killed for low memory, need a sequential rerun |
| 7 SDN | `5cbb35c` | milestone demo done pre-leak-fix; **must re-run with the Phase A bundle** |
| 8 Dashboard | `0fa7c81`; app tests `f48f5fe` | done |
| 11 Orchestration | `0a0726f` | code done; **real-data Phase A never completed** (last attempt interrupted at round 10/20) |
| 9 End-to-end run | — | not started |
| 10 Polish | README `2dcad51` | in progress |

---

## 1. Problem, motivation, framing (abstract/intro material)

- Problem: intrusion detection for IoT/IIoT traffic. Constraints: raw
  traffic shouldn't leave devices/sites (privacy, bandwidth); new attack
  types appear that a supervised classifier bounded by its label taxonomy
  can't name; federated training is exposed to poisoned client updates;
  detection is only useful if it drives enforcement.
- Five techniques, each covering a gap the others leave:
  1. **Boosting (LightGBM), first pass, supervised**: names known attack
     types cheaply and decisively.
  2. **Client-side autoencoder, second pass, unsupervised**: runs only on
     traffic boosting didn't confidently call an attack; flags
     reconstruction error above a per-client benign threshold as
     `anomalous` (zero-day signal). Trained only on boosting-filtered
     "normal" traffic.
  3. **Cosine-similarity trust filter**: server-side screening of client
     autoencoder weight deltas against poisoning.
  4. **Federated learning (Flower)**: trains the autoencoder across
     non-IID clients without moving raw traffic.
  5. **SDN (Mininet + Open vSwitch, OpenFlow 1.3)**: turns cascade
     classifications into flow-table actions (drop / meter rate-limit).
- Two-phase architecture (deliberate): Phase A = offline FL training
  producing evaluated models; Phase B = online mitigation demo loading
  them. This keeps ML correctness separate from distributed-networking
  problems, and the evaluation numbers come from Phase A.
- Contribution framing: an integration of all five, with measured
  per-stage and per-defense contributions (ablation), poisoning-fraction
  breakdown, zero-day (leave-one-class-out) detection, and live enforcement.
  Also a secondary contribution: data-quality findings on Edge-IIoTset
  (§5.3) that inflate results for any model trained on the authors'
  preprocessed CSV.

### Literature grounding (verbatim from CLAUDE.md; do not restate)

This isn't an untested architecture — each piece has real precedent in
peer-reviewed work from 2020 onward, and it's worth knowing which
papers back which decision, both for confidence going in and for the
related-work section of the writeup.

- **Federated autoencoder-based anomaly detection**: well-established.
  Mothukuri et al., "Federated-learning-based anomaly detection for
  IoT security attacks," *IEEE Internet of Things Journal*, 2021. Li et
  al., "DeepFed: Federated deep learning for intrusion detection in
  industrial cyber-physical systems," *IEEE Transactions on Industrial
  Informatics*, 2020. Hernandez-Ramos et al., "Intrusion Detection
  based on Federated Learning: a systematic review," *ACM Computing
  Surveys*, 2025 — the subfield is mature enough to be systematically
  reviewed, not a novel combination on its own.
- **Cosine-similarity trust filtering**: strongly established, and the
  literature also confirms the exact non-IID-vs-malicious confound this
  spec already flags as an open problem, not a bug. Zhu et al.,
  "Byzantine-robust Federated Learning via Cosine Similarity
  Aggregation," *Computer Networks*, 2024. Li et al.'s spatial-temporal
  analysis work (Byzantine-robust FL through spatial-temporal
  clustering, 2021) — essentially the published version of the
  round-to-round EMA trust score this spec requires. SignGuard (ICDCS
  2022) validates pairing cosine similarity with a magnitude/sign check
  specifically against sign-flipping attacks — the exact test attacker
  component 5 requires. A 2025 Byzantine-robust FL paper states plainly
  that gradient-similarity-based defenses degrade under highly non-IID
  data — the literature's version of the finding from the earlier
  prototype run.
- **Boosting for attack classification, and specifically a two-stage
  boosting+autoencoder cascade**: a 2023 paper in *Mathematical
  Biosciences and Engineering* proposes exactly this pairing — LightGBM
  for a first-pass decision, an autoencoder's reconstruction residual
  for a secondary decision on samples the first pass called normal.
  That's the direct precedent for the boosting-first ordering this spec
  now uses. On the federated-boosting side: Chen, Saad, and Poor,
  "Secure federated XGBoost learning for IoT intrusion detection,"
  *IEEE Transactions on Information Forensics and Security*, 2021;
  SecureBoost, *IEEE Intelligent Systems*, 2021 (the federated
  histogram-based approach this spec treats as a stretch goal).
- **FL + SDN mitigation**: El Houda, Hafid, and Khoukhi, "MiTFed: A
  Privacy Preserving Collaborative Network Attack Mitigation Framework
  Based on Federated Learning Using SDN and Blockchain," *IEEE
  Transactions on Network Science and Engineering*, 2023 — closest
  direct precedent for the FL-decides/SDN-enforces structure here.
  Edge-IIoTset itself (Ferrag et al., *IEEE Access*, 2022) is explicitly
  positioned by its authors for federated learning use, not just
  centralized training.

**On novelty**: a handful of searches turned up no single published
system combining all five of these specifically — federated
autoencoder, cosine-similarity trust filtering, a boosting-first
cascade, FL orchestration, and SDN mitigation, together. Every pair or
triple exists somewhere in the literature, but not the full five-way
combination. That's a reasonable basis to frame this as a genuine
integration contribution in the writeup — but it rests on a handful of
targeted searches, not a systematic review, so don't state novelty as
a settled fact in the paper without running a fuller, more exhaustive
literature search first to make sure nothing more recent or more
obscure already does this combination.

---

## 2. Related work (one line each)

From CLAUDE.md's literature grounding. Bibliographic details are as
CLAUDE.md gives them; verify each against the actual paper before citing.

| Work | Year / venue | What it establishes / how this project differs |
|---|---|---|
| Mothukuri et al., "Federated-learning-based anomaly detection for IoT security attacks" | 2021, IEEE Internet of Things Journal | Federated anomaly detection for IoT attacks. This project adds a supervised first stage, poisoning defense and SDN enforcement. |
| Li et al., "DeepFed: Federated deep learning for intrusion detection in industrial cyber-physical systems" | 2020, IEEE Trans. Industrial Informatics | Federated deep IDS for ICPS. No boosting cascade, no update-level Byzantine filtering, no SDN mitigation. |
| Hernandez-Ramos et al., "Intrusion Detection based on Federated Learning: a systematic review" | 2025, ACM Computing Surveys | FL-IDS is mature enough to be systematically reviewed; a federated autoencoder alone is not novel. |
| Zhu et al., "Byzantine-robust Federated Learning via Cosine Similarity Aggregation" | 2024, Computer Networks | Cosine-similarity aggregation defense. This project filters on deltas, adds a MAD relative threshold, norm clip, EMA trust and (from `7d127a0`) a sign check. |
| Li et al., Byzantine-robust FL through spatial-temporal clustering | 2021 | Round-to-round (temporal) trust, essentially the EMA trust score here. From `7d127a0` the EMA trust also gates exclusion. |
| SignGuard | 2022, ICDCS | Pairs cosine with a magnitude/sign check against sign-flipping, the exact test attacker used here. |
| (unnamed) Byzantine-robust FL paper | 2025 | Gradient-similarity defenses degrade under highly non-IID data. Matches the MAD-band widening measured in §4.6. **Needs a full citation before use.** |
| (unnamed) LightGBM + autoencoder-residual two-stage detector | 2023, Mathematical Biosciences and Engineering | Direct precedent for the boosting-first cascade ordering. **Needs a full citation.** |
| Chen, Saad, Poor, "Secure federated XGBoost learning for IoT intrusion detection" | 2021, IEEE TIFS | Federated boosting for IoT IDS. Here boosting is server-trained on a calibration set and broadcast, not federated. |
| SecureBoost | 2021, IEEE Intelligent Systems | Federated histogram boosting. A stretch goal in the spec; not implemented. |
| El Houda, Hafid, Khoukhi, "MiTFed" | 2023, IEEE Trans. Network Science and Engineering | Closest precedent for FL-decides / SDN-enforces. MiTFed uses blockchain; this project adds a cascade and a poisoning filter. |
| Ferrag et al., Edge-IIoTset | 2022, IEEE Access | The dataset, positioned by its authors for FL use. §5.3 lists preprocessing problems found in its released CSV. |

**Added during the build** (not in CLAUDE.md; the details below are from
memory, so verify them before citing):

| Work | Year / venue | Use here |
|---|---|---|
| Cao, Fang, Liu, Gong, "FLTrust: Byzantine-robust Federated Learning via Trust Bootstrapping" | 2021, NDSS | Precedent for giving updates with negative cosine to a reference direction zero weight (ReLU(cos)). Justifies the `opposes_consensus` check (§4.6). FLTrust uses a server-held root dataset for its reference; this project uses the coordinate-wise median delta. |
| Yin, Chen, Ramchandran, Bartlett, "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates" | 2018, ICML | Coordinate-wise median and trimmed mean as robust aggregators, and their breakdown when the attacker fraction reaches the trim fraction or 50%. Relevant to the trimmed-mean-only ablation row and the 50% sweep point. |

**Novelty caveat (carry forward):** see the verbatim "On novelty" paragraph
above. A handful of targeted searches is not a systematic review. Run an
exhaustive search before claiming novelty in the paper.

---

## 3. System architecture as built

### 3.1 Components → modules
| # | Component | Module |
|---|---|---|
| 1 | Data pipeline | `fl_ids/data/pipeline.py`, `repair.py` |
| 2 | Boosting (bootstrap, broadcast, incremental update) | `fl_ids/models/boosting.py`, `boosting_update.py` |
| 3 | Autoencoder | `fl_ids/models/autoencoder.py` |
| — | Cascade decision rule (single shared implementation) | `fl_ids/models/cascade.py` |
| 4 | Flower client | `fl_ids/fl/client.py` |
| 5 | Trust filter | `fl_ids/robustness/trust_filter.py` |
| 6 | Trimmed-mean aggregation | `fl_ids/robustness/aggregation.py` |
| — | Sign-flip test attacker | `fl_ids/robustness/attackers.py` |
| 7 | Custom Strategy (`TrustFilteredStrategy`, subclasses FedAvg) | `fl_ids/fl/strategy.py` |
| 8 | Flower server (real multi-process) | `fl_ids/fl/server.py`, `launch.py` |
| 9 | SDN mitigation bridge (Flask REST) | `fl_ids/sdn/bridge.py` |
| 10 | OpenFlow 1.3 controller (os-ken) + Mininet topology | `fl_ids/sdn/controller.py`, `topology.py` |
| — | Live pcap → features (tshark), live inference | `fl_ids/sdn/feature_extraction.py`, `live_inference.py` |
| 11 | Phase A / Phase B entrypoints, model bundle | `fl_ids/orchestration/` |
| 12 | Evaluation harness | `fl_ids/eval/` |
| 13 | Streamlit dashboard | `fl_ids/dashboard/` |

### 3.2 Cascade decision rule (one definition, used by eval, dashboard, SDN)
- Boosting outputs a class and its probability. If the class is an attack
  and the probability is ≥ `cascade.confidence_threshold` (0.7), output
  that attack type with confidence = that probability. Stop.
- Otherwise the row goes to the autoencoder. If reconstruction error >
  the client's threshold, output `anomalous` with confidence =
  clip(error/threshold − 1, 0, 1). Otherwise output `benign`.
- Bug found and fixed in `854cd3b`: the threshold existed in two config
  fields, and only one was ever read. It now lives only in `cascade.*`.

### 3.3 Boosting stage
- Label source: **server-held calibration set** (5% of data, stratified).
  A design decision; the alternative, client-local labels, was not chosen.
- Cold start: trained before round 1. Clients never filter with an
  untrained model.
- Distribution: serialized LightGBM model bytes in each round's
  `configure_fit` config (`broadcast_every_n_rounds: 1`).
- Boosting always sees **raw** features. The autoencoder sees
  per-client-normalized features (§4.2).
- Incremental update (`54d8d18`): every 3 rounds, clients that survived
  the trust filter surface ≤50 autoencoder-flagged rows each, with
  analyst-confirmed labels (false alarms come back as Normal). The server
  continues training (+50 trees, LightGBM `init_model`) on calibration
  set + accumulated alerts. If the alerts contain a class the model has
  never seen, it **retrains**: continued training can't learn a new
  class, because its predicted probability is ~1e-14, its softmax
  hessian ~0, and `min_sum_hessian_in_leaf` blocks every split. In
  simulation a never-seen class went from 0% to 17% recognized after one
  update (`54d8d18` commit message).

### 3.4 Autoencoder
- Dense MLP, input 91 → 32 → 16 → **8** → 16 → 32 → 91, ReLU between
  layers, linear output. **7,299 parameters** (float32), computed from
  the architecture. Consistent with the harness's measured 583,920
  bytes/round = 10 clients × 2 directions × 7,299 × 4 B.
- MSE loss, Adam lr 1e-3, 5 local epochs, batch 64.
- Trains only on rows the current global boosting model passes (predicts
  Normal, or an attack below 0.7 confidence).
- Threshold = 97th percentile of reconstruction error on the client's
  held-out benign validation slice, recomputed every round.
- Returns a fixed-bin histogram of reconstruction error (50 bins over
  [0, 10]) with each update.

### 3.5 Trust filter + aggregation (as of `7d127a0`)
- Operates on **deltas** (client weights − the round's starting global
  weights), never raw weights.
- Reference direction = **coordinate-wise median** of the round's deltas.
- Per client: cosine similarity to the reference, and the L2 norm.
- **Excluded if any of three holds** (reason recorded per client per round):
  1. `mad_outlier`: similarity < median − 3·MAD (relative, per round).
  2. `opposes_consensus`: similarity < 0 (`min_cosine_similarity`).
  3. `low_trust`: EMA trust < 0.5 (`min_trust_score`). Trust =
     EMA(α=0.3) of (cos+1)/2, starting at 1.0.
- Survivors are norm-clipped to 2× the round's median norm, aggregated
  by **trimmed mean** (15% per side), and the result is added back onto
  the round's starting weights.
- If no client survives, aggregation is skipped for that round.

### 3.6 Test attacker (sign-flip)
- Trains honestly, then sends −5 × (true delta) (`amplification` 5.0).
- Reported example count: `max_client` (claims the largest client's
  count, `e5fd1c4`), so plain FedAvg faces the literature's
  count-inflating attacker. The trust filter and trimmed mean ignore
  counts.

### 3.7 Phase B path
- The Phase A bundle (`fl_ids/orchestration/artifacts.py`) is
  self-describing: boosting model, final global AE weights, each client's
  scaler (mean, scale) and threshold, the z-score clip, cascade settings,
  held-out metrics and provenance (seed, git commit, time). Phase B
  rebuilds everything from the manifest, never from the current config,
  and rejects an inconsistent bundle (thresholds recomputed from the
  saved weights must match the clients' final-round values).
- Mininet: one host per FL client (`h<i+1>` = client `i`, scored with
  that client's scaler and threshold) on one OVS switch; the controller
  is os-ken OpenFlow 1.3.
- Traffic = real Edge-IIoTset `.pcap` replayed with `tcpreplay` from a
  host. Features come from tshark with the authors' field list.
- Bridge: `{device_id, classification, confidence, stage}` → block
  (confidence ≥ 0.85, drop rule on `ipv4_src`), rate-limit (≥ 0.5,
  OpenFlow meter), else allow/monitor.
- Topology is source-agnostic: a switch port can bridge to a physical/VM
  NIC (not built yet).
- Root: Mininet runs under one sudoers rule for the argument-less
  `python3 topology.py`. Parameters travel through a fixed `/tmp` file.

---

## 4. Deviations from the spec (CLAUDE.md), and why

Reviewers will ask about these. Each has a commit.

1. **Controller built on os-ken, not provided** (`5cbb35c`). The spec
   expected an existing custom OpenFlow 1.3 controller and said not to
   default to Ryu. None was available, and the user asked for one to be
   built. It's built on os-ken, the maintained OpenStack fork of Ryu,
   not on abandoned Ryu. Its switch state (flow and meter stats, ports)
   is queried over OpenFlow multipart requests, so what the dashboard
   shows is what `ovs-ofctl dump-flows` shows.
2. **Boosting on raw features, autoencoder on per-client-normalized
   features** (`680e1a7`). The spec's per-client normalization, applied
   to boosting too, destroyed it: identical held-out rows scored ~0.94
   accuracy with raw features vs ~0.06 through a heavily skewed client's
   own scaler (leak-era measurement; the effect's size is what matters).
3. **Per-client z-scores clipped at ±10** (`6bb8317`). Not in the spec.
   Counter-like features (`udp.stream`, `tcp.seq`, `tcp.ack`) can barely
   vary inside one client's Dirichlet shard. In the DDoS_HTTP-holdout
   setup (seed 42), client 5's benign validation rows sat up to 1.41e4
   std out on `udp.stream` (train std 2.93). That client's mean
   reconstruction error was ~7,500 on 293 rows while every other
   client's was < 5, and the run's mean benign validation loss stayed
   at 405.6 → 400.6 over 20 rounds. With the clip it went 0.152 → 0.0146.
   Diagnostic script, not the harness; commit `6bb8317` message.
   Earlier "divergence" (final val loss 6.6e13 in a Backdoor-holdout
   run) was this effect, not unstable training: per-client delta norms
   stayed 0.3–9 throughout.
4. **Incremental boosting update source changed** (`54d8d18`). The spec
   says to update from "aggregated reconstruction-error histograms and
   any new labels surfaced from surviving clients". Histograms carry no
   features, so nothing trainable. Labels live only in the server's
   calibration set. Chosen with the user: surviving clients surface
   capped analyst-confirmed alerts. **This relaxes "raw traffic never
   leaves a client"** for ≤50 confirmed rows per client per update
   round. State it in the paper.
5. **Evaluation uses an in-process simulation** of the same client,
   attacker and aggregation code (`854cd3b`), because ~150 training runs
   per regeneration can't each afford real processes. The real
   multi-process path is exercised by the integration tests and Phase A.
6. **Trust filter exclusion rule extended** (`7d127a0`, decided with the
   user). The spec specifies a MAD relative threshold, and warns against
   fixed cosine cutoffs. Measured problem with MAD alone (20% sign-flip
   attackers, seed 42): honest similarities spread from ~0.15 (heavily
   skewed shards) to ~0.85, so MAD grew and the cutoff fell from +0.32
   (round 1) to as low as −0.78. Attackers at −0.02…−0.57 passed.
   Harness: malicious-client survival **0.775** at 20% (§8.4a). Added
   `opposes_consensus` (cos < 0: the sign boundary, FLTrust precedent,
   not a tuned value) and `low_trust` (EMA < 0.5, so history counts;
   before, the EMA was computed but gated nothing). Diagnostic after the
   change (12 rounds, seed 42, attackers {0,7}): attacker 7 excluded
   10/12 rounds, attacker 0 5/12 (~62% of attacker-rounds vs ~22%
   before). Honest clients excluded 9/96 client-rounds (vs 2/96).
7. **Convergence metric** (`11103fe`). "Rounds to convergence" = first
   round after which validation loss stays within 10% of the run's
   *total improvement* above its best (best + 0.1·(first − best)), and
   the run ends there; diverged runs report NaN. The earlier band,
   best × 1.1, reported NaN for a healthy run converging to ~0.015
   (±0.0015 is below round-to-round noise).
8. **Headline metrics are not accuracy** (`0224a6d`). About 71% of
   traffic is benign, so accuracy tracks benign FPR and *rises* when a
   poisoned autoencoder stops flagging anything. Headlines: attack
   macro-recall (each attack type weighted equally), benign FPR,
   detection F1, zero-day macro-recall, autoencoder-alone macro-recall.
   All variants are scored through one `flag_attacks` +
   `detection_metrics` path; boosting-only means "the cascade rule with
   no backstop", not bare argmax.
9. **Evaluation mirrors deployment normalization** (`0224a6d`). Each
   test row is scored with its own client's scaler and threshold. It
   used to go through client 0's scaler with a threshold pooled across
   clients.

---

## 5. Dataset and preprocessing (implementation details)

### 5.1 Source
- Edge-IIoTset, `DNN-EdgeIIoT-dataset.csv` (the authors' 61-feature
  selection), plus the per-device/per-attack `.pcap` captures for
  Phase B. 24 captures are present locally.
- Loaded 47 of 63 CSV columns (16 dropped per the authors' recipe:
  IDs/timestamps/payload, plus `Attack_label`, which is derivable from
  `Attack_type`).

### 5.2 After cleaning (current code, `6bea978` onward; log 2026-09-24 18:29)
- Rows: 2,219,216 loaded → **1,909,719** after cleaning (0 NaN rows,
  309,497 duplicates dropped).
- MITM repair: 1,214 dataset rows replaced by 1,229 re-extracted rows
  (before dedup).
- **Features after one-hot encoding: 91** (was 95 with the leak, 88
  after the placeholder fix, 91 after the MITM repair added real MITM
  DNS categories).
- 15 classes. Class distribution after cleaning (computed from the
  encoded labels, 2026-09-24):

| Class | Rows | % |
|---|---:|---:|
| Normal | 1,363,998 | 71.42 |
| DDoS_UDP | 121,567 | 6.37 |
| DDoS_ICMP | 67,939 | 3.56 |
| SQL_injection | 50,826 | 2.66 |
| DDoS_TCP | 50,062 | 2.62 |
| Vulnerability_scanner | 50,026 | 2.62 |
| Password | 49,933 | 2.61 |
| DDoS_HTTP | 48,544 | 2.54 |
| Uploading | 36,807 | 1.93 |
| Backdoor | 24,026 | 1.26 |
| Port_Scanning | 19,977 | 1.05 |
| XSS | 15,066 | 0.79 |
| Ransomware | 9,689 | 0.51 |
| Fingerprinting | 853 | 0.04 |
| MITM | 406 | 0.02 |
| **Total** | **1,909,719** | |

- Note: MITM and Fingerprinting are tiny after dedup. Their per-class
  metrics rest on very few test rows. Report supports next to them.

### 5.3 Data-quality findings in the released dataset (possible secondary contribution)
1. **Placeholder label leak** (`c1cc656`). The "field doesn't apply"
   placeholder is spelled `0` or `0.0` depending on the source capture,
   and the spelling follows the label. In `mqtt.topic`,
   `mqtt.conack.flags`, `mqtt.protoname` and `dns.qry.name.len`, every
   Normal row is `0` and every attack row `0.0`, with zero overlap in
   2.2M rows. After one-hot encoding, `mqtt.topic_0.0` alone separates
   benign from attack at **AUROC 1.0000**. After canonicalizing both to
   `0`, the strongest single feature is `tcp.flags.ack` at AUROC ~0.71;
   a test guards the maximum at < 0.9. Found because the zero-day
   experiment showed autoencoder-alone detection of exactly 100% for
   every held-out class. **Any published model trained on this CSV with
   the authors' recipe may be learning the file format.**
2. **Column-shifted MITM rows** (`6bea978`). In the authors' MITM
   capture CSV, values sit under the wrong column names. Row-aligned
   against our tshark extraction of the same pcap: their `tcp.seq` is
   really `udp.time_delta` (agreement 1.000), `udp.time_delta` is
   `dns.qry.type` (0.993), `dns.qry.qu` is `dns.retransmit_request_in`
   (0.993). 98.8% of those rows are in the DNN CSV, so a model learns an
   ARP-spoofing attack with no ARP fields. Repaired by re-extracting the
   authors' own pcap with the same fields. Afterwards, live-extracted
   MITM traffic matches training MITM rows 100%. Modbus (normal) has the
   same corruption, but none of its rows are in the DNN CSV, so it's
   simply out of distribution live.
3. **tshark version drift** (`33205b3`, `0949abf`). tshark 4.x prints
   booleans as True/False (the authors' output: 1/0) and hex as `0x00`
   (the authors': `0x00000000`). Unhandled, this silently zeroed
   `tcp.flags.ack` on every live packet. Share of our live Backdoor rows
   found among the authors' rows: 6.6% before, 0.998 after. Survey of
   captures after both fixes: 12 of 14 attack captures match at
   0.85–1.00, and MQTT normal devices at 0.967–0.999.

### 5.4 Splits and partition (config at `7d127a0`)
- Server calibration set: 5% of all rows, stratified. Boosting trains on
  it. Phase A holds back 20% of it (`calibration_eval_fraction`) to score
  each boosting version.
- Federated pool: the remainder, subsampled to **60,000 rows**
  (stratified) to bound runtime across ~150 evaluation runs.
- **10 clients**, Dirichlet label-skew partition **α = 0.3**, seed 42.
- Per client: 15% test slice, then 15% of the benign remainder as a
  benign validation slice (threshold calibration). The rest is training.
- Observed skew (seed 42): some clients get no benign validation rows,
  e.g. client 5 in Phase A (warning logged; threshold = ∞, so that
  client's autoencoder never flags). Client training sizes vary from
  ~800 to ~12,400 rows (diagnostic, pre-clip partition code, which is
  unchanged by the clip).
- Pending: exact per-client × per-class count table for seed 42, to
  generate after the regeneration (memory-bound while it runs).

---

## 6. Hyperparameters (configs/config.yaml at `7d127a0`) and why

| Parameter | Value | Why |
|---|---|---|
| seed | 42 | fixed everywhere; LightGBM also `deterministic=True`, `force_row_wise`, `num_threads=1` (multi-threaded histograms weren't bit-reproducible: one class's recall swung 0.0–0.88 across identical runs, `680e1a7`) |
| num_clients / α | 10 / 0.3 | spec: 8–10 clients, α ∈ [0.1, 0.5] |
| val_benign / test fraction | 0.15 / 0.15 | |
| normalized_clip | 10 | §4.3 |
| calibration_fraction | 0.05 | |
| LightGBM rounds / lr / leaves | 300 / 0.05 / 31 | with `min_sum_hessian_in_leaf=1`, `lambda_l2=1`: without leaf regularization training diverged (held-out logloss min ~0.18 at iteration ~16, then 7–14 by 300), making benign FPR swing 0.1%↔16% when 1% of calibration rows changed. 31/0.05 gave the best held-out logloss (0.041 vs 0.056 for 127/0.1) (`2009d90`) |
| cascade confidence threshold | 0.7 | spec default |
| AE hidden / bottleneck | [32, 16] / 8 | spec: bottleneck 8 |
| AE lr / epochs / batch | 1e-3 / 5 / 64 | |
| anomaly percentile | 97 | spec default; implies ~3% benign FPR on validation by construction |
| FL rounds | 20 | spec start value; also used for the Phase 6 regeneration |
| trim fraction | 0.15 | spec |
| norm clip multiplier | 2.0 | spec |
| MAD threshold | 3.0 | |
| trust EMA α | 0.3 | 30% new round, 70% history |
| min_cosine_similarity / min_trust_score | 0.0 / 0.5 | §4.6 |
| attacker amplification / claimed count | 5.0 / max_client | §3.6 |
| boosting update every / +trees / alert budget | 3 rounds / 50 / 50 per client | |
| SDN block / rate-limit confidence | 0.85 / 0.5 | spec (tuned in the earlier prototype) |
| pool_subsample_size | 60,000 | runtime |

---

## 7. Evaluation protocol

- Test set = concatenation of every client's own test slice; each row is
  scored with its client's scaler and threshold (deployment-faithful).
- Variants (`fl_ids/eval/variants.py`), same data, same seed:
  `full_pipeline` (trust filter + trimmed mean, boosting pre-filter,
  cascade), `plain_fedavg`, `trimmed_mean_only` (no cosine filter),
  `autoencoder_only` (no boosting pre-filter; detection = AE alone),
  `boosting_only` (cascade rule, no backstop, no FL).
- Ablation run at 30% sign-flip attackers (clients {0, 6, 9} at seed 42).
- Poisoning sweep: full pipeline at 0/20/30/40/50% attackers. Reports
  malicious-client survival rate = share of attacker-rounds in which an
  attacker passed the filter into aggregation.
- Zero-day (leave-one-attack-class-out): for each of the 14 attack
  classes, remove it from the calibration set **and** every client's
  data before partitioning (so it's absent even from scaler statistics).
  Train, then score up to 5,000 of its rows as unseen traffic, each
  assigned to a random client and scored with that client's scaler and
  threshold. Reported by stage: boosting-only detection (cascade rule),
  autoencoder alone, cascade, share added by the autoencoder, boosting's
  top misattributed label, AE reconstruction-error AUROC vs benign, and
  costs (benign FPR, known-attack recall). Done standalone at 0%
  attackers, and per variant/fraction inside the ablation and sweep
  (`zero_day_macro_recall`).
- Communication cost: bytes/round = Σ clients (down + up) of flattened
  float32 weights. The boosting broadcast is not counted (flag it when
  reporting).

---

## 8. Results

### 8.1 Bootstrap boosting (server calibration hold-out)
Source: `runs/phase_a/boosting_metrics.json`, Phase A run 2026-09-24
18:29, code at `6bea978` (post-leak-fix, post-MITM-repair; boosting is
unaffected by later commits, which touch only normalized features and
the trust filter). Scored on the 20% calibration hold-out (19,097 rows).

- Accuracy 0.9626, macro F1 0.8775, weighted F1 0.9597, benign FPR 0.00022.

| Class | F1 | Recall | Support |
|---|---:|---:|---:|
| Backdoor | 0.9744 | 0.9500 | 240 |
| DDoS_HTTP | 0.7268 | 0.6144 | 485 |
| DDoS_ICMP | 0.9993 | 1.0000 | 679 |
| DDoS_TCP | 0.9990 | 1.0000 | 501 |
| DDoS_UDP | 1.0000 | 1.0000 | 1216 |
| Fingerprinting | 0.8000 | 0.6667 | 9 |
| MITM | 0.6667 | 0.5000 | 4 |
| Normal | 0.9792 | 0.9998 | 13640 |
| Password | 0.8238 | 0.7214 | 499 |
| Port_Scanning | 0.8951 | 0.9600 | 200 |
| Ransomware | 0.9231 | 0.8660 | 97 |
| SQL_injection | 0.8263 | 0.7303 | 508 |
| Uploading | 0.7368 | 0.6087 | 368 |
| Vulnerability_scanner | 0.9631 | 0.9400 | 500 |
| XSS | 0.8487 | 0.7616 | 151 |

Observation: without the leak, the classes boosting struggles with
(DDoS_HTTP, Uploading, SQL_injection, Password, XSS) are the
application-layer attacks that look most like benign HTTP/MQTT traffic.
That is the traffic the autoencoder backstop exists for.
Per-class precision and AUROC: pending (the harness's final
`stage_report` output).

### 8.2 Ablation (30% sign-flip attackers, 20 rounds, seed 42, commit `7d127a0`) — COMPLETE
Source: `results/ablation_table.csv`, `results/ablation_zero_day_detail.csv`
(written 2026-09-25 04:48), `results/ablation_per_class_recall.csv`
(2026-09-24 20:48). Attackers = clients {0, 6, 9}. Zero-day columns: each
variant retrained once per held-out attack class (14 × 5 runs), with 30%
attackers in every run.

Full table (exact CSV values):

| Variant | attack_macro_recall | attack_micro_recall | benign_fpr | detection_f1 | zero_day_macro_recall | autoencoder_alone_attack_macro_recall | final_val_loss | rounds_to_convergence | total_communication_bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| full_pipeline | 0.8868 | 0.8876 | 0.0260 | 0.9093 | 0.7308 | 0.4209 | 0.02121 | 3 | 11,678,400 |
| plain_fedavg | 0.8791 | 0.8834 | 0.0309 | 0.9014 | 0.5946 | 0.0000 | 9.199e+14 | NaN (diverged) | 11,678,400 |
| trimmed_mean_only | 0.8785 | 0.8822 | 0.0330 | 0.8982 | 0.5394 | 0.0000 | 4830 | NaN (diverged) | 11,678,400 |
| autoencoder_only | 0.1912 | 0.1337 | 0.0348 | 0.2191 | 0.3250 | 0.1912 | 0.02948 | 9 | 11,678,400 |
| boosting_only | 0.8750 | 0.8799 | 0.0000 | 0.9361 | 0.5178 | n/a | n/a | NaN (diverged) | 0 |

Zero-day detection rate by held-out class (cascade detection for
full_pipeline / plain_fedavg / trimmed_mean_only; AE alone for
autoencoder_only; cascade rule without backstop for boosting_only). Up to
5,000 held-out rows each (MITM 406, Fingerprinting 853 after dedup):

| Held-out class | full_pipeline | plain_fedavg | trimmed_mean_only | autoencoder_only | boosting_only |
|---|---:|---:|---:|---:|---:|
| Backdoor | 0.8060 | 0.7960 | 0.8038 | 0.0036 | 0.8570 |
| DDoS_HTTP | 0.3976 | 0.3244 | 0.3278 | 0.0562 | 0.3194 |
| DDoS_ICMP | 1.0000 | 1.0000 | 1.0000 | 0.8060 | 1.0000 |
| DDoS_TCP | 0.9750 | 0.9238 | 0.9180 | 0.1652 | 0.9026 |
| DDoS_UDP | 0.9028 | 0.9998 | 0.0922 | 0.8914 | 0.0922 |
| Fingerprinting | 0.8429 | 0.8300 | 0.9601 | 0.5064 | 0.8007 |
| MITM | 0.8867 | 0.0000 | 0.0000 | 0.8744 | 0.0000 |
| Password | 0.4228 | 0.3710 | 0.3732 | 0.1252 | 0.3420 |
| Port_Scanning | 0.9814 | 0.9212 | 0.9218 | 0.1310 | 0.9078 |
| Ransomware | 0.5358 | 0.5334 | 0.5332 | 0.0158 | 0.5298 |
| SQL_injection | 0.6144 | 0.5698 | 0.5672 | 0.0966 | 0.5256 |
| Uploading | 0.5128 | 0.4784 | 0.4780 | 0.0954 | 0.4264 |
| Vulnerability_scanner | 0.8346 | 0.1036 | 0.1004 | 0.6802 | 0.1026 |
| XSS | 0.5180 | 0.4734 | 0.4756 | 0.1022 | 0.4426 |

Readings:
- **Zero-day macro-recall: full pipeline 0.7308 vs boosting-only 0.5178**
  (+0.213), vs plain FedAvg 0.5946 and trimmed-mean-only 0.5394 (the same
  cascade with a poisoned AE). This is the backstop's contribution on
  attacks boosting never saw, and what poisoning takes away from it.
- Largest gains are where boosting has nothing to fall back on: MITM
  0.8867 (boosting-only 0.0000), Vulnerability_scanner 0.8346 (0.1026),
  DDoS_UDP 0.9028 (0.0922). Poisoned variants lose them (MITM 0.0000 for
  plain_fedavg and trimmed_mean_only; Vulnerability_scanner ~0.10).
- Anomalies to explain, not hide: (a) plain_fedavg gets DDoS_UDP 0.9998
  > full 0.9028 with a diverged AE (val loss 9.2e14): a blown-up AE still
  thresholds at the 97th benign percentile, so extreme-valued attacks
  clear it (this is what its 0.5946 macro-recall above boosting-only
  reflects). (b) trimmed_mean_only gets Fingerprinting 0.9601 > full
  0.8429. (c) Backdoor: boosting-only 0.857 > full 0.806. Candidate
  cause for (c): full_pipeline's boosting model is incrementally updated
  from alerts during training, boosting_only's is not. Unverified.
- Convergence: full_pipeline converges by round 3 (final val loss 0.0212).
  plain_fedavg (9.2e14) and trimmed_mean_only (4,830) diverge (NaN).
  autoencoder_only converges by round 9 (0.0295).
- Communication: 11,678,400 bytes over 20 rounds for every FL variant
  (= 583,920/round). Boosting broadcast excluded (§8.6).

Harness log lines for the main columns (same run, matching the CSV to 3 dp):

| Variant | Attack macro-recall | Benign FPR | AE-alone attack macro-recall |
|---|---:|---:|---:|
| full_pipeline | 0.887 | 0.0260 | 0.421 |
| plain_fedavg | 0.879 | 0.0309 | 0.000 |
| trimmed_mean_only | 0.878 | 0.0330 | 0.000 |
| autoencoder_only | 0.191 | 0.0348 | 0.191 |
| boosting_only | 0.875 | 0.0000 | n/a |

Readings:
- The trust filter is what keeps the federated AE usable under 30%
  attackers: AE-alone 0.421 vs 0.000 for both baselines.
- The boosting pre-filter roughly doubles AE-alone recall (0.421 vs 0.191).
- On *known* attacks the backstop adds little at the cascade level (0.887
  vs 0.875) and costs ~2.6% benign FPR. Its value has to come from the
  zero-day results.
- Cascade-level recall barely moves between variants because boosting
  (never FL-trained, so unaffected by poisoning) dominates. **Poisoning
  damage is only visible in the AE-alone and zero-day columns. Don't
  report cascade accuracy alone.**

Per-attack-type recall by variant (`results/ablation_per_class_recall.csv`,
written 20:48, same run):

| Attack type | full_pipeline | plain_fedavg | trimmed_mean_only | autoencoder_only | boosting_only |
|---|---:|---:|---:|---:|---:|
| Backdoor | 0.9640 | 0.9550 | 0.9550 | 0.0270 | 0.9459 |
| DDoS_HTTP | 0.6457 | 0.6413 | 0.6413 | 0.0135 | 0.6323 |
| DDoS_ICMP | 1.0000 | 0.9969 | 0.9969 | 0.3302 | 1.0000 |
| DDoS_TCP | 1.0000 | 1.0000 | 1.0000 | 0.1545 | 1.0000 |
| DDoS_UDP | 1.0000 | 1.0000 | 1.0000 | 0.1206 | 1.0000 |
| Fingerprinting | 1.0000 | 1.0000 | 1.0000 | 0.3333 | 1.0000 |
| MITM | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Password | 0.7511 | 0.7553 | 0.7468 | 0.1435 | 0.7511 |
| Port_Scanning | 0.9804 | 0.9216 | 0.9314 | 0.0294 | 0.9020 |
| Ransomware | 0.9250 | 0.9000 | 0.9000 | 0.0000 | 0.9000 |
| SQL_injection | 0.7469 | 0.7510 | 0.7429 | 0.0531 | 0.7429 |
| Uploading | 0.6566 | 0.6446 | 0.6386 | 0.0843 | 0.6386 |
| Vulnerability_scanner | 0.9611 | 0.9572 | 0.9611 | 0.1868 | 0.9533 |
| XSS | 0.7846 | 0.7846 | 0.7846 | 0.2000 | 0.7846 |

### 8.3 Poisoning-resistance sweep (full pipeline, 20 rounds, seed 42, commit `7d127a0`)
Source: `results/logs/poisoning_sweep.log`. **The run was killed on
2026-09-25 04:48** (Claude Code stopped the background job for low
system memory, when the standalone zero-day experiment started loading
the dataset alongside it). This happened at 53 of 70 zero-day runs, and
the sweep writes its CSV/plot only at the end, so they don't exist. The
per-fraction values below are its harness log lines. Zero-day
macro-recall per fraction: pending a rerun.

| Malicious fraction | Attack macro-recall | Benign FPR | AE-alone macro-recall | Malicious-client survival |
|---:|---:|---:|---:|---:|
| 0.00 | 0.884 | 0.0296 | 0.431 | n/a |
| 0.20 | 0.884 | 0.0341 | 0.374 | 0.275 |
| 0.30 | 0.887 | 0.0260 | 0.421 | 0.317 |
| 0.40 | 0.885 | 0.0285 | 0.413 | 0.237 |
| 0.50 | 0.878 | 0.0340 | 0.000 | 0.960 |

- Breakdown at 50%: the coordinate-wise median reference *becomes* the
  attackers' direction, so honest clients look like the outliers. This
  is the expected limit of median-based defenses.
- The 20% point's AE-alone (0.374) is below the 30% point's (0.421).
  Single seed; not yet explained. Candidate: at 20%, the stricter rule
  excluded honest near-zero clients (§4.6's cost). **Needs a
  multi-seed check before any claim about the trend.**

### 8.4 Filter change, before vs after (same harness, 20 rounds, seed 42)
(a) MAD-only filter, commit `11103fe`. Points from the run that was
stopped once this problem was found (`results/logs` of that run, not
kept; values copied from its log lines at the time):

| Point | Attack macro-recall | Benign FPR | AE-alone | Survival |
|---|---:|---:|---:|---:|
| sweep 0% | 0.884 | 0.0296 | 0.431 | n/a |
| sweep 20% | 0.880 | 0.0323 | 0.397 | **0.775** |
| ablation full_pipeline (30%) | 0.883 | 0.0295 | 0.404 | — |
| ablation plain_fedavg (30%) | 0.879 | 0.0309 | 0.000 | — |
| ablation trimmed_mean_only (30%) | 0.878 | 0.0330 | 0.000 | — |

(b) New filter, `7d127a0`: §8.2 and §8.3. At 30%: AE-alone 0.404 →
0.421, benign FPR 0.0295 → 0.0260. At 20%: survival 0.775 → 0.275,
AE-alone 0.397 → 0.374 (worse). The trade-off is mixed on one seed;
report both sides.

### 8.5 Zero-day (leave-one-class-out)
Per-variant zero-day at 30% attackers: complete, see §8.2. The standalone
per-stage experiment (0% attackers, `zero_day_holdout.csv`) was not run
(regeneration stopped, §8.3). Early log lines (commit `7d127a0`, 20
rounds, seed 42):
- Backdoor held out, ablation (30% attackers): full_pipeline cascade
  detection 0.806, plain_fedavg 0.796, trimmed_mean_only 0.804.
- Backdoor held out, sweep at 0%: 0.855.
- Everything else pending: `results/zero_day_holdout.{csv,png}` and the
  per-variant `ablation_zero_day_detail.csv`.

### 8.6 Communication cost and convergence
- Per round: 583,920 bytes (10 clients × down + up × 7,299 float32
  parameters). Total over 20 rounds: 11,678,400 bytes per FL variant
  (ablation CSV, `7d127a0`).
- Rounds to convergence (ablation CSV): full_pipeline 3; autoencoder_only
  9; plain_fedavg and trimmed_mean_only diverged (NaN). Sweep values:
  pending.
- The boosting broadcast is not included (LightGBM text model; the
  bootstrap `boosting_model.txt` was 11,346,574 bytes on disk, so
  broadcasting it every round dwarfs the AE traffic). **Worth reporting;
  consider `broadcast_every_n_rounds` > 1 or broadcasting only on
  version change.**

### 8.7 SDN / Phase B
- Milestone demo (`5cbb35c`, pre-leak-fix models trained inside the
  demo, not a Phase A bundle): real `Backdoor_attack.pcap` replayed via
  tcpreplay through Mininet/OVS. The switch's counters showed 23,788
  packets transiting. tshark features → cascade → bridge → flow rule
  `priority=100,ip,nw_src=10.0.0.1 actions=drop`, visible in `ovs-ofctl
  dump-flows`. The classification was **DDoS_HTTP, not Backdoor**. The
  leak and tshark-drift findings (§5.3) are the likely causes; not
  re-verified yet.
- **Mitigation latency: not measured yet.** Pending the Phase B run with
  the Phase A bundle.

---

## 9. Do not cite (superseded numbers, with reason)

- Every real-data number from before `c1cc656` (label leak): Phase 5
  milestone numbers (90.5% accuracy, 0.907 weighted F1), leak-era
  boosting 0.983 accuracy, and the Phase 8 demo's reconstruction error
  0.71 → 0.47.
- Everything in `results/stale_pre_6bb8317/`: pre-clip, pre-MITM-repair,
  pre-attacker-count, MAD-only filter. Includes the ablation where plain
  FedAvg "survived" 30% attackers (val loss 0.36), and zero-day val
  losses of 6.6e13 / 3.6e5 (§4.3).
- The Phase 6 commit's (`854cd3b`) described results (e.g. FedAvg val
  loss ~1e16, recall 0.22): leak-era, accuracy-headline era.
- §8.4(a): only as the before/after comparison for the filter change,
  never as results.

---

## 10. Limitations (for the paper)

1. **Non-IID vs malicious data confound** (from CLAUDE.md, confirmed in
   practice). The filter targets corrupted *updates*. A client whose
   *data* is unusual but whose updates are honest can't be told apart
   from an honest heterogeneous client by cosine similarity. Under heavy
   skew, honest small-shard clients sit near 0 similarity and are
   sometimes excluded (~9% of honest client-rounds at 20% attackers,
   diagnostic §4.6).
2. **Late-round attackers look like noise.** After convergence, a
   small-shard attacker's sign-flipped delta is as near-orthogonal
   (+0.02…+0.11) as honest small-shard clients' (0.00…0.07). Direction
   tests can't separate them. Damage is bounded by the norm clip and
   trimmed mean, but the "excluded in most rounds" claim holds for one
   of the two attackers only (10/12 vs 5/12).
3. **Breakdown at 50% attackers** (§8.3): inherent to median-based references.
4. **Trimmed mean alone fails when attackers exceed the trim fraction**
   (15% per side over 10 clients trims one update per side, so 3
   colluding attackers get through).
5. **Only one attack model tested** (sign-flip ×5 with inflated count).
   Adaptive attackers (e.g. ones that stay just above cos 0, or
   inner-product manipulation) are untested.
6. **Backstop cost**: the 97th-percentile threshold gives ~3% benign FPR
   by construction. On known attacks the cascade gains ~0.012
   macro-recall over boosting alone at 30% attackers (§8.2).
7. **Privacy relaxation**: the incremental boosting update ships ≤50
   analyst-confirmed flagged rows per surviving client per update round
   (§4.4).
8. **Clients with no benign validation data** (Dirichlet skew) get an
   infinite threshold, so their AE never flags. Reported, not hidden.
9. **Tiny classes**: MITM (406 rows) and Fingerprinting (853) after
   dedup. Per-class numbers rest on few test rows.
10. **Single seed so far.** Every Phase 6 number is seed 42. Multi-seed
    variance is pending and needed before claiming trends like §8.3's
    20% vs 30% point.
11. **Subsampled federated pool** (60k rows) for runtime.
12. **Evaluation runs in-process**: same code, but no network effects;
    the real multi-process path is covered by tests and Phase A.
13. **Boosting broadcast size** (~11 MB/round) isn't in the
    communication-cost number (§8.6).
14. **Dataset corrections**: departing from the authors' preprocessing
    (§5.3) makes results not directly comparable to papers using the
    raw CSV. Argue that those papers are inflated, carefully.
15. **Novelty** rests on targeted searches only (CLAUDE.md caveat).

---

## 11. Figures and tables to generate

- [ ] Poisoning-resistance curve: x = malicious fraction; y = AE-alone
      macro-recall, zero-day macro-recall, attack macro-recall; second
      panel: malicious-client survival. (The harness writes
      `poisoning_resistance_sweep.png`; check its style.)
- [ ] Ablation table (§8.2 plus the CSV's zero-day, F1, convergence and
      bytes columns).
- [ ] Per-attack-type recall heatmap: variant × class (§8.2 table).
- [ ] Zero-day per-class bar chart: boosting-only vs AE-alone vs
      cascade, 14 held-out classes (`zero_day_holdout.png` exists as a
      harness output; check it).
- [ ] Trust-score trajectories per client over rounds, honest vs
      attacker, 20% and 30% (from the strategy's round history /
      simulation records). Mark exclusion rounds by reason.
- [ ] Similarity-distribution-per-round figure showing the MAD band
      widening past the attackers (§4.6). Strongest visual for the
      filter-change argument.
- [ ] Reconstruction-error distributions: benign vs known attacks vs
      zero-day class, per client, with the threshold line.
- [ ] Benign validation loss per round: clipped vs unclipped
      normalization (§4.3).
- [ ] Data-quality figure: single-feature AUROC before/after the
      placeholder fix (§5.3.1).
- [ ] Per-client × per-class partition heatmap (seed 42, α = 0.3).
- [ ] Architecture diagram: two phases, cascade, filter, SDN.
- [ ] Phase B timeline: replay start → classification → flow rule
      (latency, once measured).

---

## 12. Pending (next phases will fill these)
- Final CSVs of the Phase 6 regeneration: ablation (zero-day, F1,
  convergence, bytes), sweep, zero-day per class. Then per-class
  precision/F1/AUROC per stage.
- Multi-seed repeats of the key points (at least 20% and 30%).
- Real-data Phase A to completion → bundle → Phase B with a real pcap →
  mitigation latency.
- Phase 9 end-to-end recorded run.
