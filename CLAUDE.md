# FL-IDS: Federated Learning-Based Intrusion Detection System

## What you're building

A federated learning intrusion detection system for IoT/IIoT traffic,
combining five specific techniques that are already decided — don't
substitute or "improve on" these, they're the result of a prior
literature review and a tested proof-of-concept. Two of them form a
deliberate **cascade**, in this order:

1. **A boosting ensemble** (LightGBM/XGBoost) as the **first-pass**
   classifier — supervised, trained on known attack-type labels,
   handles the bulk of traffic decisively and cheaply.
2. **Client-side autoencoders** as the **second-pass, unsupervised**
   check — runs only on traffic the boosting stage classified as
   "normal," to catch zero-day/novel attacks a supervised classifier
   (bounded by its known label taxonomy) would miss. Raw traffic never
   leaves a client.
3. **Cosine-similarity trust filtering** — server-side defense that
   screens client autoencoder weight updates for poisoning before
   aggregation.
4. **Federated learning** (Flower) as the orchestration layer tying
   clients and server together for the autoencoder stage.
5. **SDN** (Mininet + OpenFlow 1.3) enforcing mitigation — the combined
   boosting + autoencoder classification translates into flow-table
   actions that block/rate-limit malicious devices.

This is a real academic project, not a demo — it needs to produce
defensible evaluation numbers (accuracy, poisoning-resistance,
ablations) as well as actually run.

**These five — IDS (the detection/classification outcome), FL, the
autoencoder, cosine-similarity trust filtering, and SDN — are the
actual point of the project, not supporting plumbing.** Data loading,
the Flower server process, the orchestration scripts, the dashboard —
all of that exists to make these five things demonstrably work and be
visible working. Don't let any of them end up as a thin pass-through
that's technically present but not doing anything substantive:

- **IDS** — the end classification outcome is a two-stage cascade:
  boosting's attack-type label when it's confident, falling through to
  the autoencoder's reconstruction-error anomaly flag when boosting
  says "normal" but reconstruction error disagrees. Must be evaluated
  stage-by-stage (boosting alone, autoencoder alone, cascade together)
  with real per-attack-type metrics, not just "it outputs a label."
- **FL** — real multi-process federated rounds against non-IID data,
  not a single-process stand-in.
- **Autoencoder** — reconstruction-error behavior needs to be visibly
  correct: low and stable on benign traffic, elevated on attacks
  (especially attacks unlike anything boosting was trained on), visible
  in both the evaluation output and the dashboard.
- **Cosine similarity filtering** — its effect must be *measurable*:
  the ablation table (component 12) needs a row that isolates what it
  contributes, and the dashboard needs to show trust scores live, per
  client, per round.
- **SDN** — mitigation actions need to be real, observable flow-table
  changes driven by real classifications on real traffic, not a
  logging stub standing in for the controller call.

## Literature grounding

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

## Engineering principles — apply these from the first commit, not retrofitted later

- **Git from the start.** `git init` before writing any code. Commit at
  the end of every milestone below (see Phases), with messages that say
  what changed and why, not just "wip".
- **Modular structure**, one concern per module — mirrors the component
  list below. Don't let the autoencoder, the FL wrapper, and the
  filtering logic bleed into one file.
- **Config, not magic numbers.** All tunable parameters (thresholds,
  learning rates, round counts, trim fractions) live in one config file
  (YAML or a dataclass), not scattered as literals through the code.
- **Type hints throughout**, and docstrings on every public function and
  class — what it does, its inputs/outputs, and anything non-obvious
  about why it's built the way it is.
- **Logging, not print statements.** Use Python's `logging` module with
  configurable verbosity, so a real run's output is usable, not a wall
  of prints.
- **Tests are not optional.** Unit tests per module (`pytest`), plus one
  integration test that runs the full pipeline end-to-end on synthetic
  data and asserts on behavior, not just "didn't crash" — see the
  Phase 4 acceptance criteria for a concrete example of what that means.
- **Fixed random seeds** everywhere randomness is involved, so runs are
  reproducible and bugs are debuggable.
- **Pin dependency versions** in `requirements.txt` as you install them,
  don't leave them floating.
- **Correctness and clarity before optimization.** Don't pre-optimize
  training loops or data pipelines; get them right and readable first.

## Tech stack

Python 3.11+, PyTorch (autoencoder), Flower (`flwr`) for FL
orchestration, LightGBM for the boosting stage, Mininet + Open vSwitch
for the SDN simulation, an existing custom OpenFlow 1.3 controller
(provided separately — ask the user for it when you reach Phase 7; it
was built as a substitute after the Ryu project turned out to be
broken/unmaintained, so don't default to installing Ryu). Dataset:
Edge-IIoTset — on Kaggle
(`mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot`).
You need **two different slices of this dataset for two different
purposes**, don't conflate them: the pre-selected
`DNN-EdgeIIoT-dataset.csv` (61 engineered features, already reduced
from the raw 1176) for Phase A training — don't redo feature
engineering the dataset authors already did — and the **raw per-device
`.pcap` captures** (the full dataset ships one `.pcap` alongside each
device's `.csv`, e.g. `Distance.pcap`, `Flame_Sensor.pcap`) for Phase B
traffic simulation. Phase B must replay real captured traffic via
`tcpreplay`, not a synthetic generator standing in for it — synthetic
generation is a fallback only if a specific capture turns out unusable,
not the default plan.

Dashboard: a Python-native option (Streamlit is a reasonable default,
since it can read straight from the same logged pipeline state with no
separate frontend build) that can refresh live while Phase A/B are
running. Choose whatever fits, but it must read real, live state from
the running pipeline — logged metrics, live trust scores, live flow
table — never mocked or pre-recorded placeholder data.

## Mandatory architecture decision: two phases, not one continuous loop

Build this as **two separate phases**, not an always-on system where FL
training and SDN mitigation run simultaneously — that conflates two
different problems (ML correctness and distributed networking) and
makes both harder to debug.

- **Phase A — offline FL training.** Clients run FL rounds against the
  partitioned dataset. Output: a trained global autoencoder + boosting
  classifier, saved to disk. This is where evaluation numbers come from.
- **Phase B — online mitigation demo.** Load the Phase A models, bring
  up the Mininet topology, replay/generate traffic, run live inference
  through the boosting-then-autoencoder cascade, have the SDN
  controller act on classifications.

Do not start Phase B work until Phase A is fully evaluated (see Phases
below) — you want defensible ML results even if the networking side
runs into environment issues.

## Cascade decision rule

Components 2, 3, 9, 12, and 13 all reference the boosting stage being
"confident," or the cascade "falling through" to the autoencoder — this
needs one precise, shared definition, not an ad-hoc one invented
separately per component:

- Boosting produces a predicted class + that class's probability. If
  the predicted class is a known attack type **and** that probability
  is ≥ a confidence threshold (start at 0.7, make it configurable), the
  cascade's final output is that attack type, with `confidence` =
  boosting's probability. The autoencoder is not consulted.
- Otherwise — boosting predicts "normal," or predicts an attack type
  without enough confidence — the sample passes to the autoencoder. If
  its reconstruction error exceeds the calibrated benign threshold
  (component 3), the final output is `"anomalous"` (a zero-day signal,
  deliberately not a specific attack type — that's the point of this
  stage), with `confidence` derived from how far past the threshold the
  error sits (e.g. `min(1.0, error / threshold - 1.0)`, clamped to
  `[0, 1]` — the exact mapping is configurable, but pick one and apply
  it consistently across evaluation, the dashboard, and the SDN bridge,
  not a different ad-hoc formula in each). If reconstruction error is
  within the threshold, the final output is `"benign"`.

Without this rule, component 12's "cascade together" evaluation and
component 13's "which cascade stage made the call" dashboard panel have
nothing precise to compute or display.

## Components

### 1. Data pipeline
Loads `DNN-EdgeIIoT-dataset.csv`, drops ID/timestamp columns, encodes
categoricals, **normalizes per-client** (never globally — global
normalization leaks cross-client distribution info and defeats the
non-IID test you're trying to run), and partitions clients non-IID via
Dirichlet(α=0.1–0.5) (make α configurable — you'll want to sweep it).
Carves out a held-out benign validation slice per client (for
reconstruction-error thresholding) and a held-out test slice (for
evaluation). Output: `{client_id: {"X": ..., "y": ..., "X_val_benign":
...}}`. Start with 8-10 simulated clients for Phases 2-4 (small enough
to debug by hand), scale up for the real poisoning-fraction sweeps in
Phase 6.

### 2. Boosting classifier — first-pass (server-side, bootstrapped early)
LightGBM, trained on labeled attack-type data. This is deliberately the
**first component built after the data pipeline**, before any FL or
autoencoder work starts — it needs to exist and be reasonably decent
*before* client autoencoders can be trained on its output (see
component 3). Decide explicitly (and document the decision) where
labels come from: a small server-held labeled calibration set, or
labels clients hold locally. This is a meaningful privacy/design
decision, not a default to fall into silently.

**Cold start matters here.** Bootstrap this on the server-held
calibration set (or a synthetic stand-in during early testing) *before*
round 1 of FL training — if the first few rounds' clients are filtering
their autoencoder training data through a boosting model that hasn't
learned anything yet, the autoencoder ends up training on garbage.
Don't let this be an afterthought.

**Distribution mechanism, not just training.** Unlike the autoencoder,
this model isn't aggregated via Flower's weight-averaging machinery —
it needs to be *broadcast* to clients each round (or every few rounds;
it doesn't need to update as fast as the autoencoder) so they can run
it locally as a filter. Flower's `configure_fit` can carry this in its
config payload (serialized model bytes), or use a simpler shared
file/object store if that's easier — either way, this is a real
implementation decision, not something to gloss over.

Later, incrementally update it (`init_model`) using aggregated
reconstruction-error histograms and any new labels surfaced from
surviving clients each round or every few rounds.

### 3. Client-side autoencoder — second-pass, unsupervised backstop
Dense encoder-decoder (not convolutional — this is tabular flow data),
small bottleneck (start at 8). **Trains only on the subset of local
traffic that the current global boosting model (component 2) classifies
as "normal"** — not on all local traffic unfiltered. This is the
component that catches what a supervised classifier, bounded by its
known label taxonomy, can't: novel or zero-day attack patterns that
don't match anything boosting was trained to recognize. MSE
reconstruction loss. Anomaly threshold = a high percentile (start at
97th) of reconstruction error on the client's benign validation slice —
recompute this every round as the global model updates, don't calibrate
once and freeze it.

### 4. Flower client
Wraps the autoencoder in `flwr.client.NumPyClient`. `fit()` first runs
the current global boosting model locally to filter local traffic down
to the "normal" subset (component 2/3's cascade), trains the
autoencoder on that subset, and returns updated weights + a
reconstruction-error histogram (fixed bins, fixed range) — the
histogram, not raw traffic or raw features, is what later feeds back
into refining the boosting stage. Must be runnable both as a real
separate process (`flwr.client.start_client`) and instantiated directly
for testing.

### 5. Cosine-similarity trust filter (server-side)
Three requirements that come from hard lessons in the earlier
prototype — build these in from day one, don't discover them the hard
way:

- **Operate on weight deltas (`new − old`), never raw weights.** Raw
  weights are dominated by the shared global-model component, so
  cosine similarity between any two clients' raw weights is ~0.99
  regardless of malicious status — the filter does nothing. Compute
  deltas against the parameters sent out at the start of the round.
- **Use a relative, per-round outlier threshold, not a fixed cosine
  cutoff.** A fixed threshold (e.g. "reject anything below 0.5") breaks
  the moment non-IID heterogeneity shifts — honest clients can
  naturally sit at 0.3–0.4 similarity under heavy heterogeneity, so a
  fixed cutoff either rejects everyone or lets attackers through
  depending on the split. Use something like median absolute deviation:
  flag clients whose similarity is a statistical outlier *within that
  round's own similarity distribution*.
- **Pair cosine similarity with a norm-clipping check.** Cosine
  similarity alone is direction-only and misses scaling attacks (same
  direction, blown-up magnitude). Reject or clip updates whose L2 norm
  exceeds some multiple (start at 2×) of that round's median norm.

Keep a running trust score per client (EMA across rounds), not just a
one-round judgment, so a client's history matters.

**Test this against the right attacker.** Corrupting a malicious
client's *local input data* mostly just looks like non-IID
heterogeneity to an unsupervised autoencoder (it never sees labels, so
label-flipping does nothing, and scaled/shifted input just produces
"yet another differently-distributed client"). To actually validate
this defense, the test attacker needs to corrupt the *update itself* —
a sign-flip / gradient-ascent attack (negate and amplify the true
delta before sending it) is the standard Byzantine test case this
component is built for. This is also a real, explicit limitation worth
documenting, not hiding: a malicious client whose local *data* (not its
update-sending behavior) is unusual is genuinely hard to distinguish
from an honest non-IID client with cosine similarity alone. State that
plainly wherever this component is documented rather than implying it's
solved.

### 6. Robust aggregation
Trimmed mean over the trust filter's surviving deltas (configurable
trim fraction, start at 15%), applied to deltas, then added back onto
the round's starting global weights.

### 7. Custom Flower Strategy
Subclasses `FedAvg`, overrides `configure_fit` (to capture the round's
starting parameters — needed by component 5 — and to carry the current
boosting model out to clients, per component 2) and `aggregate_fit`
(runs components 5 and 6, and forwards surviving clients' histograms
back to component 2's incremental update). This is the piece that
actually replaces FedAvg.

### 8. Flower server (real distributed run)
`flwr.server.start_server()` with the custom Strategy, real
`ServerConfig(num_rounds=...)` (start at 20 rounds for early testing,
increase once you're tuning for real convergence numbers in Phase 6).
Needs to run against real separate
client processes (component 4), not just a single-process manual
driver — the manual/in-process version is fine for early testing but
isn't the deliverable.

### 9. SDN mitigation bridge
A small API (REST is fine) that receives `{device_id, classification,
confidence}` — the *combined* cascade output (boosting's label when
confident, else the autoencoder's anomaly flag) — and calls into the
OpenFlow controller (component 10) to install the appropriate flow rule
based on confidence thresholds (start at ≥0.85 confidence → block,
≥0.5 → rate-limit, else allow/monitor — these were tuned and tested in
the earlier prototype; make them configurable but they're a reasonable
starting point, not a placeholder to replace blindly).

### 10. SDN controller + Mininet topology
The existing custom OpenFlow 1.3 controller (user will provide this —
don't rebuild it) plus a new Mininet topology script: one host per
simulated client, switch(es), controller connection. Phase B traffic is
**real captured traffic replayed via `tcpreplay`** against the
per-device `.pcap` files from Edge-IIoTset (see Tech stack) — this is
the default and expected approach, not a nice-to-have.

**Don't hard-code traffic as Mininet-internal-only.** Design the
topology so a switch port can optionally bridge to a real physical/VM
NIC, so traffic from an actual external device (another machine on the
LAN, a Raspberry Pi, a VM) can be routed through the same
OpenFlow-controlled switches as the Mininet-hosted replay, if the user
wants to demo against a real device instead of or alongside simulated
ones. This doesn't need to be built until it's actually needed, but the
topology script shouldn't assume every traffic source is a Mininet host
— keep the ingestion point source-agnostic.

### 11. Orchestration layer
Two entrypoints, matching the two phases: Phase A runs FL training end
to end and saves the final global autoencoder + boosting classifier to
disk. Phase B loads both, brings up Mininet, runs live inference
through the boosting-then-autoencoder cascade on replayed/generated
traffic, and calls component 9 per classification.

### 12. Evaluation harness
Precision/recall/F1/AUROC per class + false-positive rate on the held-
out test set, computed **for each cascade stage separately as well as
combined** — boosting alone, autoencoder alone, the full cascade — so
you can show what each stage actually contributes, not just a final
number. A poisoning-resistance sweep: rerun Phase A at several
malicious-client fractions (0/20/30/40/50%), plot accuracy retained.
An ablation table: full pipeline vs. plain FedAvg vs. trimmed-mean-only
(no cosine filter) vs. autoencoder-only (no boosting pre-filter) vs.
boosting-only (no autoencoder backstop) — same data, same seed, one
comparison table. Communication cost (bytes/round) and
rounds-to-convergence. SDN mitigation latency once Phase B exists.

### 13. Live monitoring dashboard
Reads real, live state from the running pipeline — not mocked data —
and is where all five pillars (see "What you're building") become
visible at once, in one place, instead of scattered across log files.
Two views, matching the two phases:

- **Training view (Phase A):** per-round reconstruction error (global
  and per-client), the boosting filter rate each round (what fraction
  of local traffic boosting passed through to autoencoder training —
  worth watching especially early on, per component 2's cold-start
  note), per-client trust score trajectory (the cosine filter's output
  — the hardest component to eyeball from raw logs), which clients
  survived filtering each round, boosting classifier metrics as they
  update.
- **Live simulation view (Phase B):** current Mininet topology and link
  status, live classifications as they happen (device, attack type,
  confidence, and which cascade stage made the call), current
  flow-table state / active mitigation actions per device, running
  traffic stats.

Wire it to read from wherever component 11 (orchestration) and the
Flower server already log/persist state — don't build a second,
parallel state-tracking system just for the dashboard.

## Phases and milestones

Work through these in order. Each phase has an explicit acceptance
criterion — don't move to the next phase until it's met, and commit to
git at the end of each one.

### Phase 0 — Scaffolding
Repo structure matching the component list above, `requirements.txt`,
config file skeleton, logging setup, `pytest` wired up with an empty
test suite that runs. **Milestone:** `pytest` runs (even with zero real
tests yet) and the repo structure is in place.

### Phase 1 — Data pipeline
Component 1, fully built and unit-tested. **Milestone:** loading the
real dataset produces correctly-shaped, non-IID-partitioned client data
with no NaNs, and a test confirms per-client feature distributions
actually differ from each other (that's what proves the non-IID split
is real, not accidentally uniform).

### Phase 2 — Boosting bootstrap
Component 2, on synthetic labeled data initially (real data comes in
Phase 5) — get a working, reasonably-performing first-pass classifier
*before* touching FL or the autoencoder at all. **Milestone:**
classification metrics (precision/recall/F1) computed on a held-out
synthetic slice, not just "the model trains without erroring," and the
model/distribution mechanism (broadcasting it to a client) is testable
in isolation.

### Phase 3 — Core FL loop on synthetic data
Components 3, 4, 8 — autoencoder trained on boosting-filtered data
(using Phase 2's bootstrap model), Flower client, a *plain* FedAvg
strategy (not the custom one yet), real Flower server + multiple real
client processes. Use a small synthetic dataset for this phase
specifically so you're debugging the FL plumbing separate from data
pipeline and robustness-layer issues. **Milestone:** a real
multi-process Flower run completes several rounds without errors,
reconstruction error trends downward on held-out benign data, AND a
test confirms each client's autoencoder is actually training on the
boosting-filtered subset, not raw unfiltered local data.

### Phase 4 — Robustness layer
Components 5, 6, 7 (the custom Strategy's aggregation half) replacing
plain FedAvg. Build the sign-flip test attacker described in component
5. **Milestone (be specific, this is the important one):** run the
pipeline with a mix of honest and sign-flip-attacking clients and show,
with printed/logged per-round trust scores, that honest clients' trust
stays high while attackers' trust visibly decays over rounds, AND that
the norm-clip or MAD-based filter actually excludes the attacker from
aggregation in most rounds. A test that just checks "the code runs" is
not sufficient here — the assertion needs to be on the actual
separation between honest and malicious trust scores.

### Phase 5 — Real data integration
Swap Phases 2/3/4's synthetic data for the real Edge-IIoTset pipeline
from Phase 1, including wiring component 2's boosting stage to real
attack-type labels. Re-validate every milestone above still holds on
real data — thresholds and hyperparameters calibrated on synthetic data
often need adjustment on real data. **Milestone:** Phases 2-4's
acceptance criteria re-verified on real data.

### Phase 6 — Evaluation
Component 12, in full, including the per-stage breakdown (boosting
alone / autoencoder alone / cascade). **Milestone:** the
poisoning-resistance sweep and ablation table exist as actual output
(a saved CSV/plot), not just code that could theoretically produce
them.

### Phase 7 — SDN integration (highest external risk — budget slack)
Components 9, 10, 11 (Phase B half). Requires the user's existing
controller code — ask for it explicitly if it hasn't been provided.
**Milestone:** a live demo where replaying a real `.pcap` capture
through the Mininet topology produces a cascade classification that
triggers an actual flow-table change, observable via `ovs-ofctl
dump-flows` — not a synthetic-traffic stand-in, and not a
manually-triggered fake classification.

### Phase 8 — Dashboard
Component 13. **Milestone:** with Phase A or Phase B actually running,
every panel described in component 13 shows real values that visibly
change as the run progresses — trust scores updating per round, flow
table state updating as mitigation happens. A panel showing static or
placeholder data doesn't meet this bar.

### Phase 9 — Full end-to-end integration
Run Phase A start-to-finish on real data, save the models, then run
Phase B start-to-finish loading those models, replaying real `.pcap`
traffic, and driving live SDN mitigation via the full cascade — all in
one documented run, with the dashboard live throughout. **Milestone:**
a single recorded run (script + log, or screen recording) that goes
from raw dataset to an actual flow-table change with no manual
intervention or patching in between. This is the point where "each
phase passed its milestone" becomes "the whole thing actually works
together" — treat those as different claims; passing every phase
individually doesn't guarantee this one.

### Phase 10 — Polish
Full README, docstring pass, confirm `pytest` covers every component,
clean up config defaults to the values that performed best in Phase 6's
sweeps.

## What "done" looks like

Two things, not one: (1) Phase 6 has already produced real evaluation
numbers before Phase 7 is even attempted — if the SDN integration runs
into environment trouble, you still have a complete, defensible ML
result; don't let Phase 7 block or delay Phase 6. (2) Phase 9's
end-to-end run actually happened and is documented — a project where
every phase individually passed its milestone but was never run
back-to-back as one pipeline is not done. And throughout all of it, the
five pillars from "What you're building" need to still be the visibly
central thing happening — not obscured by orchestration or
infrastructure code.
