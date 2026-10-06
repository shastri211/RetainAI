# RetainAI 2.0 — Engineering Roadmap

Status: **Phase 0 proposal — pending owner approval** (re-sequencing is decision D-05)
Related: [PRD.md](PRD.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [DATA_STRATEGY.md](DATA_STRATEGY.md)

No effort estimates are given: none can be justified before the environment, data contract and team size are known. Phases are ordered by dependency and risk, and each ends at an approval gate.

## 1. How this differs from the suggested structure, and why

| Suggested | Proposed | Reason (evidence) |
|---|---|---|
| 1 Data Foundation | 1 Foundation & Data | Adds environment repair, packaging, tests: the venv is broken and `requirements.txt` is empty (ARCHITECTURE.md R3, R4). |
| 2 ML Baseline | 2 Risk Baseline | Unchanged in substance. |
| 3 Explainable Risk | 3 Explainable Risk & Value | Adds revenue-at-risk / CLV proxy: it is the only "worth intervening on" signal the data can support (DATA_STRATEGY.md §4). |
| 4 Uplift | **7** Uplift & Operational Data Layer | Uplift is blocked on a data decision (D-03), is the highest-uncertainty work, and cannot be validated on Telco. Keeping it before RAG/UI would put the riskiest item on the critical path to a usable product. The decision engine takes an `effect_source` interface so uplift slots in later (ARCHITECTURE.md §3). |
| 5 Knowledge + RAG | 4 Policy & Knowledge | Deterministic policy engine first; RAG is thin and narrative-only (ARCHITECTURE.md §5). |
| 6 Decision Engine + Agent | 5 Decision Engine & Bounded Workflow | Fixed sequence first; LLM tool loop only if evaluation justifies it. |
| 7 API + UI | 6 API + UI + Approval Workflow | Merges the approval ledger so the MVP is a complete loop through human decision. |
| 8 Intervention + Outcome | 8 Outcome Tracking & Measurement | Needs the Phase 7 holdout design. |
| 9 Hardening | 9 Production Hardening | Unchanged. |
| 10 Evaluation + Release | 10 Evaluation & Release | Evaluation harnesses are built incrementally in each phase; Phase 10 consolidates. |

Milestones:
- **M1 (end of Phase 3):** analyst demo — risk, drivers, revenue-at-risk on Telco. Deliverable and testable without any LLM.
- **M2 (end of Phase 6):** **MVP** — evidence tier 1 (assumed effects, no causal claims), complete loop through human approval.
- **M3 (end of Phase 8):** V1 — uplift on simulated/public randomized data and outcome measurement, with provenance labels.
- **M4 (end of Phase 10):** release.

## Phase 1 — Foundation & Data

- **Objective:** a reproducible environment and a trustworthy, versioned data layer.
- **Inputs:** `data/raw/Telco-Customer-Churn.csv`; notebooks 01–02 (as references only); owner decisions D-13, D-15, D-16.
- **Deliverables:** pinned `requirements.txt`; `src/retainai/` package skeleton; `tests/`; data contract (schema, allowed values, types) as pydantic models; documented cleaning rules (blank `TotalCharges`, structural-redundancy encoding, duplicate-aware group IDs); processed dataset with provenance label and source hash; leakage review notes for `Contract`/`TotalCharges`; minimal CI or a documented local test command; `.gitkeep` for tracked directories; a clean-clone setup README section.
- **Dependencies:** none beyond approvals.
- **Acceptance criteria:** fresh clone → one documented command sequence → tests pass; processed data regenerates byte-identically (same hash) with a fixed seed; contract tests fail on a corrupted copy of the data (blank `Churn`, unknown category); the 11 blank-`TotalCharges` rule is documented and tested.
- **Risks:** Python/library version drift (existing notebooks ran in an unrecorded conda environment); notebook-only logic being re-implemented differently.
- **Approve before starting:** Python version and dependency tool (D-15); how to treat the uncommitted notebook 02 (D-16); whether to touch the empty `LICENSE`/dataset redistribution question now (D-13).

## Phase 2 — Risk Baseline

- **Objective:** an honest, calibrated churn-risk baseline with uncertainty.
- **Inputs:** Phase 1 processed data; D-14 decision on protected attributes.
- **Deliverables:** prior baseline, logistic regression, one gradient-boosted model; repeated stratified, duplicate-aware CV; metrics (precision, recall, F1 at cost-chosen threshold, ROC-AUC, PR-AUC, Brier, reliability curve) with intervals in `reports/`; calibration step; model card draft; saved artefact with data hash and seed.
- **Dependencies:** Phase 1.
- **Acceptance criteria:** metrics reproducible from one command; confidence intervals reported; complex model adopted only if it beats logistic regression beyond CV noise; probabilities pass a calibration check on held-out folds; leakage ablation (with/without `Contract`, `TotalCharges`) reported.
- **Risks:** noisy metrics on 7k rows; optimistic validation (no time split possible); label horizon unknown.
- **Approve before starting:** whether protected attributes may be model inputs (D-14).

## Phase 3 — Explainable Risk & Value  (→ M1)

- **Objective:** per-customer and global explanations, segments, and revenue-at-risk.
- **Inputs:** Phase 2 model.
- **Deliverables:** global importance and per-customer drivers with "association, not cause" wording; segmentation tables; revenue-at-risk; assumption-labelled value proxies (survival CLV not adopted, D-18); fairness audit by gender and senior-citizen status; analyst demo notebook/CLI; completion of the EDA findings in notebook 02 reconciled with model results.
- **Dependencies:** Phase 2.
- **Acceptance criteria:** explanation stability check across CV folds; every CLV assumption listed and each output labelled "proxy"; fairness metrics reported with decisions recorded; reviewer can reproduce a customer's drivers from stored artefacts.
- **Risks:** explanations mistaken for causal advice; CLV proxy overread.
- **Approve before starting:** value-proxy assumptions (assumed months), wording standards.

## Phase 4 — Policy & Knowledge

- **Objective:** make "what may we offer?" deterministic, and add thin retrieval for guidance.
- **Inputs:** Phase 3 outputs; authored offer catalog and playbook (SIMULATED knowledge).
- **Deliverables:** offer catalog and eligibility rules in versioned config with unit tests; `knowledge/` playbook (small, labelled SIMULATED); chunking and indexing; retrieval evaluated against BM25 baseline; hand-labelled retrieval question set; go/no-go note on RAG scope (RAG-06).
- **Dependencies:** Phase 3 for segments; D-08 (vector store), D-09 (embedding location).
- **Acceptance criteria:** eligibility engine has tests for every rule and returns rule IDs; retrieval recall@k and source accuracy reported on the question set; "no relevant guidance" returned when appropriate; corpus carries provenance labels.
- **Risks:** authored policy is arbitrary (we invent it); retrieval looks good only because the corpus is tiny.
- **Approve before starting:** the offer catalog and policy content (you own the business assumptions); RAG go/no-go criteria.

## Phase 5 — Decision Engine & Bounded Workflow

- **Objective:** produce validated, auditable recommendations from a fixed tool sequence.
- **Inputs:** Phases 2–4.
- **Deliverables:** `effect_source` interface with `ASSUMED` implementation; expected-net-value calculator with defined cost semantics; fixed workflow with trace; LLM rationale behind a provider interface; validator (schema, number traceability, citation resolution, eligible-only offers) with templated fallback; evaluation set of customers with checks.
- **Dependencies:** Phases 2–4; D-09 (provider), D-10 (agent scope).
- **Acceptance criteria:** 100% of recommended offers are in the eligible set (test); validator rejects planted bad outputs (a wrong number, a made-up citation, an ineligible offer); stub-provider run reproduces the full pipeline offline; trace stored for each recommendation.
- **Risks:** LLM variance; hidden prompt injection via retrieved text; assumed save rates driving results.
- **Approve before starting:** provider and data policy (D-09); agent scope (D-10); assumed save rates and cost definitions.

## Phase 6 — API + UI + Approval Workflow  (→ M2, MVP)

- **Objective:** a usable review loop with human approval and an audit trail.
- **Inputs:** Phase 5; D-07 (persistence), D-11 (UI stack).
- **Deliverables:** FastAPI endpoints for ranked list, customer detail, recommendation, approve/reject/override; SQLite-backed ledger (append-only); UI for J1 and J2; provenance and `ASSUMED` badges everywhere; CSV export of approved actions.
- **Dependencies:** Phase 5.
- **Acceptance criteria:** end-to-end test from raw data to approved action; override to ineligible option impossible; audit log append-only (test); badge present on every effect figure (UI test); concurrent-access limits documented.
- **Risks:** UI effort crowding out evaluation; users ignore badges.
- **Approve before starting:** UI stack; MVP exit test from PRD §16.

## Phase 7 — Uplift & Operational Data Layer  (gated by D-03)

- **Objective:** estimate effects only where the data permits, and exercise the closed loop with SIMULATED data.
- **Inputs:** D-03/D-04 approvals; Hillstrom (and optionally others) downloaded with licence checked; Phase 5 `effect_source` interface.
- **Deliverables:** uplift evaluation (Qini/AUUC, uplift-at-k) validated on a public randomized dataset; a versioned SIMULATED campaign generator with documented effect model (including no-effect and negative-effect groups); uplift estimators evaluated on whether they recover the *assumed* effects; `SIMULATED` / `ESTIMATED_RANDOMIZED` effect sources wired into the decision engine; wording rules enforced in outputs.
- **Dependencies:** Phase 5; data decisions.
- **Acceptance criteria:** estimators evaluated on a held-out randomized split, with intervals; recovery experiment documented with its circularity caveat; no output states an uplift on Telco customers as a finding; dataset licences recorded.
- **Risks:** circular simulation; overfitting uplift models (effects are small and noisy); licence limits.
- **Approve before starting:** D-03 data strategy; which public dataset(s) (D-04); simulator assumptions (you must agree the effect model before we code it).

## Phase 8 — Outcome Tracking & Measurement  (→ M3, V1)

- **Objective:** record interventions and measure incremental retention honestly.
- **Inputs:** Phase 7; Phase 6 ledger.
- **Deliverables:** intervention and outcome tables; holdout allocation design (random, reproducible); reports for gross vs incremental retention, revenue protected, cost, ROI and false-positive cost with provenance labels; reports state "no causal estimate possible" when no holdout exists.
- **Dependencies:** Phases 6–7.
- **Acceptance criteria:** simulated end-to-end campaign yields a report whose estimate is checked against the known simulated truth within stated intervals; all simulated figures labelled "illustrative".
- **Risks:** reports read as real evidence; small holdouts.
- **Approve before starting:** outcome horizon H; holdout policy.

## Phase 9 — Production Hardening

- **Objective:** make it safe and repeatable to run.
- **Inputs:** M2/M3 builds.
- **Deliverables:** Docker images and compose; PostgreSQL migration only if D-07 trigger met; dependency scan; secrets handling review; logging; rate/limit and failure handling for the LLM provider; backup/restore for the ledger; authentication if multi-user is in scope.
- **Dependencies:** Phases 6–8.
- **Acceptance criteria:** clean-machine deploy from docs; failure injection for LLM outage falls back to template; security checklist passed.
- **Risks:** effort sink if there is no deployment target.
- **Approve before starting:** whether any deployment target or real data is in scope.

## Phase 10 — Evaluation & Release

- **Objective:** one consolidated, reproducible evaluation and a release.
- **Inputs:** all phases' reports.
- **Deliverables:** consolidated evaluation report at all four metric levels (§2 below); model card; known-limitations section; demo script; README rewritten to the 2.0 positioning; release tag.
- **Dependencies:** all.
- **Acceptance criteria:** every reported figure regenerates from one command; every figure shows dataset, split method and provenance label; limitations (fictional data, assumed/simulated effects, undefined horizon) stated prominently.
- **Risks:** pressure to present simulated results as evidence.
- **Approve before starting:** release scope and wording.

## 2. Success metrics (no benchmark numbers are claimed)

| Level | Metrics | Notes |
|---|---|---|
| **Data** | Schema-contract pass rate; missing/blank rate; duplicate-content rate; category drift; reproducible data hash | Blank `TotalCharges` = 11 rows, duplicate-content groups = 20 at baseline (DATA_STRATEGY.md §3). |
| **ML (risk)** | Precision, recall, F1 at a cost-chosen threshold; ROC-AUC; PR-AUC (primary given 26.5% prevalence); Brier score and reliability curve; subgroup error and selection rates | Reported with CV intervals; thresholds chosen by cost, not 0.5. |
| **ML (uplift, V1)** | Qini coefficient, AUUC, uplift-at-k, calibration of uplift deciles | Only on randomized data; target numbers set only after seeing a baseline. |
| **RAG** | Retrieval recall@k; source/citation accuracy; "no answer" correctness | On a hand-labelled question set; compare against BM25. |
| **LLM** | Faithfulness (claims supported by inputs); numeric traceability (100% required by construction); recommendation validity (offer ∈ eligible set; 100% required); fallback rate | Validator failures counted, not hidden. |
| **Business** | Intervention success rate; incremental retention vs holdout; revenue protected; intervention cost; ROI; false-positive cost (offers to customers who would have stayed) | Only meaningful with a holdout; on SIMULATED data labelled "illustrative". |

## 3. Cross-cutting rules

1. Each phase ends with a review; the next starts only on explicit approval.
2. Every phase writes its acceptance checks as automated tests or scripts.
3. Evaluation assets (question sets, validator test cases, simulated truths) are created in the phase that needs them.
4. Any change to scope or order is recorded in DECISIONS.md.
