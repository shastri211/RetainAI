# RetainAI 2.0 — Architecture

Status: **Phase 0 proposal — pending owner approval** (see [DECISIONS.md](DECISIONS.md))
Principle: every component must answer **"why does this need to exist?"** with evidence. Anything that cannot is deferred.

## 1. Current repository state (evidence)

Facts gathered on `main` at commit `acaaa9f` plus the working tree. Status values: CONFIRMED / PARTIALLY CONFIRMED / INFERRED / NOT FOUND.

| # | Finding | Status | Evidence | What it proves |
|---|---|---|---|---|
| R1 | The repository contains no application code | CONFIRMED | `git ls-files` lists only `.gitignore, LICENSE, README.md, data/raw/Telco-Customer-Churn.csv, notebooks/01…, notebooks/02…, requirements.txt`; no `.py` files tracked | Everything beyond two notebooks is still to be built. |
| R2 | The README describes an "AI-powered churn prediction platform" with a "web-based interface" and "interactive visualizations" | CONFIRMED (text); NOT FOUND (implementation) | `README.md`; no code, UI or model file anywhere | Documentation describes intent, not a working system. Its "churn prediction platform" framing predates the 2.0 direction. |
| R3 | `requirements.txt` and `LICENSE` are empty | CONFIRMED | Both 0 bytes (`wc -c`) | No declared dependencies; no licence granted for the repository. |
| R4 | The local virtual environment is broken | CONFIRMED | `.venv/pyvenv.cfg` points to `C:\Users\chhav\anaconda3\python.exe` (missing); `.venv` holds only `pip`; system Python 3.13.2 has no pandas | The notebooks cannot be re-run from this repo's environment. Notebook kernel metadata says `base` (conda). Library versions used are unknown. |
| R5 | Empty `app/`, `src/`, `models/`, `data/external/`, `data/processed/` directories exist locally but are not in Git | CONFIRMED | Directory listing; Git does not track empty directories | A fresh clone will not contain the intended structure. |
| R6 | No tests, CI, Docker, config files, or experiment tracking | NOT FOUND | No such files in the tree | No automated validation exists yet. |
| R7 | One dataset, committed, 7,043 × 21 | CONFIRMED | `data/raw/Telco-Customer-Churn.csv`; Git history shows it was renamed from `WA_Fn-UseC_-Telco-Customer-Churn.csv` (history contains the file twice) | Data basis; see DATA_STRATEGY.md. |
| R8 | Notebook 01 is a committed data-understanding notebook (18 cells) | CONFIRMED | `notebooks/01_data_understanding.ipynb` | Established shape, dtypes, no nulls (blanks hidden as `" "`), class balance 73.5/26.5, and the 11 blank `TotalCharges`. Does not decide how to handle them. |
| R9 | Notebook 02 (EDA) has **uncommitted** work and is incomplete | CONFIRMED | `git status`: modified; 24 cells, 532 added lines vs an empty committed file; last code cell empty; table of contents lists 13 sections but content stops at 6.1 | User work in progress. Preserved untouched by Phase 0. |
| R10 | Notebook 02 states causal-sounding business conclusions | CONFIRMED | Cells under "Business Recommendation / RetainAI Application": e.g. offering contract-upgrade incentives "may encourage" commitment; contract type "will be treated as a high-impact feature" | These are associations from observational data. They must not become product claims (PRD §19). |
| R11 | Notebook 02 drops the 11 blank-`TotalCharges` rows | CONFIRMED | `df.dropna(inplace=True)` after replacing `" "` with NaN | A data-cleaning decision made inside a notebook; should become a documented, tested rule (Phase 1). |
| R12 | The Git history has one author and one branch | CONFIRMED | `git log`, `git branch -a`: `main` and `origin/main` only; author `shastri211` | Clean baseline; remote is `https://github.com/shastri211/RetainAI.git`. Repository visibility not checked. |
| R13 | Local `main` was identical to `origin/main` at start | CONFIRMED | `git status`: "up to date with 'origin/main'" | No unpushed commits at Phase 0 start. |

## 2. Environment note (Phase 1 prerequisite)

Because R3 and R4 hold, Phase 1 must start by creating a reproducible environment (Python version, pinned `requirements.txt`) before any modelling. Phase 0 installed nothing.

## 3. Target architecture — logical flow

```
Customer data (PUBLIC, fictional)
    │  validate + version
    ▼
Data layer ─► Analytics / segmentation
    │
    ▼
Risk model (calibrated)  ──►  Explanations (drivers)
    │                              │
    ├────────────► Value (revenue-at-risk, CLV proxy)
    ▼
Offer catalog + Eligibility engine (deterministic policy)
    │
    ▼                      Knowledge (playbook) ─► Retrieval (RAG) ─┐
Effect source ── ASSUMED (MVP) / SIMULATED / ESTIMATED (V1) ◄──────┤
    ▼                                                                │
Decision engine: expected net value per eligible option  ◄──────────┘
    │
    ▼
Bounded workflow (fixed tool sequence) ─► LLM rationale ─► Validator
    │
    ▼
Human approval / override ─► Intervention ledger ─► Outcome tracking ─► Evaluation
                                                                          │
                                           (V1) feeds holdout analysis ◄──┘
```

Design rule: the decision engine consumes `effect(customer, offer)` through a stable interface that always returns a value **and** an `effect_source` (`ASSUMED` | `SIMULATED` | `ESTIMATED_RANDOMIZED` | `MEASURED`). The MVP implements only `ASSUMED`. This keeps uplift (blocked on data) from being on the MVP's critical path without requiring a re-architecture later.

## 4. Technology decisions (each justified or deferred)

Style: **modular monolith**, one Python package (`src/retainai/`). No microservices: nothing here has independent scaling or deployment needs.

| Technology | Why it needs to exist | Verdict |
|---|---|---|
| Python 3.x + `src/retainai` package | All ML/data work is Python; a package gives tests and reuse from notebooks, API and UI. | **Adopt** |
| pandas, scikit-learn | Data handling, baselines, calibration, metrics. | **Adopt (Phase 1–2)** |
| Gradient boosting (LightGBM or XGBoost) | Only if it beats logistic regression beyond CV noise (ML-01). | **Conditional** |
| SHAP (or model coefficients for linear models) | Needed for per-customer drivers (UC2). | **Adopt (Phase 3)** |
| lifelines (survival) | Was proposed for a CLV proxy; the snapshot data cannot validate a survival model (D-18). | **Not adopted** |
| Uplift libraries (scikit-uplift / CausalML / EconML) or plain scikit-learn meta-learners | Needed only in Phase 7. Choose then; scikit-learn T-/S-learners plus own Qini may suffice. | **Defer** |
| pydantic | Typed schemas for data contract, recommendation object, LLM output validation. | **Adopt** |
| pytest | Required for acceptance criteria. | **Adopt** |
| SQLite (via SQLAlchemy) | Approvals, audit log and outcomes need transactional, queryable persistence; SQLite needs no service. | **Proposed MVP default (D-07)** |
| PostgreSQL | Justified only by multi-user concurrent writes or hosted deployment. SQLAlchemy keeps the move cheap. | **Defer to hardening** |
| Qdrant | The playbook corpus will be tens of documents (tens to low hundreds of chunks). Exact in-process search is sufficient; a vector DB adds a service to run. Revisit if the corpus exceeds ~10⁴ chunks, needs concurrent updates, or metadata filtering at scale. | **Defer (D-08)** |
| Embedding model | Required for semantic retrieval. A small local model (e.g. sentence-transformers) avoids sending text out; baseline against BM25 keyword search and keep the simpler one if equal. | **Adopt, evaluated against BM25** |
| LLM API (Groq / Gemini / OpenRouter) | Needed for rationale prose only. Free tiers exist but terms/limits change; choose by structured-output reliability and data terms. | **Adopt behind a thin interface (D-09)** |
| FastAPI | Needed once a UI must call the core over HTTP, and to expose recommendations/approvals as a stable contract. | **Adopt in Phase 6** |
| UI: Streamlit **or** React + Vite | The UI must show ranked lists, a detail view and an approval action. Streamlit is the lowest-cost way to reach that; React/Vite gives a product-grade front end at higher cost and needs the API anyway. | **Decision (D-11)** |
| Background workers (Celery/RQ etc.) | Batch scoring of 7,043 rows takes seconds-to-minutes in-process; nothing requires queues. | **Not adopted** |
| Docker | Reproducible run/deploy; justified once there are ≥2 processes (API + UI) or for release. Not for Phases 1–5. | **Adopt in Phase 9** |
| MLflow / experiment tracker | Metrics as versioned JSON files + git are enough at this scale. | **Not adopted** |
| Orchestration frameworks (LangChain/LangGraph etc.) | A fixed 8-step sequence is plain Python functions; a framework adds dependencies without a need. Revisit only if AG-02 is adopted and needs it. | **Not adopted** |

## 5. ML / AI boundaries

The boundary is a hard rule: each layer lists what it may and may not do.

| Layer | Owns | May produce numbers? | May decide eligibility? | Deterministic? |
|---|---|---|---|---|
| **Deterministic ML** | Churn probability, risk score, calibration, drivers, revenue-at-risk, CLV proxy, (V1) uplift score, cost and expected-value arithmetic | **Yes — only this layer** | No | Yes (seeded) |
| **Policy engine (rules)** | Offer eligibility, limits, budget caps, exclusions | No | **Yes — the only authority** | Yes |
| **RAG** | Retrieving playbook/talk-track/escalation passages with citations | No | No (informs, never authorises) | Yes given index (retrieval) |
| **LLM** | Natural-language rationale and contact brief from structured inputs | **No** (may only restate numbers it was given) | No | No (output validated) |
| **Bounded workflow** | Fixed order of read-only tool calls; assembling the recommendation; recording the trace | No | No | Yes in MVP |
| **Human** | Approve, reject, override; execute the intervention | — | Final say within eligible set | — |

Why policy rules are separate from RAG: "is this offer allowed?" must have one reproducible answer, testable by unit tests. Retrieved prose cannot guarantee that. RAG explains and cites; rules decide.

Why the MVP workflow is not an LLM agent: a single-customer recommendation needs the same steps in the same order every time (risk → drivers → value → eligibility → retrieval → ranking → draft → validate). An LLM choosing steps would add variance and cost with no need. An LLM-driven tool loop (AG-02) is admitted later only if it measurably beats the fixed sequence on the evaluation set.

## 6. Recommendation object (contract)

Produced by the decision engine, validated with pydantic, persisted:

```
recommendation:
  id, created_at, customer_ref (pseudonymous)
  data: {dataset_hash, provenance_label}
  risk: {probability, model_version, calibration_method}
  drivers: [{feature, direction, magnitude}]        # associational
  value: {revenue_at_risk, clv_proxy, method, assumptions}
  options: [
    {offer_id, eligible, rule_ids_passed, rule_ids_failed,
     cost, effect: {value, effect_source}, expected_net_value,
     citations: [chunk_id]}
  ]
  rationale: {text, model, prompt_version, validation: pass|fail|fallback}
  workflow_trace: [tool, inputs_ref, outputs_ref]
  status: proposed | approved | rejected | overridden
  review: {reviewer, time, reason, chosen_offer_id}
```

Expected net value (MVP): `P(churn) × assumed_save_rate(offer) × value − cost(offer)`, with `assumed_save_rate` read from policy config. Its limits (save probability proportional to risk; sure-things and sleeping-dogs invisible) must be shown to users. V1 replaces `P(churn) × assumed_save_rate` with an estimated uplift τ, and cost must include discounts given to customers who would have stayed anyway; the exact cost definition is a Phase 5 task.

## 7. Data and storage layout (proposed, to be created in Phase 1)

```
data/raw/        immutable source files (exists)
data/processed/  derived tables, parquet/CSV, provenance-labelled
data/external/   public randomized datasets (Phase 7), not committed if large
knowledge/       authored playbook documents (SIMULATED knowledge), versioned
config/          offer catalog, eligibility rules, assumed effects, budget limits
models/          trained artefacts (gitignored: *.pkl, *.joblib)
reports/         metrics JSON, evaluation reports (committed, small)
src/retainai/    data/, features/, models/, explain/, value/, policy/, knowledge/,
                 decision/, workflow/, llm/, api/, ledger/
app/             UI
tests/
notebooks/
docs/
```

## 8. Security and privacy architecture

- Secrets via environment only; `.env` already git-ignored.
- Pseudonymise `customerID` before any LLM call; send only the structured fields required for the rationale.
- LLM output never executes anything; it is text that passes the validator or is replaced by a template.
- Retrieved and user-entered text is wrapped as data in prompts; tools are read-only; the workflow has no network or file-write tools.
- Audit log is append-only; recommendation versions are immutable once shown.
- Authentication/roles deferred (FUT) because there is no real data or multi-user deployment; revisit before either.

## 9. Observability and quality gates

- Each phase ships an automated check (pytest or script) that encodes its acceptance criteria (ROADMAP.md).
- Metrics are written to `reports/*.json` with git commit, data hash and seed.
- A programmatic faithfulness check (numbers and citations traceable) runs on every generated rationale.
- Model/data drift monitoring is V1.

## 10. Architecture risks

| Risk | Mitigation |
|---|---|
| Over-engineering for a 7k-row dataset | Defer Postgres, Qdrant, workers, frameworks until a stated trigger. |
| Hidden coupling to a specific LLM provider | One interface; swap test with a stub provider. |
| Assumed effects mistaken for estimates | `effect_source` is mandatory in the contract and shown in UI. |
| Simulated layer drifting into "results" | Provenance labels in table names and outputs. |
