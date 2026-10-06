# RetainAI 2.0 — Product Requirements Document

Status: **Phase 0 draft — pending owner approval**
Related: [DATA_STRATEGY.md](DATA_STRATEGY.md) · [ARCHITECTURE.md](ARCHITECTURE.md) · [ROADMAP.md](ROADMAP.md) · [DECISIONS.md](DECISIONS.md)

Conventions: requirement IDs are stable. Tags: **[MVP]**, **[V1]**, **[FUT]**. "Assumed effect" means a number we wrote in a config, not something estimated from data.

## 1. Problem statement

Retention teams rarely lack churn scores; they lack a defensible answer to *who should be contacted, with what, and did it work*. A churn probability alone fails in four ways:

1. A high-risk customer is not necessarily worth an intervention (they may leave regardless, or be low value).
2. A score without reasons cannot be acted on or challenged.
3. Offers must obey business policy (eligibility, limits, cost) that a model does not know.
4. Without tracking outcomes against a comparison group, the programme can never learn whether it helps.

RetainAI 2.0 turns churn prediction into a **decision-support system with an audit trail**: risk → reasons → value → policy-allowed options → evidence-backed recommendation → human decision → recorded outcome.

Honest scope statement: the only customer data in the repository is a public sample for a fictional telco (see DATA_STRATEGY.md §2). It supports risk, explanation and revenue-at-risk. It cannot support any claim that an intervention works. RetainAI must say so wherever effects appear.

## 2. Target users

| User | Need |
|---|---|
| Retention analyst / campaign manager | Build a prioritised, justified contact list within a budget. |
| Retention specialist (agent / account manager) | See why one customer is at risk and what is allowed to be offered, with sources. |
| Programme owner / finance approver | Approve spend, see cost vs protected revenue, see whether it worked. |
| ML / data owner (maintainer) | Monitor model quality, data quality and decision quality; reproduce any past recommendation. |

## 3. Personas (role-based, illustrative)

- **Campaign Manager** — plans monthly outreach with a fixed budget; wants a ranked list and a cost estimate; distrusts black boxes.
- **Retention Specialist** — talks to customers; has minutes per case; needs a short rationale, allowed offers, and a script reference; must be able to say "no, I disagree".
- **Programme Owner** — signs off budget and policy; wants incremental (not gross) retention and an audit log.
- **ML Maintainer** — owns models and data checks; needs versioned artefacts, metrics, drift signals.

## 4. Primary use cases

- **UC1** Rank at-risk customers by calibrated risk and revenue at risk. [MVP]
- **UC2** Explain one customer's risk (top drivers, with an "association not cause" caveat). [MVP]
- **UC3** Show which interventions are *policy-eligible* for a customer and why others are not. [MVP]
- **UC4** Produce a recommendation with a cited rationale and expected net value, flagging every assumption. [MVP]
- **UC5** Human approves, rejects or overrides; every decision is logged. [MVP]
- **UC6** Record that an intervention was carried out and its later outcome. [V1]
- **UC7** Estimate incremental effect (uplift) and rank by expected *incremental* value. [V1; blocked on data decision D-03]
- **UC8** Report incremental retention, cost and ROI against a holdout. [V1]

## 5. User journeys

**J1 — Campaign planning (Campaign Manager).** Choose segment filter and budget → system lists customers by expected net value → each row shows risk, revenue at risk, best eligible option, evidence-tier badge (assumed / simulated / estimated) → manager removes some, approves the batch → approvals logged.

**J2 — Single-customer review (Specialist).** Open customer → see risk and top drivers → see eligible options with rule IDs and cited playbook passages → read generated rationale (every number traceable) → approve / override with reason → mark intervention executed.

**J3 — Programme review (Owner).** Open outcome report → see treated vs holdout retention, cost, net value, with data-provenance labels → decide to continue, change policy or stop.

## 6. Business value

Value is stated as **hypotheses to be tested**, not promised numbers.

- Spend retention budget where expected incremental value is highest instead of where risk is highest.
- Reduce wasted offers to customers who would stay anyway or leave anyway.
- Make policy compliance verifiable (every recommendation passes deterministic rules).
- Create the measurement loop that lets effect be learned over time.

Dataset-internal sizing only (fictional data): churned customers hold 30.5% of the file's monthly charges (DATA_STRATEGY.md §3). This illustrates why the problem matters; it is not a forecast.

## 7. Functional requirements

| ID | Requirement | Tag |
|---|---|---|
| FR-01 | Load, validate and version the customer dataset; reject or flag rows violating the data contract. | MVP |
| FR-02 | Score every customer with a calibrated churn probability and model version. | MVP |
| FR-03 | Global and per-customer explanations of risk drivers. | MVP |
| FR-04 | Revenue-at-risk and a labelled CLV proxy per customer. | MVP |
| FR-05 | Segmentation views (contract, tenure band, service bundle, payment method). | MVP |
| FR-06 | Offer catalog (types, cost, duration, eligibility rules) held as structured, versioned configuration. | MVP |
| FR-07 | Deterministic eligibility engine returning pass/fail with rule IDs for each customer–offer pair. | MVP |
| FR-08 | Knowledge retrieval over the retention playbook with source citations. | MVP (thin) |
| FR-09 | Recommendation object (see ARCHITECTURE.md §6) with ranked eligible options, expected net value, effect source, citations, rationale. | MVP |
| FR-10 | Approve / reject / override workflow with reason capture and immutable audit log. | MVP |
| FR-11 | Batch export of approved actions (CSV). | MVP |
| FR-12 | Record intervention execution and outcome per customer. | V1 |
| FR-13 | Uplift estimation and effect-aware ranking. | V1 |
| FR-14 | Holdout (control) design and incremental-retention/ROI report. | V1 |
| FR-15 | Model and data monitoring (drift, calibration over time). | V1 |
| FR-16 | Multiple concurrent users with roles. | FUT |

## 8. Non-functional requirements

- **Reproducibility:** pinned dependencies, fixed seeds, data file hash recorded with every artefact; any recommendation can be regenerated from stored inputs and versions.
- **Determinism:** the same inputs and versions yield the same risk, eligibility and ranking. Only the prose rationale may vary; it is stored once generated.
- **Latency (targets, to be validated):** single-customer recommendation interactive (seconds); batch scoring of the full dataset in minutes on a laptop. No numbers beyond these have been measured.
- **Simplicity:** runs locally on a single machine without cloud services (except optionally a hosted LLM API).
- **Testability:** deterministic components have unit tests; every phase has an automated acceptance check.
- **Transparency:** every output displays its data-provenance label and effect source.
- **Accessibility/UX:** basic keyboard and colour-contrast compliance in the UI (Phase 6).

## 9. ML requirements

- **ML-01** Baselines first: majority/prior, logistic regression, then a gradient-boosted model. A complex model is adopted only if it beats logistic regression by a margin larger than cross-validation noise.
- **ML-02** Primary metrics: PR-AUC and ROC-AUC; also precision, recall, F1 at a cost-chosen threshold (not 0.5 by default); calibration (reliability curve, Brier score). Report confidence intervals from repeated stratified, duplicate-aware cross-validation.
- **ML-03** Calibrate probabilities (e.g. isotonic or Platt) and verify on held-out data.
- **ML-04** Leakage review of every feature (notably `Contract`, `TotalCharges`; DATA_STRATEGY.md §3).
- **ML-05** Fairness audit by gender and senior-citizen status (error rates, selection rates) before recommendations ship.
- **ML-06** Value proxies from observed `MonthlyCharges` only: monthly revenue, risk-weighted revenue-at-risk, and a CLV proxy using an explicit, configurable assumed number of months, labelled ASSUMED. A survival-based CLV is not adopted (DECISIONS.md D-18). [MVP]
- **ML-07** Uplift models only on randomized data (public Hillstrom for method validation; SIMULATED layer for pipeline), evaluated with Qini/AUUC and uplift-at-k; no uplift claim on Telco without randomized treatment data. [V1]
- **ML-08** No LLM produces, alters or "estimates" a model number.
- **ML-09** Model artefacts versioned with metrics, data hash, git commit.

Known limitation to surface to users: with no effect data, MVP expected value uses *assumed* save rates, and treats save probability as proportional to risk. That is a strong assumption, flagged in outputs.

## 10. RAG requirements

RAG is justified only for **unstructured** guidance: retention playbook narrative, talk tracks, escalation procedure, explanations of offer intent. It is *not* the source of truth for what is allowed.

- **RAG-01** Corpus is a small, versioned set of documents in the repo, authored by us and labelled **SIMULATED business knowledge** (no real company policy exists in this project).
- **RAG-02** Each chunk carries doc ID, version, section, and provenance label.
- **RAG-03** Retrieval returns chunks with scores; the system may say "no relevant guidance found" instead of forcing an answer.
- **RAG-04** Citations in outputs must resolve to retrieved chunks (programmatic check).
- **RAG-05** Retrieval evaluated on a hand-labelled question set (recall@k, source accuracy) before it is wired into recommendations.
- **RAG-06** Go/no-go in Phase 4: if the structured policy engine and a simple lookup meet the need, RAG scope shrinks to the narrative playbook only.

## 11. LLM requirements

- **LLM-01** Allowed roles: turn structured inputs (risk, drivers, value, eligible options, retrieved passages) into a short rationale and a customer-contact brief.
- **LLM-02** Inputs are structured JSON assembled by code; the LLM receives numbers already computed.
- **LLM-03** Output validated: schema-valid JSON; every number appears in the input; every cited ID was retrieved; every named offer is in the eligible set. A failing output is rejected and replaced with a templated fallback.
- **LLM-04** Low temperature; prompt and model name/version stored with each output.
- **LLM-05** Provider isolated behind one small interface so it can be swapped (decision D-09).
- **LLM-06** No customer identifiers or free-text PII are sent; pseudonymous IDs only.
- **LLM-07** Retrieved text is treated as data, never as instructions (prompt-injection resistance).

## 12. Agent requirements

"Agent" here means a **bounded workflow**, not an autonomous system.

- **AG-01** MVP workflow is a fixed, deterministic sequence of approved read-only tools: get risk → get drivers → get value → run eligibility → retrieve guidance → rank by expected net value → draft rationale → validate. No LLM-chosen loop is required for a single-customer recommendation (see ARCHITECTURE.md §5).
- **AG-02** Optional (V1, gated by evaluation): let an LLM choose among the same read-only tools within a hard step limit, kept only if it beats the fixed sequence on the evaluation set.
- **AG-03** The workflow has no write access to any external system; its only side effect is proposing a recommendation.
- **AG-04** Step limit, timeout, and a full trace (tool calls and results) stored per recommendation.

## 13. Human-in-the-loop requirements

- **HITL-01** No intervention is marked actionable without a human approval.
- **HITL-02** Reviewer can approve, reject, or override (choose another eligible option or none), with a mandatory reason on reject/override.
- **HITL-03** Override to a non-eligible option is not possible in the UI; exceptions require a recorded policy-exception flag by an authorised role. [V1]
- **HITL-04** The audit log is append-only and stores who, when, what, and the exact recommendation version shown.
- **HITL-05** Human execution of the intervention is outside the system in MVP; the system records that it happened.

## 14. Outcome measurement requirements

- **OM-01** Every executed intervention stores customer, offer, date, cost, treatment-group flag, and later outcome (retained at horizon H; H is configured).
- **OM-02** Outcome reports compare treated vs a randomly-held-out group; without a holdout, the report states "no causal estimate possible" and shows only descriptive retention.
- **OM-03** Report gross vs incremental retention, revenue protected, intervention cost, ROI and false-positive cost, with provenance labels. Numbers on SIMULATED data are labelled "illustrative".
- **OM-04** Holdout sizing and allocation (random, reproducible) are part of campaign setup. [V1]

## 15. Security and privacy requirements

- **SP-01** No real personal data in the repository. The sample dataset contains no names/contact details; `customerID` is an opaque code, still treated as an identifier.
- **SP-02** Secrets (API keys) only via environment variables / `.env` (already git-ignored); never committed or logged.
- **SP-03** Pseudonymise IDs before any external API call; log what was sent (fields, not content values).
- **SP-04** Treat all retrieved and user-entered text as untrusted input to the LLM.
- **SP-05** Protected attributes (gender; age proxy via senior-citizen) are not used to decide eligibility or ranking; fairness audit (ML-05); modelling use decided in D-14.
- **SP-06** Role-based access and authentication before any multi-user deployment. [FUT]
- **SP-07** Data-retention and deletion policy before any real data is loaded. [FUT]
- **SP-08** Dependency pinning and vulnerability scan in hardening phase.

## 16. MVP scope

The MVP is **evidence tier 1**: Telco-only (PUBLIC, fictional) with *assumed* effects, no causal claims.

Included: FR-01 to FR-11; ML-01 to ML-06, ML-08, ML-09; RAG-01 to RAG-06 (thin); LLM-01 to LLM-07; AG-01, AG-03, AG-04; HITL-01, HITL-02, HITL-04, HITL-05; SP-01 to SP-05.

MVP exit test: an analyst can generate a ranked, budget-limited, policy-compliant, explained recommendation list, a reviewer can approve/override each, and every number and citation in the output is traceable. Every effect-related figure is displayed with the badge "ASSUMED".

## 17. V1 scope

FR-12 to FR-15; ML-07; AG-02 (if evaluation supports it); HITL-03; OM-01 to OM-04. Depends on D-03 (simulated operational layer + public randomized dataset for method validation). Even in V1 the effect badge reads SIMULATED or ESTIMATED-ON-PUBLIC-DATA, never "measured on your customers".

## 18. Future scope

Real-data ingestion and connectors; authentication and roles; scheduled batch scoring; real A/B experimentation workflow; model monitoring dashboards; multi-channel execution integrations; multi-tenant support.

## 19. Explicitly out of scope

- Claiming causal effects of interventions on the Telco data.
- Presenting simulated or synthetic results as real-world evidence.
- Autonomous execution of offers or contact with customers.
- LLM-generated predictions, probabilities, or eligibility decisions.
- Real-time streaming scoring; microservices; mobile apps.
- Pricing optimisation and credit/fraud use cases.
- Handling real personal data (until FUT items SP-06/SP-07 exist).

## 20. Risks

| Risk | Likelihood / impact | Mitigation |
|---|---|---|
| Users read assumed/simulated effects as real | High / High | Mandatory badges; labelled tables; wording rules (DATA_STRATEGY.md §1). |
| Simulated layer is circular | High / Medium | Document generator; predeclared effects; wording "recovers assumed effect". |
| Small dataset yields noisy metrics | High / Medium | Repeated CV, intervals, simple models first. |
| Label horizon undefined | Certain / Medium | State in all docs; do not label scores as "within N days". |
| Leakage via `Contract` / `TotalCharges` | Medium / High | Leakage review (ML-04) before modelling. |
| Scope creep (RAG, agents, UI) | High / High | Phase gates and evidence criteria in ROADMAP.md. |
| LLM hallucination in rationale | Medium / High | Structured inputs, validator, fallback template. |
| Proxy discrimination (age) | Medium / High | SP-05, ML-05. |
| Dataset licence uncertainty | Medium / Medium | D-13. |
| Broken local environment slows Phase 1 | Certain / Low | Phase 1 first task (ARCHITECTURE.md §2). |

## 21. Assumptions

1. The project is a portfolio/learning-grade product built by a small team (1–2 engineers) without real customer data.
2. The owner will accept "assumed effect" labelling for the MVP.
3. A hosted LLM API with a free or low-cost tier is acceptable for development (D-09).
4. The retention playbook and policy will be authored by the project (SIMULATED knowledge), since no real policy exists.
5. "Churn" in the file means the customer had left by snapshot time; horizon unknown.
6. The IBM sample may legally remain in the repository (unverified; D-13).

## 22. Open questions

See [DECISIONS.md](DECISIONS.md). The ones that change what gets built: D-03 (data strategy), D-05 (roadmap order), D-07 (persistence), D-09 (LLM provider), D-10 (agent scope), D-11 (UI stack), D-14 (protected attributes).
