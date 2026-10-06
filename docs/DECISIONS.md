# RetainAI 2.0 — Decision Log

Status values: **Proposed** (I recommend it; low controversy, will proceed unless you object) · **Needs approval** (you must decide before the dependent phase) · **Owner action** (something only you can do).
Nothing here has been implemented. Each entry names the phase it blocks.

## Summary of decisions needing your approval

| ID | Decision | Recommendation | Blocks |
|---|---|---|---|
| D-03 | Data strategy | Telco + SIMULATED layer + isolated public randomized dataset | Phase 7 |
| D-04 | Which public uplift dataset(s) | Hillstrom first | Phase 7 |
| D-05 | Roadmap re-sequencing | Accept proposed order | Phase 1 |
| D-07 | Persistence | Files → SQLite; Postgres only on trigger | Phase 6 |
| D-08 | Vector store | No Qdrant in MVP | Phase 4 |
| D-09 | LLM provider and data policy | Thin interface; choose provider before Phase 5 | Phase 5 |
| D-10 | Agent scope | Fixed workflow in MVP | Phase 5 |
| D-11 | UI stack | Streamlit for MVP unless a product-grade front end is a goal | Phase 6 |
| D-13 | Licence and dataset redistribution | Owner action | Phase 1 |
| D-14 | Protected attributes in the model | Exclude gender; decide senior-citizen after audit | Phase 2 |
| D-15 | Python version and dependency tooling | pip + pinned `requirements.txt` | Phase 1 |
| D-16 | Uncommitted notebook 02 | Owner action | Phase 1 |
| D-17 | MVP definition | Evidence tier 1 (assumed effects, no causal claims) | Phase 1 |

---

## D-01 — Product framing: retention intervention intelligence  · Proposed

**Decision:** RetainAI 2.0 is a decision-support loop (risk → reasons → value → policy-allowed options → recommendation → human decision → outcome), not a churn dashboard.
**Why:** consolidated direction from the four analyses; confirmed feasible in part by the data audit (risk, explanation and value are supported; effect is not).
**Consequence:** the README's current "churn prediction platform" wording is outdated (rewritten in Phase 10, or earlier if you prefer).

## D-02 — The Telco dataset is PUBLIC sample data for a fictional company  · Proposed (factual)

**Decision:** label it PUBLIC; never call it REAL. All documents and UI carry this label.
**Why:** IBM's documentation describes a *fictional* telco company ([IBM docs](https://www.ibm.com/docs/SSEP7J_11.1.0/com.ibm.swg.ba.cognos.ig_smples.doc/c_telco_dm_sam.html)). The project brief refers to a "real dataset"; that wording should not be carried into product claims.
**Caveat:** I did not confirm the committed file is byte-identical to IBM's original.

## D-03 — Data strategy  · **Needs approval**

**Options:** A Telco only · B Telco + public datasets · C replace · D Telco + simulated operational layer · E LLM-generated data.
**Recommendation:** D plus an isolated slice of B (see DATA_STRATEGY.md §6). Reject C and E.
**Why:** A cannot deliver any effect-aware capability; B's datasets cannot be joined to Telco customers, so they only validate method; D lets us exercise and test the full loop with known ground truth at the cost of circularity, which we control by labelling and by predeclaring effects.
**Trade-off you accept:** the headline "uplift" in V1 is simulated or method-validation only. It demonstrates capability, not customer behaviour.
**If you disagree:** choose A and drop Phases 7–8 from V1; the MVP is unchanged because it already works on assumed effects.

## D-04 — Public uplift dataset selection  · **Needs approval**

**Recommendation:** Hillstrom first (small, interpretable, revenue outcome). Criteo only if scale is needed and non-commercial use is acceptable (CC BY-NC-SA 4.0). Lenta and MegaFon optional because no licence is stated. See DATA_STRATEGY.md §7.
**Question for you:** is this project strictly non-commercial/portfolio, or could it become commercial? That decides whether Criteo is usable at all.

## D-05 — Roadmap re-sequencing  · **Needs approval**

**Decision:** move uplift from Phase 4 to Phase 7; put policy and thin RAG before the decision engine; merge API, UI and approvals into Phase 6; MVP (M2) is tier 1 with assumed effects. Mapping and reasons in ROADMAP.md §1.
**Why:** uplift is blocked on D-03 and cannot be validated on Telco; the decision engine can accept it later through the `effect_source` interface.
**Alternative:** keep the original order, accepting that Phase 4 stalls until D-03 is settled.

## D-06 — Architecture style: modular monolith  · Proposed

**Decision:** one Python package, no microservices, no queues.
**Why:** no independent scaling or deployment need exists (ARCHITECTURE.md §4).

## D-07 — Persistence  · **Needs approval**

**Recommendation:** files for Phases 1–5; SQLite through SQLAlchemy for the ledger in Phase 6; PostgreSQL only when concurrent multi-user writes or a hosted deployment are required.
**Why:** approvals and audit logs need transactional storage, but not a server.
**Alternative:** PostgreSQL from the start if you want a deployment-realistic stack for portfolio purposes; it adds a service to run from Phase 6.

## D-08 — Vector store  · **Needs approval**

**Recommendation:** no Qdrant in the MVP. In-process exact search over a small corpus, evaluated against BM25.
**Trigger to revisit:** corpus above ~10⁴ chunks, concurrent updates, or heavy metadata filtering.
**Alternative:** adopt Qdrant now for portfolio value; accept an extra service and no measurable gain at this scale.

## D-09 — LLM provider and data policy  · **Needs approval**

**Options:** Groq, Gemini, OpenRouter (or a local model).
**Recommendation:** one thin provider interface plus a stub provider for tests; choose the hosted provider by structured-output reliability, free-tier limits and data-retention terms checked at that time (these change; I did not research current terms in Phase 0).
**Policy to approve:** only pseudonymous IDs and structured fields leave the machine; acceptable because the data is a fictional public sample. This must be revisited before any real data is used.

## D-10 — Agent scope  · **Needs approval**

**Recommendation:** MVP uses a fixed, deterministic tool sequence; an LLM-selected tool loop is added only if it beats the fixed sequence on the evaluation set (AG-02).
**Why:** a single-customer recommendation has a fixed sequence of steps; LLM tool selection would add variance without a demonstrated need.
**Alternative:** build the LLM tool loop in MVP for portfolio value; accept added non-determinism and evaluation effort.

## D-11 — UI stack  · **Needs approval**

**Options:** Streamlit; React + Vite on FastAPI.
**Recommendation:** Streamlit for the MVP unless a product-grade front end is itself a goal; the workflow (ranked list, detail, approve) is simple, and UI effort is not where the project's risk lies.
**Alternative:** React + Vite if the README's "web-based interface" is meant to be a polished product.
**Note:** FastAPI is still introduced in Phase 6 either way, to keep the core separate from the UI.

## D-12 — Policy engine as the only eligibility authority  · Proposed

**Decision:** eligibility is decided by deterministic rules in versioned config; RAG explains and cites; the LLM never decides.
**Why:** "is this allowed?" must be unit-testable and reproducible.

## D-13 — Licence and dataset redistribution  · **Owner action**

**Facts:** the repository `LICENSE` is empty (0 bytes); the IBM sample CSV is committed (twice in history); I did not verify the terms under which IBM's sample data may be redistributed; repository visibility on GitHub was not checked.
**Request:** pick a licence for your own code, and confirm the dataset's redistribution terms (or move the CSV to a documented download step). I made no change.

## D-14 — Protected attributes  · **Needs approval**

**Facts:** gender shows 26.9% vs 26.2% churn (almost no signal); senior-citizen status shows 41.7% vs 23.6% (signal, and a proxy for age).
**Recommendation:** exclude `gender` from modelling; keep `SeniorCitizen` in the risk model only after the fairness audit (ML-05), never in eligibility or ranking logic (SP-05).
**Alternative:** exclude both from everything; accept possibly lower accuracy.

## D-15 — Python version and dependency tooling  · **Needs approval**

**Facts:** `.venv` is broken (points to a missing Anaconda install); `requirements.txt` is empty; system Python is 3.13.2 and has no data libraries; notebooks used a conda `base` kernel with unrecorded versions.
**Recommendation:** pip + a pinned `requirements.txt`, rebuilding the `.venv` on a single stated Python version.
**Alternative:** conda environment file, or uv/Poetry.

## D-16 — Uncommitted notebook 02  · **Owner action**

**Facts:** `notebooks/02_exploratory_data_analysis.ipynb` has uncommitted edits (24 cells, 532 added lines). Phase 0 left it untouched (SHA-256 verified before and after branching) and did not include it in the Phase 0 commit.
**Request:** commit it on your own terms (on `main` or a separate branch) before Phase 1. The Phase 0 branch was created from `main`, so it does not contain this work in its history.
**Note:** its recommendations are associational (ARCHITECTURE.md R10) and its cleaning step should become a tested rule in Phase 1.

## D-17 — MVP definition  · **Needs approval**

**Decision:** MVP = evidence tier 1: Telco-only, assumed effects with visible badges, full loop through human approval, no causal claims (PRD §16). V1 adds uplift on simulated/public data and outcome measurement.
**Why:** it is the largest product the data supports honestly.
