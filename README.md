# RetainAI

### Predict. Explain. Retain.

## What RetainAI is

RetainAI is being built as a **retention decision-support system**: identify customers at risk, explain why, estimate the revenue exposed, check which retention actions are allowed, recommend one, record the human decision, and eventually learn from outcomes. The full direction is in [docs/PRD.md](docs/PRD.md), [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/ROADMAP.md](docs/ROADMAP.md).

## Current phase

**Phase 1 — Customer & Retention Intelligence Foundation.** This phase delivers a clean, reproducible, tested base: a validated dataset contract, deterministic feature engineering, baseline churn models, risk scores, model-grounded reasons, revenue-exposure proxies and a subgroup audit. Details: [docs/PHASE_1_PIPELINE.md](docs/PHASE_1_PIPELINE.md).

## What it can do today

- Validate the Telco CSV against an explicit contract and report known data issues instead of hiding them.
- Train and compare a prior baseline, logistic regression, random forest and gradient boosting, with calibration, threshold and capacity analysis.
- Score customers: `risk_score`, `risk_band` (LOW / MEDIUM / HIGH), model version, timestamp, and up to three model-grounded reasons.
- Show per-customer revenue-at-risk proxies and per-band totals.
- Audit model behaviour across gender, senior citizen, partner and dependents.

## What it cannot do

- It cannot say what would make a customer stay. There are **no causal claims and no treatment-effect estimates**; reasons are patterns the model uses, not causes.
- It does not recommend interventions, check offer eligibility, use an LLM or RAG, run agents, track outcomes, or have a UI or API. Those are later phases and each must earn its place with evidence.
- Its value figures are proxies, not true customer lifetime value.

## Dataset limitations

The only data is the IBM Telco Customer Churn sample (`data/raw/Telco-Customer-Churn.csv`, 7,043 rows). IBM documents it as data for a **fictional** company, so results say nothing about a real customer base. It has no treatments, offers, intervention costs, dates, controls or post-intervention outcomes, only a single static snapshot with an **undefined churn horizon**. Scores are therefore not "probability of leaving within N days", and no temporal validation is possible. See [docs/DATA_STRATEGY.md](docs/DATA_STRATEGY.md). The redistribution terms of the sample have not been verified, and the repository `LICENSE` file is empty (see `docs/DECISIONS.md` D-13).

## Set up the environment

Tested on Python 3.13 (Windows). Dependencies are pinned in `requirements.txt` (runtime) and `requirements-dev.txt` (adds pytest). The project is CPU-only and light enough for a 16 GB laptop.

```bash
python -m venv .venv
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` on macOS/Linux), then:

```bash
pip install -r requirements-dev.txt
pip install -e . --no-deps
```

The notebooks in `notebooks/` additionally need matplotlib, seaborn and Jupyter, which are not part of the pinned set.

## Validate the dataset

```bash
python -m retainai validate
```

Prints the result of every contract check, lists the known issues (KI-1 to KI-4) and writes `reports/data_validation.json`. The command exits non-zero if the contract is violated. To also write the cleaned and model-ready tables to `data/processed/`:

```bash
python -m retainai prepare
```

## Train the baseline

```bash
python -m retainai train
```

Takes about two to three minutes. Writes `reports/baseline_metrics.json` and `reports/fairness_audit.json`, and saves the shipped model and its metadata to `models/churn_baseline/` (git-ignored). Logistic regression is shipped because the tree models are within noise of it and its explanations are exact.

## Generate risk scores

```bash
python -m retainai score
```

Scores `data/raw/Telco-Customer-Churn.csv` by default and writes `data/processed/risk_scores.csv`. Because that is the model's own training file, the output is flagged `in_sample` and is optimistic. Use `--input` with any raw-format CSV (the `Churn` column is optional) and `--output` with a `.csv` or `.json` path. Revenue-at-risk proxies:

```bash
python -m retainai value
```

## Run the tests

```bash
python -m pytest
```

Add `--runslow` to retrain on the real data and check that the committed reports reproduce.

## Repository layout

```
data/raw/        immutable source data
data/processed/  generated tables (git-ignored)
src/retainai/    data, features, models, scoring, explain, value, fairness, cli
tests/           unit, end-to-end and reproducibility tests
reports/         committed, reproducible metrics and audit reports
models/          trained artefacts (git-ignored)
docs/            product, data, architecture, roadmap, decisions, Phase 1 pipeline
notebooks/       exploratory notebooks (not part of the pipeline)
```
