# RetainAI 2.0 — Data Strategy

Status: **Phase 0 proposal — pending owner approval** (see [DECISIONS.md](DECISIONS.md), D-02 to D-04)

This document records what the existing data can and cannot support, what was measured to establish that, and the recommended data strategy.

## 1. Provenance labels

Every dataset, table, column group, chart and UI element in RetainAI must carry exactly one of these labels. They are never merged.

| Label | Meaning |
|---|---|
| **REAL** | Observed from a real business process, with provenance we can state. |
| **PUBLIC** | Published by a third party for public use. Says nothing by itself about whether it is real or synthetic. |
| **SYNTHETIC** | Generated data intended to resemble real data (e.g. a competition dataset that is documented as generated). |
| **SIMULATED** | Produced by RetainAI's own documented generator, with assumptions we wrote. Used only to exercise and test the pipeline. |

Rule: nothing labelled SYNTHETIC or SIMULATED may be reported as evidence of real-world effect. Any ROI, uplift or retention figure computed on it is "illustrative under stated assumptions".

## 2. Existing dataset: what it is

File: `data/raw/Telco-Customer-Churn.csv` (977,501 bytes, 7,043 rows, 21 columns, no BOM, CRLF line endings).

**Provenance: PUBLIC, and documented by its publisher as a fictional company.** IBM's Cognos Analytics sample documentation describes it as data about *a fictional telco company* providing home phone and internet services to 7,043 customers in California in one quarter ([IBM docs](https://www.ibm.com/docs/SSEP7J_11.1.0/com.ibm.swg.ba.cognos.ig_smples.doc/c_telco_dm_sam.html)). The row count matches the committed file. I did not verify that the committed CSV is byte-identical to IBM's original; it is a copy of the widely circulated Kaggle/IBM variant. It must therefore **not** be called REAL data anywhere in the product or in notes such as "real dataset". It is realistic-looking sample data.

The licence/terms under which the CSV may be redistributed in this repository were **not verified** in this session (see DECISIONS.md D-13).

## 3. Measured facts (audit performed with the Python standard library)

All numbers below were computed directly from the CSV in this session. No dependencies were installed.

| Fact | Value |
|---|---|
| Rows / columns | 7,043 / 21 |
| Unique `customerID` | 7,043 (no duplicate IDs) |
| Churn = Yes | 1,869 (26.54%) |
| Blank `TotalCharges` (stored as `" "`, so the column parses as text) | 11 rows; all have `tenure = 0` and `Churn = No` |
| Rows identical to another row on all 20 non-ID columns | 22 extra rows in 20 groups (likely coincidence among low-cardinality fields; unverified) |
| `MultipleLines = "No phone service"` exactly when `PhoneService = No` | 682 of 682 (structural redundancy) |
| Add-on services = `"No internet service"` exactly when `InternetService = No` | 0 exceptions either way (structural redundancy) |
| Whitespace/padding issues in string fields other than the blank `TotalCharges` | 0 |
| Monthly revenue held by churned customers, as a share of all monthly revenue in the file | 30.5% (139,131 of 456,117; fictional currency units, dataset-internal) |
| Two-year contract customers with `tenure < 24` months | 142 |
| `TotalCharges` within 2% of `tenure × MonthlyCharges` | 50% of non-blank rows (5th–95th percentile of relative deviation ≈ −7.6% to +7.5%; range −31% to +57%) |

Churn rate by segment (rows, churn %):

| Segment | Result |
|---|---|
| Contract | Month-to-month 3,875 (42.7%); One year 1,473 (11.3%); Two year 1,695 (2.8%) |
| Internet service | DSL 2,421 (19.0%); Fiber optic 3,096 (41.9%); None 1,526 (7.4%) |
| Payment method | Electronic check 2,365 (45.3%); Mailed check 1,612 (19.1%); Bank transfer 1,544 (16.7%); Credit card 1,522 (15.2%) |
| Tenure (months) | 0–6: 1,481 (52.9%); 7–12: 705 (35.9%); 13–24: 1,024 (28.7%); 25–48: 1,594 (20.4%); 49–72: 2,239 (9.5%) |
| Senior citizen | No 5,901 (23.6%); Yes 1,142 (41.7%) |
| Gender | Female 3,488 (26.9%); Male 3,555 (26.2%) |
| Tech support | Yes 2,044 (15.2%); No 3,473 (41.6%); no internet 1,526 (7.4%) |

Interpretation notes:

- Churned customers have a median tenure of 10 months and a median monthly charge of 79.65, versus 29 months and 70.35 for the whole file (notebook 01 output, `describe`). Churn is concentrated among new, higher-paying, month-to-month, fiber, electronic-check customers.
- Gender shows almost no difference in churn (26.9% vs 26.2%), so it carries little predictive value and is a fairness hazard. Senior-citizen status carries signal but is a protected-class proxy (age). See DECISIONS.md D-14.
- The 142 two-year contracts with tenure under 24 months show that `Contract` is not necessarily the contract the customer held for their whole tenure; its time reference is undocumented. Treat it as a possible source of leakage until explained.
- `TotalCharges` is only loosely reproducible from `tenure × MonthlyCharges` (prices change, or the generator is noisy). It is partly redundant with `tenure`, so it needs a collinearity check in modelling.

## 4. Capability matrix — what this dataset can honestly support

| Capability | Supported? | Basis and constraints |
|---|---|---|
| Churn prediction | **Yes** | Binary `Churn` label, 26.5% prevalence, 7,043 rows. Small data: expect wide confidence intervals; use repeated stratified CV with duplicate-aware grouping. |
| Risk scoring | **Yes, with caveats** | Ranking is fine. Calibration must be measured and, if needed, corrected. The label horizon (churned "when?") is **not defined in the file**, so a score cannot be stated as "probability of churning within N days". |
| Segmentation | **Yes** | Rich categorical and tenure/charge fields. |
| Explainability | **Yes, associational only** | Feature attributions describe what the model uses, not what causes churn. Product wording must say so. |
| Customer prioritisation | **Partly** | Risk × revenue (`MonthlyCharges`) is available. This is *revenue at risk*, not "worth intervening on", which needs an effect estimate. |
| Customer lifetime value | **Partly (proxy only)** | `tenure` + `Churn` form right-censored survival data (duration, event), so survival-based expected remaining revenue is possible under stated assumptions. There is no margin, cost-to-serve or discount rate. Any CLV is a labelled revenue proxy. |
| Intervention recommendation | **Not from data** | No intervention was ever recorded. Recommendations can only come from policy rules plus *assumed* effects. |
| Uplift / treatment-effect modelling | **No** | No treatment indicator, no control group, no post-treatment outcome. |
| Causal inference about interventions | **No** | Observational cross-section; no assignment mechanism; observable "treatments" such as tech support or contract type are self-selected. Differences in churn between them are associations. |
| Intervention outcome evaluation | **No** | No outcomes of interventions exist. |
| Temporal analysis, time-based validation | **No** | No dates or event history; one snapshot. Only `tenure` is time-like. Validation can only be random/stratified, which is optimistic compared with a forward-in-time split. |

## 5. Missing information (all confirmed absent from the 21 columns)

| Missing item | Status | Evidence |
|---|---|---|
| Treatment / intervention history | NOT FOUND | Column list contains no offer, campaign, contact or treatment field (`data/raw/Telco-Customer-Churn.csv` header). |
| Time dimension (dates, snapshots) | NOT FOUND | No date column; only `tenure` (0–72 months). |
| Control groups / assignment mechanism | NOT FOUND | No such field; no experiment described. |
| Offer history and offer cost | NOT FOUND | No such field. |
| Post-intervention outcomes | NOT FOUND | Only a single `Churn` label of undocumented horizon. |
| Customer event history (tickets, usage, complaints, calls) | NOT FOUND | No such fields. |
| Margin / cost-to-serve | NOT FOUND | Only `MonthlyCharges` and `TotalCharges`. |
| Churn reason / satisfaction | NOT FOUND in this file | IBM publishes a richer variant of the sample with satisfaction, a churn score and a lifetime-value index ([IBM community](https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113)). I did not examine it. If considered, scores such as a churn score are likely to be leakage and must be vetted first. |

## 6. Options evaluated

**A. Existing Telco only.**
Supports Phases 1–6 honestly (risk, explanation, revenue-at-risk, policy-gated recommendations with *assumed* effects). Cannot support uplift or any effect claim. Lowest cost. Weakness: the product's strongest differentiator (effect-aware prioritisation) is absent or only assumed.

**B. Telco + additional public datasets.**
No public dataset can be joined to Telco customers, because there is no shared key or population. Public randomized-treatment datasets (section 7) are therefore only usable as a **separate, isolated validation of the uplift method** (do our estimators and our Qini/AUUC evaluation work on real randomized data?). They cannot supply effect sizes for Telco customers.

**C. Replace the dataset.**
I found no public dataset combining real telecom churn with randomized retention interventions (search was not exhaustive). Replacing Telco would discard the work already done (two notebooks) for no clear gain. Not recommended.

**D. Telco + a clearly-labelled simulated operational layer.**
Add SIMULATED tables for offer catalog, offer costs, campaign assignment and treated outcomes, produced by a versioned, documented generator. Because we write the data-generating process, the ground-truth effect is known, so we can test whether the pipeline (and uplift estimators) recover it, and can exercise the full loop (recommend → approve → intervene → measure) end to end. Weakness: circularity. A model "finding" an effect we coded proves only that the code works. Results must never be presented as findings about customers.

**E. Another justified approach — rejected: LLM-generated data.**
Generating customer or outcome data with an LLM is not reproducible, not auditable and not statistically controlled. Not recommended.

### Recommendation: D, plus an isolated slice of B

1. **Telco (PUBLIC, fictional) remains the customer base** for risk, explanation, segmentation, revenue-at-risk.
2. **A SIMULATED operational layer** is added in the uplift phase (ROADMAP Phase 7), not before. It is generated from a versioned config file, seeded, and stored in tables whose names and columns carry the `SIMULATED` label.
3. **One public randomized dataset** is used to validate the uplift/evaluation methodology on real randomized data, in a module that does not touch Telco. Hillstrom first (small, 3-arm, has spend); others only if needed.
4. Until step 2 exists (MVP), the decision engine uses **ASSUMED** effects from a policy config file and labels them as assumptions in every output.

Candidate simulation design (to be finalised in Phase 7, not decided now): treat the real `Churn` label as the *no-intervention* outcome Y(0). Define a documented effect model that produces Y(1) for treated customers (a save probability that varies by segment, including customers with no effect, and a small share with negative effect). Randomly assign treatment in the simulated campaign so that effects are identifiable. Anti-circularity rules: the effect function is defined in a config *before* modelling, is not tuned to flatter any model, and the report always states "recovers the assumed effect", never "customers respond this way".

## 7. Public treatment/uplift datasets (candidates, not downloaded)

Nothing below has been downloaded or added to the repository. Licence statements are from the publishers' pages as fetched in this session; anything marked "unverified" must be checked before use.

| | Hillstrom E-Mail | Criteo Uplift v2.1 | Lenta Uplift | MegaFon Uplift |
|---|---|---|---|---|
| Label | PUBLIC; presented by its author as the result of a randomized e-mail test; provenance not independently verified | PUBLIC; built from randomized incrementality tests on ad exposure | PUBLIC; competition data (BigTarget Hackathon 2020, Lenta/Microsoft) | PUBLIC and **SYNTHETIC** (documented as generated data) |
| Source | [MineThatData blog, 2008](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html) | [Criteo AI Lab](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) | via [scikit-uplift docs](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_lenta.html) | via [scikit-uplift docs](https://www.uplift-modeling.com/en/latest/api/datasets/fetch_megafon.html); originally ods.ai competition |
| Contents | 64,000 customers; random 1/3 Men's e-mail, 1/3 Women's e-mail, 1/3 no e-mail; outcomes: visit, conversion, spend over 2 weeks; recency, history, channel, region, new-customer flags | ~13.98M rows; 12 anonymised features, treatment (84.6% treated), exposure, visit, conversion | 687,029 rows; 75% treated / 25% control; target `response_att` (store visit); demographics, purchase history | 600,000 rows; 50 anonymised features; 50% treated; binary conversion (purchase), ~20% rate |
| Licence / access | No explicit licence or restriction found on the page. Treat as "unspecified; attribute the source" until clarified | **CC BY-NC-SA 4.0**: non-commercial only, attribution and share-alike; 297 MB compressed; citation requested | No licence stated in the documentation; unverified | No licence stated in the documentation; competition terms unverified |
| Enables | Uplift estimators + Qini/AUUC on real randomized data; incremental revenue (spend) and cost-aware targeting; multi-arm treatments | Scale test; very low base rates | Retail-grocery campaign uplift; larger test | Telecom-flavoured pipeline test (but anonymised and synthetic) |
| Limitations | Retail e-mail, not telecom; outcome is purchase, not churn; one campaign, 2 weeks; 2008 | Advertising domain; anonymised features; NC licence may block any commercial use; large download | Grocery; outcome is a visit, not churn | Synthetic; features uninterpretable; outcome is a purchase, not churn; licence unclear |

**Recommendation:** use **Hillstrom** as the first validation set (small, interpretable, has revenue outcome, no known restriction beyond the unspecified licence). Criteo only if scale testing is needed and non-commercial use is acceptable. Treat Lenta and MegaFon as optional because their licences are unstated. The X5 RetailHero uplift data also exists in scikit-uplift; it was not evaluated.

None of these datasets measure *retention interventions on churn* in telecom. They validate method, not Telco effects.

## 8. Data-handling rules adopted for the project

1. Raw files in `data/raw/` are immutable. Cleaning happens in code and writes to `data/processed/`.
2. Every derived table records its provenance label and source file hash.
3. `TotalCharges` is parsed explicitly (blank → null, then handled by a documented rule). The 11 blank rows are all `tenure = 0`, no churn; the rule (drop vs impute 0) is a documented Phase 1 decision.
4. Train/validation splitting must be duplicate-aware (the 22 duplicated-content rows must not straddle splits).
5. Feature timing must be reviewed for leakage (`Contract`, `TotalCharges`) before modelling.
6. No customer-level data is sent to a hosted LLM without the pseudonymisation and policy rules in ARCHITECTURE.md.
7. Reported metrics must state dataset, split method, label, and provenance label.
