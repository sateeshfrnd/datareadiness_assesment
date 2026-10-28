# DR-01  Data Intake & Structure (structure)
---
**Asks:** is the data readable, structurally valid, and is the modelling problem defined?

**Computes:** schema, row/column counts, duplicate columns, fully-empty rows/columns, format readability. 

**On this data:** reads cleanly; 2,000 rows × 12 columns; no duplicate column names; no fully-empty rows/columns; problem_type=classification and target=purchase_intent supplied.

**Outcome:** PASS / Information. (This is the gate that would Hard Stop immediately if the file were empty or corrupt — here it passes.)

# DR-02  Target (Y) Viability (quality)
---
**Asks:** is the target meaningful and compatible with the objective? 

**Computes:** target type, null %, unique count, class balance, variance. 

**On this data:** purchase_intent is binary, 0 nulls, 2 classes, minority class = 15%. Compatible with classification. But 15% is below the "balanced" band, so the imbalance is flagged (not fatal — it's above any minimum-rows hard stop). 

**Outcome:** WARN / Warning — "class imbalance 15%; use stratification and PR-AUC downstream."

# DR-03  Predictor (X) Quality (quality)
---
**Asks:** which predictors are structurally unusable or weak? 

**Computes:** per-column type, cardinality, uniqueness, constant/near-constant, ID-like indicators. 

**On this data:**
account_status is constant ("active") → zero variance → FAIL (drop it).
customer_id is unique for every row → ID-like → FAIL (exclude from modelling).
country is 94% a single value → near-constant → Warning.
Remaining predictors look structurally usable. 

**Outcome:** FAIL / Failure on account_status and customer_id; Warning on country.

# DR-04 Missingness & Completeness (quality)
---
**Asks:** how much is missing, and in what pattern? 

**Computes:** null count/% per column, row-completeness, pattern summary.

** On this data:** annual_income 22% missing (exceeds the configured 20% failure threshold); age 8% (exceeds 10% warning? no — under 10%, so borderline Information/Warning depending on config); everything else 0%. 

**Outcome:** FAIL / Failure on annual_income; Information/Warning on age. Recommendation: review imputation vs drop before modelling.

```
This is a good example of the engine's split-of-concerns: DR-04 only computes _missing_percentage:annual_income = 22.0._ The ThresholdEngine compares it to the versioned rule (> 20 → Failure) and assigns the Failure. The check itself never hard-codes "20".
```

# DR-05 — Sample Size & Coverage (quality)
---
**Asks:** is there enough usable data and representation? 

**Computes:** usable rows, subgroup/class coverage, rows-per-feature. 

**On this data:** 2,000 usable rows; minority class has 300 rows (above a typical minimum like 100); ~9 usable predictors → ~220 rows/feature (healthy). But some country values (e.g. "SG") have lessthen 30 rows → thin subgroup coverage. 

**Outcome:** PASS / Information overall, with a Warning on sparse subgroup coverage. Hard stop would only trigger if usable rows fell below the configured minimum.

# DR-06 — Signal Screening (signal)
---
**Asks:** which predictors actually relate to the target? 

**Computes:** association metric chosen by variable type (e.g. mutual information / point-biserial for numeric-vs-binary, Cramér's V / chi-square for categorical). 

**On this data:** sessions_last_30d shows strong association; email_opt_in mild; annual_income/age moderate; random_tag ≈ 0 → weak-signal Warning (candidate to drop). purchase_amount shows a near-perfect association — suspiciously strong, which DR-08 picks up as leakage rather than genuine signal. 

**Outcome:** Information + ranked signal; Warning on random_tag.

# DR-07 — Redundancy & Multicollinearity (risk)
---
**Asks:** are predictors duplicating each other? 

**Computes:** pairwise correlation, VIF (or equivalent), derived-feature indicators. 

**On this data:** annual_income ↔ income_bracket correlate ≈ 0.95 with high VIF (because income_bracket is literally derived from income). 

**Outcome:** WARN/FAIL / Warning→Failure — "redundant pair; keep one (annual_income), drop income_bracket."


# DR-08 — Leakage & Structural Risk (risk)
---
**Asks:** is any predictor leaking the answer or structurally dangerous? 

**Computes:** target-derived names/lineage, temporal ordering vs signup_date, extreme predictive association, identifier detection.

**On this data:** purchase_amount is exactly 0 for every non-buyer and >0 for every buyer — a near-perfect, post-outcome quantity. That's textbook target leakage. customer_id reconfirmed as an identifier. 

**Outcome:** Hard Stop on purchase_amount — this is a blocks_progression=true severity. It stops the pipeline with a clear reason, because a model trained on it would look perfect and fail in production.

# DR-09 — Drift & Stability (risk)
---
**Asks:** has the data shifted versus a reference dataset version? 

**Computes:** PSI/CSI, distribution shift, schema drift, target/feature drift — requires a reference_dataset_version_id. 

**On this data:** if you pass last quarter's version as the reference, say sessions_last_30d has PSI ≈ 0.28 (moderate drift) and schema is unchanged → Warning. If no reference is supplied, the check is SKIPPED (not failed) — the engine won't invent a reference. The exact reference-selection rule is still an open question (OQ-08). 

**Outcome:** Warning (with reference) or SKIPPED (without).

# DR-10 — Evaluation Readiness (evaluation)
---
**Asks:** what validation strategy is feasible? 

**Computes:** synthesises earlier findings — sample size, class balance, temporal/group structure. This produces a strategy recommendation, not a model score. 

**On this data:** 2,000 rows + 15% minority → recommend stratified k-fold; signup_date implies a temporal element → consider a time-based split to avoid look-ahead; minority of 300 supports cross-validation. 

**Outcome:** Information — "stratified 5-fold, or time-ordered split given signup_date; report PR-AUC alongside AUC."

# DR-11 — Baseline Modelling (baseline)
---
**Asks:** is there useful predictive signal in practice? (A benchmark, not a production model.) 

**Computes:** a simple, reproducible baseline (e.g. logistic regression) with a fixed seed and split from config. 

**On this data:** this runs only after prerequisites pass — and here DR-08's hard stop blocks it until the leaky purchase_amount is removed. The teaching point:

- Leaky column left in → AUC ≈ 0.99 (a false triumph).
- Leaky column removed → AUC ≈ 0.74, PR-AUC ≈ 0.41 → a believable, useful baseline.

**Outcome: **benchmark metrics, clearly labelled non-production. Blocked until the hard stop is resolved.

# DR-12 — Final Recommendation (recommendation)
---
**Asks:** aggregate all evidence into a suitability outcome. 

**Computes:** policy-driven aggregation of required-check statuses + any blocking flags. 

**On this data (first pass):** because DR-08 raised a Hard Stop, the blocking policy dominates → NO-GO, with the reason "target leakage in purchase_amount." No readiness number is invented — the scoring formula/weights are a confirmed open question (OQ-02). After remediation (drop purchase_amount, account_status, customer_id, income_bracket; handle annual_income missingness): re-run → likely Conditional Go, carrying the residual warnings (class imbalance, age missingness, drift). Outcome: NO-GO → (after fixes) Conditional Go.



