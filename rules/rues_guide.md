# DRA Rules File — Authoring Guide

A complete reference for writing a **rules file** for the generic Data Readiness
Assessment engine (`dra_engine.py`, run via `run_dra.py`). Use it whenever you
need to create rules for a new dataset.

The engine is dataset-agnostic. The rules file is the *only* thing you write per
dataset: it declares **which checks to run, on which columns, and what counts as
a problem**. The engine computes each metric in PySpark, scores it against your
thresholds, and aggregates everything into a GO / CONDITIONAL GO / NO-GO verdict.

---

## Table of contents
1. [File format and top-level structure](#1-file-format-and-top-level-structure)
2. [The `dataset` block](#2-the-dataset-block)
3. [Declaring checks (`type`, `params`, `thresholds`)](#3-declaring-checks)
4. [Thresholds, operators, severities](#4-thresholds-operators-severities)
5. [How a metric becomes a verdict](#5-how-a-metric-becomes-a-verdict)
6. [The final decision](#6-the-final-decision)
7. [Full check reference](#7-full-check-reference)
8. [Column typing and auto-behaviours](#8-column-typing-and-auto-behaviours)
9. [A recipe for a brand-new dataset](#9-a-recipe-for-a-brand-new-dataset)
10. [Cookbook: column role → checks](#10-cookbook-column-role--checks)
11. [Common pitfalls](#11-common-pitfalls)
12. [Running the assessment](#12-running-the-assessment)
13. [Annotated example](#13-annotated-example)

---

## 1. File format and top-level structure

Rules can be **YAML** (`.yaml`/`.yml`, needs `pyyaml`) or **JSON** (`.json`,
standard library — use this if `pyyaml` isn't installed). Both express the same
object with three top-level keys:

```yaml
dataset:          # metadata about the data as a whole
dataset_checks:   # checks that run on the whole dataset (a list)
columns:          # per-column checks (a map of column name -> list of checks)
```

All three are optional, but a useful file has at least `dataset` + some checks.

---

## 2. The `dataset` block

```yaml
dataset:
  target: purchase_intent        # column being predicted (or null)
  problem_type: classification   # classification | regression
  time_column: signup_date       # a date/timestamp column, or null
  reference_data: null           # path to a drift-reference dataset, or null
```

| Field | Purpose | Needed by |
|-------|---------|-----------|
| `target` | the column being modelled | `class_balance`, `signal`, `leakage_auc`, `baseline`, and `sample_size` (for usable-row counting) |
| `problem_type` | `classification` or `regression` (default `classification`) | `signal`/`leakage_auc` only run for classification; `baseline` switches between ROC-AUC and RMSE |
| `time_column` | a time column used to split old vs new rows | `drift_psi` (when no `reference_data`) |
| `reference_data` | explicit baseline dataset for drift | `drift_psi`; also set via `--reference` on the CLI |

If `target` is `null`, the target-dependent checks degrade gracefully (they
record an informational "no target configured" row rather than failing).

---

## 3. Declaring checks

Every check is an object with up to three keys:

```yaml
- type: missingness                 # REQUIRED: a registered check name
  params: {min: 18, max: 100}       # OPTIONAL: extra inputs some checks need
  thresholds: [...]                 # OPTIONAL: rules that assign a severity
```

- **`dataset_checks`** is a *list* of such objects.
- **`columns`** is a *map* from column name to a *list* of such objects:

```yaml
columns:
  annual_income:
    - type: missingness
      thresholds: [{op: gt, value: 20, severity: failure}]
    - type: redundancy
      params: {with: [income_bracket]}
      thresholds: [{op: gte, value: 0.95, severity: failure}]
```

A check with **no `thresholds`** still runs and reports its metric as
`Information` (handy for profiling, e.g. `numeric_stats`).

---

## 4. Thresholds, operators, severities

A `thresholds` entry is a rule: *"if the metric `op` `value`, raise `severity`."*

```yaml
{op: gt, value: 20, severity: failure}        # if value > 20 -> Failure
{op: between, value: [0, 100], severity: warning}  # if OUTSIDE [0,100] -> Warning
```

### Two shapes of `thresholds`

- **Flat list** — for a check that emits a single metric. The list applies to
  that metric. (For a *multi-metric* check, a flat list is applied to **every**
  emitted metric.)
  ```yaml
  thresholds:
    - {op: gt, value: 20, severity: failure}
    - {op: gt, value: 10, severity: warning}
  ```
- **Dict keyed by metric name** — for multi-metric checks, to target a specific
  metric. **Keys must be the raw metric name** (see [§7](#7-full-check-reference)),
  not the `metric:column` label shown in the report.
  ```yaml
  thresholds:
    row_count:              [{op: lt, value: 1, severity: hard_stop}]
    empty_column_count:     [{op: gt, value: 0, severity: failure}]
  ```

### Operators
`gt` (>), `gte`/`ge` (>=), `lt` (<), `lte`/`le` (<=), `eq` (==), `ne`/`neq` (!=),
`between` (value is `[lo, hi]`; **breaches when the metric is OUTSIDE the
inclusive band**).

### Severities (increasing)
`information` < `warning` < `failure` < `hard_stop`
(aliases accepted: `info`, `warn`, `fail`, `stop`/`hardstop`).

Optional `blocks: true` can be added to a rule. **Note for the DRA engine:** the
final decision is driven by *severity* (`hard_stop` ⇒ NO-GO). The `blocks` flag is
recorded in the output but does not itself change the DRA verdict — to force a
NO-GO, use `severity: hard_stop`.

---

## 5. How a metric becomes a verdict

For each metric a check emits, the engine:
1. Looks up the applicable threshold rules.
2. Keeps the rules that are **breached**.
3. Assigns the **most severe** breached rule's severity.
4. If no rule is breached → `PASS` / `Information`.
5. If the metric is **non-numeric or NaN** (e.g. a check couldn't compute, or a
   text recommendation) → informational passthrough. The engine never invents a
   default.

Status labels map from severity: `Information→PASS`, `Warning→WARN`,
`Failure→FAIL`, `Hard Stop→HARD_STOP`.

---

## 6. The final decision

`aggregate_decision` scans all results:

| Condition | Decision |
|-----------|----------|
| any `Hard Stop` present | **NO-GO** |
| else any `Failure` present | **CONDITIONAL GO** |
| else (only warnings / clean) | **GO** |

The CLI exits with code `2` on NO-GO (handy for pipeline gating).

---

## 7. Full check reference

`Level` = where it attaches (a `column:` entry or `dataset_checks:`).
`Metric(s)` = the exact metric name(s) emitted — **use these as keys** in
dict-form thresholds.

### Column checks

| `type` | Params | Metric(s) emitted | What it computes |
|--------|--------|-------------------|------------------|
| `missingness` | — | `missing_pct` | % null in the column |
| `distinct_count` | — | `distinct_count` | number of distinct values (use `lte 1` to flag constants) |
| `unique_ratio` | — | `unique_ratio` | distinct / rows (use `gte 0.99` for ID-like) |
| `top_value_pct` | — | `top_value_pct` | % held by the most frequent value (near-constant / dominant category) |
| `numeric_stats` | — | `mean`, `std`, `min`, `max` | descriptive stats (usually no thresholds; threshold `std` with `lte 0` to catch constants) |
| `outlier_pct` | — | `outlier_pct` | % outside the 1.5×IQR band |
| `range_check` | `min`, `max` | `out_of_range_pct` | % of non-null values outside `[min, max]` |
| `allowed_values` | `values: [...]` | `invalid_value_pct` | % of non-null values not in the allowed set |
| `class_balance` | — | `target_n_classes`, `minority_class_count`, `minority_class_pct` | class distribution (put on the **target**; classification) |
| `signal` | — | `signal_strength` | association with target: \|corr\| (numeric) or Cramér's V (categorical). **Classification only** |
| `redundancy` | `with: [cols]` (default: all other numeric) | `abs_correlation__<col>` (per pair), `max_abs_correlation` | pairwise \|correlation\| with other columns |
| `vif` | `with: [cols]` (default: all other numeric) | `vif` | variance inflation factor (multicollinearity); numeric only |
| `leakage_auc` | — | `single_feature_auc` | single-feature AUC vs target (direction-agnostic). **Classification + numeric only** |
| `drift_psi` | — | `psi` | population stability index vs reference. Needs `reference_data`/`--reference` **or** a `time_column`; numeric only |

### Dataset checks

| `type` | Params | Metric(s) emitted | What it computes |
|--------|--------|-------------------|------------------|
| `structure` | — | `row_count`, `column_count`, `duplicate_column_count`, `empty_column_count` | basic shape + structural defects |
| `sample_size` | — | `usable_rows`, `rows_per_feature`, `minority_class_count`* | volume vs feature count (*`minority_class_count` only if `class_balance` ran on the target) |
| `baseline` | — | `baseline_metric` | quick benchmark model: ROC-AUC (classification) or RMSE (regression); informational. Needs `target` |
| `evaluation_readiness` | — | `recommended_evaluation` (text) | suggests a validation strategy; informational (no thresholds) |

---

## 8. Column typing and auto-behaviours

At startup the engine classifies every column into **numeric / categorical /
datetime** from the Spark schema. A column is treated as **datetime** (and thus
excluded from numeric checks) if **any** of these hold:
- its Spark type is `Date`/`Timestamp`, **or**
- its name contains the substring `date` (case-insensitive), **or**
- it is the configured `time_column`.

Two automatic behaviours worth knowing:
- **Baseline auto-exclusion:** any column where `unique_ratio` or `leakage_auc`
  scores `Failure` or worse is automatically **excluded from the `baseline`
  model** (IDs and leaky columns don't pollute the benchmark). This is why
  column checks run before dataset checks.
- **`numeric_stats` / `evaluation_readiness`** emit informational values; they
  won't change the verdict unless you add numeric thresholds (only possible for
  `numeric_stats`).

---

## 9. A recipe for a brand-new dataset

1. **Inspect the schema** — list columns, types, and look at a sample. Note the
   target, any ID columns, date columns, and the role of each feature.
2. **Fill the `dataset` block** — set `target`, `problem_type`, and
   `time_column` if you have one.
3. **Add `dataset_checks`** — almost always `structure` + `sample_size`, plus
   `baseline` and `evaluation_readiness` when you have a target.
4. **Add `columns`** — for each column you care about, pick checks by its role
   (see the cookbook below) and set thresholds.
5. **Set thresholds in severity bands** — stack rules so a metric escalates,
   e.g. missingness `>10 → warning`, `>20 → failure`.
6. **Verify column names exist** (watch spaces/casing) and run.
7. **Tune after the first run** — adjust thresholds once you see real values.

---

## 10. Cookbook: column role → checks

| Column role | Recommended checks |
|-------------|--------------------|
| Primary key / ID | `unique_ratio` (`gte 0.99 → failure`) → auto-excluded from baseline |
| Target (classification) | `class_balance` (minority `lt 20 → warning`, `lt 5 → failure`) |
| Target (regression) | `missingness`, `numeric_stats`, `outlier_pct` |
| Numeric feature | `missingness`, `range_check`, `outlier_pct`; `redundancy`/`vif` if collinearity is a concern; `signal` (classification) |
| Categorical feature | `allowed_values`, `top_value_pct` (near-constant), `distinct_count` (constant), `signal` |
| Date/time column | set as `time_column`; enables `drift_psi` on numeric features |
| Suspected leak (post-outcome value) | `leakage_auc` (`gte 0.98 → hard_stop`) |
| Derived / duplicate column | `redundancy` with the source column(s) |

---

## 11. Common pitfalls

- **Dict-threshold keys must be the raw metric name** (`row_count`,
  `missing_pct`, `single_feature_auc`, …), *not* the `metric:column` label from
  the report.
- **`between` breaches when OUTSIDE** the band. For "value should be in [0,100]"
  use `{op: between, value: [0, 100], severity: ...}`.
- **Column names with spaces** must be quoted in YAML: `"Fats and oils":`.
  The engine handles the Spark side.
- **The `date` substring rule** reclassifies columns as datetime. A numeric
  column whose name contains `date` (e.g. `candidate_score`) will be dropped
  from numeric checks. Rename or be aware.
- **`regression` disables `signal` and `leakage_auc`** (they return NaN →
  informational). Use `redundancy`/`vif`/`outlier_pct` for regression features.
- **`drift_psi` needs** either `reference_data`/`--reference` or a `time_column`
  with at least ~50 rows on each side of the split; column must be numeric.
- **Unnamed index columns** (a leading comma in the CSV header) load in Spark as
  `_c0`. Drop them in preprocessing; don't write checks against them.
- **`pyyaml` not installed** → the runner can't read `.yaml`. Use a `.json`
  rules file (same structure) or `pip install pyyaml`.
- **A referenced column that doesn't exist** produces an `ERROR` result row
  ("column not in dataset"); it won't crash the run, but the check won't run.

---

## 12. Running the assessment

```bash
# from POC/DRA/
python run_dra.py --data <data> --rules <rules> [--output <dir>] \
                  [--format csv|parquet|delta|table] [--reference <path>]

# examples
python run_dra.py --data customers.csv --rules rules.example.json
python run_dra.py --data data_to_model.csv --rules rules.data_to_model.yaml
python run_dra.py --data cat.schema.table --rules rules.yaml --format table   # Databricks UC
python run_dra.py --data cur.csv --rules rules.yaml --reference prev.csv       # explicit drift ref
```

Outputs are written to the `--output` directory (default `dra_output/`):
`report.html` (self-contained, with plots), `results.csv`, `summary.json`.
Exit code is `2` on NO-GO.

---

## 13. Annotated example

```yaml
dataset:
  target: purchase_intent
  problem_type: classification
  time_column: signup_date

dataset_checks:
  - type: structure                      # multi-metric -> dict-keyed thresholds
    thresholds:
      row_count:              [{op: lt, value: 1, severity: hard_stop}]
      duplicate_column_count: [{op: gt, value: 0, severity: failure}]
      empty_column_count:     [{op: gt, value: 0, severity: failure}]
  - type: sample_size
    thresholds:
      usable_rows:      [{op: lt, value: 100, severity: hard_stop}]
      rows_per_feature: [{op: lt, value: 10,  severity: warning}]
  - type: evaluation_readiness           # informational
  - type: baseline                       # ROC-AUC benchmark (informational)

columns:
  purchase_intent:                       # the target
    - type: class_balance
      thresholds:
        minority_class_pct:
          - {op: lt, value: 5,  severity: failure}
          - {op: lt, value: 20, severity: warning}

  customer_id:
    - type: unique_ratio                 # ID-like -> auto-excluded from baseline
      thresholds: [{op: gte, value: 0.99, severity: failure}]

  annual_income:
    - type: missingness                  # single-metric -> flat list
      thresholds:
        - {op: gt, value: 20, severity: failure}
        - {op: gt, value: 10, severity: warning}
    - type: redundancy
      params: {with: [income_bracket]}
      thresholds: [{op: gte, value: 0.95, severity: failure},
                   {op: gte, value: 0.80, severity: warning}]

  purchase_amount:
    - type: leakage_auc                  # forces NO-GO if it leaks the target
      thresholds: [{op: gte, value: 0.98, severity: hard_stop}]
```

See `rules.example.yaml` (customers demo) and `rules.data_to_model.yaml`
(regression, soap-formulation dataset) in this folder for full worked files.

> This guide documents the **DRA / ML-readiness** engine. The sibling
> `DE-DRA/` engine reuses the same rules structure and operators but offers a
> data-engineering check vocabulary (`freshness`, `primary_key_uniqueness`,
> `schema_drift`, `cross_column`, …) for daily batch quality. The authoring
> approach is identical; only the check names and dimensions differ.
