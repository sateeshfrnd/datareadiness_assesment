# Metric ID
LEAK-03
# Step code
LEAKAGE
# Metric
Repeated-measures groups
# What it measures
Detects whether several rows belong to the same unit (subject, patient, batch, experiment, device); such groups break independence and inflate CV scores under random splits.
# Implementation approach (step by step)
1. Use group_column if declared; otherwise auto-candidates: IDENTIFIER / CATEGORICAL columns with n_distinct between 2% and 50% of rows (values repeat).
2. Group statistics (groupBy(group).count()): n_groups, rows per group (min / median / max), % rows in groups with > 1 row.
3. Intra-class correlation of the target by group, one-way random effects ICC(1) = (MSB − MSW) / (MSB + (k0 − 1)·MSW), k0 = (N − Σn_g² / N) / (G − 1); binary target treated as 0/1 (or statsmodels MixedLM).
4. Optional: ICC of the top-10 signal features (features that mostly identify the group).
5. Group-predictiveness: CV score of a model using only group identity (cross-fitted target encoding) vs naive – high → random split would leak group information.
6. Pass group_required to EVAL-01 (GroupKFold / StratifiedGroupKFold).
# Formula / key logic
ICC(1) = (MSB − MSW) / (MSB + (k0 − 1)·MSW) ; design effect = 1 + (m̄ − 1)·ICC
# Input data required
Dataset (full for counts); sample for ICC
# Context / configuration input
group_column (optional), entity definitions
# Output generated
group_column_used, detected_candidates[], n_groups, rows_per_group stats, pct_rows_in_repeated_groups, target_icc, design_effect, group_required
# Default thresholds (PASS / WARN / FAIL)
target_icc: WARN ≥ 0.10 (group-aware CV required). FAIL raised in EVAL-02 if n_groups is too small for k folds.
# Execution tier
Grouped
# ML-specific?
Partial
# Context info needed?
Partial
# ML model-related info needed
Target; group / entity definition
# Primary build approach
Custom logic
# Databricks inbuilt capability
groupBy().count(); applyInPandas for per-group statistics
# 3rd-party libraries / functions
numpy; statsmodels MixedLM (optional); sklearn.preprocessing.TargetEncoder
# Custom logic to write
Candidate detection; ICC; group-predictiveness check
# Depends on
INTAKE-02, TARGET-01
# Developer notes / edge cases
Auto-detected candidates must be confirmed by the user – a repeating column such as 'city' is not a repeated-measures unit. ICC < 0.05 negligible; 0.05–0.10 minor; ≥ 0.10 material.
