# Metric ID
LEAK-01

# Step code
LEAKAGE

# Metric
Target leakage score

# What it measures
Composite 0–1 score per feature estimating the risk that it carries information unavailable at prediction time, or derived from the target.

# Implementation approach (step by step)
Evidence components per feature, each scaled 0–1:
1. e1 univariate predictiveness: from TARGET-04 → (AUC_adj − 0.5) / 0.5, or R².
2. e2 importance dominance: share of total importance in a quick shallow tree / HistGradientBoosting fit (5-fold stratified, on the sample); share > 0.5 is suspicious.
3. e3 drop-column impact (top-10 suspicious features only): CV score with all features minus CV score without the feature, relative to lift over naive.
4. e4 temporal availability: 1 if feature availability time > prediction time (from LEAK-02 / metadata), else 0.
5. e5 lineage / name hint: 1 if the name or Unity Catalog lineage shows it derives from the target's source.
6. leakage_score = max(e4, 0.6·e5, 0.4·e1 + 0.3·e2 + 0.3·e3) – hard evidence dominates.
7. Produce plain-English evidence for each suspicious feature; accept user waivers with justification.
   
# Formula / key logic
L = max(e4, 0.6·e5, 0.4·e1 + 0.3·e2 + 0.3·e3)

# Input data required
Sample; TARGET-04; LEAK-02; metadata / lineage

# Context / configuration input
target_column, prediction-time definition, feature availability (optional), UC lineage (optional), waivers

# Output generated
Per feature: leakage_score, e1..e5, evidence, risk_band, waived

# Default thresholds (PASS / WARN / FAIL)
leakage_score: WARN ≥ 0.50, FAIL ≥ 0.80 (hard gate unless waived).

# Execution tier
Sample

# ML-specific?
Yes

# Context info needed?
Yes

# ML model-related info needed
Target, prediction point / time, feature availability

# Primary build approach
Custom logic

# Databricks inbuilt capability
Unity Catalog lineage system table (system.access.column_lineage) for e5 where available

# 3rd-party libraries / functions
sklearn.ensemble.HistGradientBoostingClassifier / Regressor, sklearn.tree, sklearn.model_selection.cross_val_score

# Custom logic to write
Evidence fusion; drop-column test; evidence narrative; waiver handling

# Depends on
TARGET-04, LEAK-02

# Developer notes / edge cases
Runs its own quick model so it does not depend on BASELINE (avoids a cycle); BASE-04 adds a dataset-level 'too good to be true' check. Waivers are stored with user, timestamp and reason for audit.
