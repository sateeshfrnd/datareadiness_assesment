# Metric ID
MISS-02

# Step code
MISSING

#Metric
MCAR vs MAR test

# What it measures
Tests whether missingness is completely at random (MCAR) or depends on observed variables (MAR); determines whether deletion is unbiased and which imputation fits.

# Implementation approach (step by step)
Global – Little's MCAR test (numeric features with any missingness, sample ≤ 50k rows):
1. Estimate mean vector μ and covariance Σ with the EM algorithm.
2. Group rows by missing-data pattern j (n_j rows, observed variable set o_j).
3. d² = Σ_j n_j (x̄_j,o − μ_o)ᵀ Σ_oo⁻¹ (x̄_j,o − μ_o); df = Σ_j |o_j| − p.
4. p-value = 1 − χ²_cdf(d², df); p ≥ 0.05 → consistent with MCAR.
Per column (missing_pct ≥ 1%) – MAR evidence:
5. Indicator M_k = 1 if missing.
6. Compare other features between M_k = 1 and 0: numeric → Mann–Whitney U (rank-biserial effect); categorical → chi-square (Cramér's V); Benjamini–Hochberg adjustment.
7. Multivariate: logistic regression M_k ~ other features (imputed), 5-fold CV ROC-AUC; AUC ≥ 0.65 → missingness predictable → MAR.
8. Informative missingness: test M_k against the target; significant → recommend a missing-indicator feature.
9. Label each column: MCAR-consistent / MAR (driving variables listed) / possibly MNAR (related to target but not to observed features – cannot be proven).


# Formula / key logic
Little: d² ~ χ²(df). Per column: AUC(M_k ~ X_−k)

# Input data required
Stratified sample; MISS-01 results

# Context / configuration input
target_column (informative missingness)

# Output generated
little_chi2, little_df, little_pvalue, global_mechanism; per column: mechanism_label, missing_model_auc, top_associated_vars[], target_assoc_p

# Default thresholds (PASS / WARN / FAIL)
Little p < 0.05 → WARN (not MCAR; listwise deletion biased). Per-column AUC ≥ 0.65 → WARN (MAR). Target association p < 0.01 → INFO (add indicator).

# Execution tier
Sample

# ML-specific?
Partial

# Context info needed?
Partial

# ML model-related info needed
Target column (informative missingness)

# Primary build approach
Custom logic

# Databricks inbuilt capability
— (sample collected with sampleBy → toPandas)
# 3rd-party libraries / functions
scipy.stats (chi2, mannwhitneyu, chi2_contingency); statsmodels.stats.multitest.multipletests; sklearn LogisticRegression; pyampute MCARTest as reference implementation of Little's test (check maintenance status before adopting)

# Custom logic to write
Little's test with EM (≈ 100 lines); pattern grouping; per-column diagnostics; labelling rules

# Depends on
MISS-01, INTAKE-02
# Developer notes / edge cases
Little's test rejects at large n almost always – keep the sample ≤ 50k and present it as indicative. Use numeric features only (one-hot categoricals inflate df). Status NA when no column has ≥ 1% missing. Report wording must say MNAR is 'possible', never 'confirmed'.
