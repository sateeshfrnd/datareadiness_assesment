# Metric ID
RED-01
# Step code
REDUNDANCY
# Metric
VIF (variance inflation factor)
# What it measures
How much each numeric feature's variance is explained by the other features (multicollinearity).
# Implementation approach (step by step)
"1. On the sample: numeric features only, constants removed, median-imputed, standardised.
2. Pearson correlation matrix R; VIF_j = [R⁻¹]_jj – one matrix inversion for all features (numpy.linalg.pinv for stability).Formula / key logic

3. Cross-check with statsmodels variance_inflation_factor (regress each feature on the others) when p is small.
4. Singular R (VIF → ∞): find exact linear dependencies via matrix rank / QR and report the dependent sets.
5. Optional iterative elimination: drop the highest-VIF feature (prefer lower SIG-04 score) until all VIF ≤ threshold → suggested reduced set."
#  Formula / key logic
VIF_j = 1 / (1 − R²_j) = (R⁻¹)_jj
# Input data required
Sample (numeric features); PRED-05; SIG-04
# Context / configuration input
None
# Output generated
Per feature: vif, r2_on_others; singular_sets[]; suggested_drop_list[]
# Default thresholds (PASS / WARN / FAIL)
VIF: WARN > 5, FAIL > 10.
# Execution tier
Sample
# ML-specific?
Partial
# Context info needed?
No
# ML model-related info needed
None (most relevant for linear / interpretable models)
# Primary build approach
3rd-party library
# Databricks inbuilt capability
Optional: pyspark.ml.stat.Correlation.corr on full data for R
# 3rd-party libraries / functions
numpy.linalg.pinv, numpy.linalg.matrix_rank; statsmodels.stats.outliers_influence.variance_inflation_factor
# Custom logic to write
Matrix approach; singular-set detection; iterative elimination
# Depends on
PRED-05, SIG-04
# Developer notes / edge cases
For p > 1,000 compute VIF only for the top-N features by signal score. Categorical features need generalised VIF (GVIF) – out of scope for v1; covered by Cramér's V in RED-02.


