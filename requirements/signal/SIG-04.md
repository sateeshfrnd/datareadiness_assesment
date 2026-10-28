# Metric ID
SIG-04
# Step code
SIGNAL
# Metric
Normalised signal score and rank
# What it measures
Single composite score per feature combining MI, ANOVA / association and rank correlation on a common absolute [0, 1] scale, with ranking and a noise-floor comparison.
# Implementation approach (step by step)
1. Inputs per feature, all on a 0–1 correlation-like scale: s_MI = r_MI (SIG-01); s_F = η or Cramér's V (SIG-02); s_ρ = |ρ| or √ε² (SIG-03).
2. Composite = Σ w_i·s_i / Σ w_i over available measures (default w_MI = 0.4, w_F = 0.3, w_ρ = 0.3; missing measure → weights re-normalised).
3. Rank features descending (dense rank) and store percentile rank.
4. Band: strong ≥ 0.30, moderate 0.10–0.30, weak < 0.10 (Cohen's r conventions).
5. Noise check: feature flagged 'indistinguishable from noise' if composite ≤ the shadow features' composite (shadows scored the same way).
6. Step-level measures: max_signal_score, n_strong, n_moderate, n_above_noise.

# Formula / key logic
score_j = (0.4·r_MI + 0.3·η + 0.3·|ρ|) / Σw ; rank = dense_rank(score desc)

# Input data required
SIG-01, SIG-02, SIG-03 outputs

# Context / configuration input
Signal weights (config)

# Output generated
Per feature: signal_score, rank, percentile, band, above_noise; step: max_signal_score, n_strong, n_moderate, n_above_noise

# Default thresholds (PASS / WARN / FAIL)
max_signal_score: WARN < 0.30, FAIL < 0.10. WARN if n_above_noise = 0.

# Execution tier
Sample

# ML-specific?
Yes

# Context info needed?
Yes

# ML model-related info needed
Target (inherited from SIG-01..03)

# Primary build approach
Custom logic

# Databricks inbuilt capability
—

# 3rd-party libraries / functions
pandas (rank)

# Custom logic to write
Absolute normalisation; weighting; banding; noise comparison

# Depends on
SIG-01, SIG-02, SIG-03

# Developer notes / edge cases
Absolute (not min–max) normalisation is deliberate: min–max always gives the top feature 1.0 and hides datasets with no real signal. The signal score is reused as the weight when rolling up feature-level metrics in other steps.







