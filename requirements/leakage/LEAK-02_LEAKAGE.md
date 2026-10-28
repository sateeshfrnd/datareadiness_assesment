# Metric ID
LEAK-02

# Step code
LEAKAGE

# Metric
Temporal order violations

# What it measures
Detects records or features that break time ordering – information from the future relative to the prediction point – and invalid or out-of-order timestamps.

# Implementation approach (step by step)
Applicable when time_column is provided (otherwise NA).
1. Validity: count null / unparseable timestamps, future timestamps (> extraction or run date) and implausibly old dates.
2. Feature-after-label: for each declared feature-timestamp column or availability rule, count rows where feature_ts > prediction_ts (or label_ts) → hard violation.
3. Entity sequence (when an entity / group column exists): Window.partitionBy(entity).orderBy(time) with lag() / row_number() to find duplicate timestamps per entity and non-monotonic sequences versus ingestion order.
4. Label-window overlap: if the label covers [t, t + h], flag features whose aggregation windows end after t (from metadata).
5. Split simulation: share of entities whose records would straddle train and test under a random split → informs EVAL-01 (time-based split needed).

# Formula / key logic
violation_pct = count(feature_ts > prediction_ts) / n_rows

# Input data required
Dataset (time, entity, feature-timestamp columns)

# Context / configuration input
time_column, prediction-time rule, label horizon, entity / group column, feature availability metadata, extraction date

# Output generated
invalid_ts_count, future_ts_count, feature_after_label_count{feature}, duplicate_ts_per_entity, out_of_order_pct, entities_spanning_split_pct

# Default thresholds (PASS / WARN / FAIL)
feature_after_label_count: FAIL > 0 (hard gate). Invalid / future timestamps: WARN > 0.1%. out_of_order_pct: WARN > 1%.

# Execution tier
Sample + grouped (window functions run on full data per entity)

# ML-specific?
Yes

# Context info needed?
Yes

# ML model-related info needed
Prediction-time definition, label horizon, time and entity columns

# Primary build approach
Custom logic

# Databricks inbuilt capability
Spark window functions (lag, row_number), try_to_timestamp, current_date()

# 3rd-party libraries / functions
--

# Custom logic to write
Rule set; metadata-driven availability checks; split simulation

# Depends on
INTAKE-02, INTAKE-05

# Developer notes / edge cases
Most temporal leakage can only be found with metadata about when each feature becomes known – make feature availability a first-class optional input in the context form.
