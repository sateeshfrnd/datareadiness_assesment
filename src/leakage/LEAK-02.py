"""
LEAK-02: Temporal Order Violations
Independent PySpark module.

Usage:
    from leak_02_temporal_order import run_leak02
    results = run_leak02(df, config)

Returns a Python dictionary. Counts/percentages are collected to the driver;
use this module for metric reporting, not for returning the full input data.
"""

from datetime import date, datetime
from typing import Any, Dict, Optional

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

def _count_true(df: DataFrame, condition) -> int:
    return int(df.filter(condition).count())


def _parse_timestamp(column_name: str):
    # to_timestamp supports common Spark timestamp formats. If a value cannot
    # be parsed, Spark returns NULL under normal non-ANSI timestamp parsing.
    return F.to_timestamp(F.col(column_name))


def run_leak02(df: DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Required config:
      time_column: timestamp column for each record

    Optional config:
      entity_column: entity/group identifier
      ingestion_order_column: actual ingestion sequence column (recommended)
      extraction_date: 'YYYY-MM-DD'; defaults to current date on Spark
      oldest_allowed_date: 'YYYY-MM-DD'; optional implausibly-old boundary
      prediction_time_column: timestamp against which feature availability is checked
      feature_timestamp_columns: {'feature_name': 'feature_timestamp_column', ...}
      label_time_column: alternative comparison timestamp if prediction time is absent
      aggregation_window_end_columns: {'feature_name': 'aggregation_end_timestamp_column', ...}
      random_split_test_fraction: default 0.2
      random_split_seed: default 42
      invalid_future_warn_pct: default 0.1
      out_of_order_warn_pct: default 1.0
      sample_fraction: optional fraction in (0, 1]; sequence checks remain grouped
                      on the selected sample if this is set.

    Notes:
      - Feature-after-prediction is a hard failure if any configured feature
        timestamp is later than the prediction/label timestamp.
      - For genuine ingestion-order checks, pass ingestion_order_column.
        Without it, the module uses monotonically_increasing_id as a fallback,
        which is not guaranteed to represent real source ingestion order.
      - Split simulation randomly assigns rows to train/test, then measures
        the share of entities represented in both splits.
    """
    time_col = config.get("time_column")
    if not time_col or time_col not in df.columns:
        return {
            "metric_id": "LEAK-02",
            "metric_name": "Temporal order violations",
            "status": "NA",
            "reason": "time_column is not provided or is not present in the DataFrame",
        }

    entity_col = config.get("entity_column")
    prediction_col = config.get("prediction_time_column")
    label_col = config.get("label_time_column")
    compare_col = prediction_col if prediction_col in df.columns else (
        label_col if label_col in df.columns else None
    )
    feature_ts_cols = config.get("feature_timestamp_columns", {})
    window_end_cols = config.get("aggregation_window_end_columns", {})
    extraction_date = config.get("extraction_date")
    oldest_allowed_date = config.get("oldest_allowed_date")
    sample_fraction = config.get("sample_fraction")

    work = df
    if sample_fraction is not None and 0 < float(sample_fraction) < 1:
        work = work.sample(
            withReplacement=False,
            fraction=float(sample_fraction),
            seed=int(config.get("random_split_seed", 42)),
        )

    # Parse the main timestamp; preserve raw column so invalid values are counted.
    work = work.withColumn("__leak02_time", _parse_timestamp(time_col))

    total_rows = int(work.count())
    invalid_ts_count = _count_true(
        work,
        F.col(time_col).isNull() | F.col("__leak02_time").isNull()
    )

    if extraction_date:
        extraction_expr = F.to_timestamp(F.lit(extraction_date))
    else:
        extraction_expr = F.current_timestamp()

    future_ts_count = _count_true(
        work,
        F.col("__leak02_time").isNotNull()
        & (F.col("__leak02_time") > extraction_expr)
    )

    implausibly_old_count = 0
    if oldest_allowed_date:
        oldest_expr = F.to_timestamp(F.lit(oldest_allowed_date))
        implausibly_old_count = _count_true(
            work,
            F.col("__leak02_time").isNotNull()
            & (F.col("__leak02_time") < oldest_expr)
        )

    # Feature availability: feature timestamp must not be later than the
    # prediction time, or label time when prediction time is not supplied.
    feature_after_label_count = {}
    feature_violation_pct = {}
    for feature_name, feature_ts_col in feature_ts_cols.items():
        if feature_ts_col not in work.columns or compare_col is None:
            feature_after_label_count[feature_name] = None
            feature_violation_pct[feature_name] = None
            continue

        feature_ts = _parse_timestamp(feature_ts_col)
        comparison_ts = _parse_timestamp(compare_col)
        violation_condition = (
            feature_ts.isNotNull()
            & comparison_ts.isNotNull()
            & (feature_ts > comparison_ts)
        )
        count = _count_true(work, violation_condition)
        feature_after_label_count[feature_name] = count
        feature_violation_pct[feature_name] = (
            round((count / total_rows) * 100, 4) if total_rows else 0.0
        )

    # Entity-level duplicate timestamps and ingestion-order reversals.
    duplicate_ts_per_entity = 0
    out_of_order_count = 0
    out_of_order_pct = None
    entities_spanning_split_pct = None
    entities_spanning_split_count = 0
    entity_count = 0

    if entity_col and entity_col in work.columns:
        valid_entity_time = work.filter(
            F.col(entity_col).isNotNull() & F.col("__leak02_time").isNotNull()
        )

        duplicate_groups = (
            valid_entity_time
            .groupBy(entity_col, "__leak02_time")
            .count()
            .filter(F.col("count") > 1)
        )
        # Count duplicate entity-timestamp groups (not all rows in those groups).
        duplicate_ts_per_entity = int(duplicate_groups.count())

        if config.get("ingestion_order_column") in work.columns:
            order_col = config["ingestion_order_column"]
        else:
            order_col = "__leak02_fallback_ingest_id"
            valid_entity_time = valid_entity_time.withColumn(
                order_col, F.monotonically_increasing_id()
            )

        # Compare each record's event time with the previous event time in
        # ingestion order. A smaller timestamp than the previous one is out of order.
        ingestion_window = Window.partitionBy(entity_col).orderBy(F.col(order_col))
        sequenced = valid_entity_time.withColumn(
            "__leak02_prev_time", F.lag("__leak02_time").over(ingestion_window)
        )
        out_of_order_count = _count_true(
            sequenced,
            F.col("__leak02_prev_time").isNotNull()
            & (F.col("__leak02_time") < F.col("__leak02_prev_time"))
        )
        sequence_rows = int(valid_entity_time.count())
        out_of_order_pct = (
            round((out_of_order_count / sequence_rows) * 100, 4)
            if sequence_rows else 0.0
        )

        entity_count = int(
            valid_entity_time.select(entity_col).distinct().count()
        )

        # Random row-level split simulation: does each entity appear in both sets?
        test_fraction = float(config.get("random_split_test_fraction", 0.2))
        if 0 < test_fraction < 1:
            split_seed = int(config.get("random_split_seed", 42))
            split_df = valid_entity_time.withColumn(
                "__leak02_split",
                F.when(F.rand(split_seed) < test_fraction, F.lit("test"))
                 .otherwise(F.lit("train"))
            )
            split_membership = (
                split_df.groupBy(entity_col)
                .agg(F.countDistinct("__leak02_split").alias("__split_count"))
            )
            entities_spanning_split_count = int(
                split_membership.filter(F.col("__split_count") > 1).count()
            )
            entities_spanning_split_pct = (
                round((entities_spanning_split_count / entity_count) * 100, 4)
                if entity_count else 0.0
            )

    # Label-window overlap: feature aggregation end must not be after label start.
    label_window_overlap_count = {}
    label_start_col = prediction_col if prediction_col in work.columns else label_col
    for feature_name, window_end_col in window_end_cols.items():
        if window_end_col not in work.columns or not label_start_col:
            label_window_overlap_count[feature_name] = None
            continue
        end_ts = _parse_timestamp(window_end_col)
        label_start_ts = _parse_timestamp(label_start_col)
        label_overlap = (
            end_ts.isNotNull()
            & label_start_ts.isNotNull()
            & (end_ts > label_start_ts)
        )
        label_window_overlap_count[feature_name] = _count_true(work, label_overlap)

    invalid_pct = (invalid_ts_count / total_rows * 100) if total_rows else 0.0
    future_pct = (future_ts_count / total_rows * 100) if total_rows else 0.0
    invalid_future_warn_pct = float(config.get("invalid_future_warn_pct", 0.1))
    out_of_order_warn_pct = float(config.get("out_of_order_warn_pct", 1.0))

    any_feature_violation = any(
        value is not None and value > 0
        for value in feature_after_label_count.values()
    )
    if any_feature_violation:
        status = "FAIL"
    elif (
        invalid_pct > invalid_future_warn_pct
        or future_pct > invalid_future_warn_pct
        or (out_of_order_pct is not None and out_of_order_pct > out_of_order_warn_pct)
    ):
        status = "WARN"
    else:
        status = "PASS"

    return {
        "metric_id": "LEAK-02",
        "metric_name": "Temporal order violations",
        "status": status,
        "total_rows": total_rows,
        "invalid_ts_count": invalid_ts_count,
        "future_ts_count": future_ts_count,
        "implausibly_old_count": implausibly_old_count,
        "invalid_ts_pct": round(invalid_pct, 4),
        "future_ts_pct": round(future_pct, 4),
        "feature_after_label_count": feature_after_label_count,
        "feature_violation_pct": feature_violation_pct,
        "duplicate_ts_per_entity": duplicate_ts_per_entity,
        "out_of_order_count": out_of_order_count,
        "out_of_order_pct": out_of_order_pct,
        "label_window_overlap_count": label_window_overlap_count,
        "entity_count": entity_count,
        "entities_spanning_split_count": entities_spanning_split_count,
        "entities_spanning_split_pct": entities_spanning_split_pct,
        "notes": [
            "Feature-after-prediction/label violations are a hard failure when count > 0.",
            "Pass ingestion_order_column for meaningful ingestion-sequence checks.",
            "Random split simulation is indicative; it does not replace EVAL-01 evaluation.",
            "Missing or unparsable feature timestamps are not counted as feature-after-label violations; monitor timestamp validity separately.",
        ],
    }

def main():
    # path = "abfss://data-readiness@bnlwestgunileverbd00036.dfs.core.windows.net/Source/data_to_model_20261014.csv"  
    # path = '/Workspace/Users/sateeshfrnd@gmail.com/SAMBA/datareadiness_assesment_p/sample_datasets/data_to_model.csv'    
    path = '/Workspace/Users/sateeshfrnd@gmail.com/SAMBA/datareadiness_assesment_p/sample_datasets/TotalSummaryDatawithFormulationwithRPM1022and1625.csv'    
    df = spark.read.csv(
        path,
        header=True,
        inferSchema=True
    ) 
    df.printSchema()   
    df.show(5,False)
    config = {
        "time_column": None,
        "entity_column": "TrialBusinessId_ of Sigma"
    }

    '''
    # If your original data contains timestamps that were removed during aggregation, retain them or join the appropriate timestamp metadata before running this metric.

    # For example, suppose you have a timestamp column named event_timestamp and a prediction cutoff named prediction_timestamp:
    config = {
    "time_column": "event_timestamp",
    "entity_column": "TrialBusinessId_ of Sigma",
    "prediction_time_column": "prediction_timestamp",
    "extraction_date": "2026-10-27",
    "oldest_allowed_date": "2020-01-01",
    "random_split_test_fraction": 0.2,
    "random_split_seed": 42
}

    '''

    results = run_leak02(df, config)

    print(results)

if __name__ == "__main__":
    main()
