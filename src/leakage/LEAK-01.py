"""
LEAK-01: Target Leakage Score
----------------------------
Independent leakage-profiling module for a PySpark DataFrame.

Dependencies:
    pyspark
    pandas
    numpy
    scikit-learn

The Spark DataFrame is sampled/limited and converted to pandas for the
small scikit-learn models, as the requirement specifies a sample-tier check.

Example:
    from leak_01_target_leakage import profile_target_leakage

    result = profile_target_leakage(
        df=spark_df,
        target_column="is_fraud",
        problem_type="classification",  # or "regression" / "auto"
        prediction_time="2026-01-01 00:00:00",
        feature_availability={
            "future_txn_count": "2026-01-02 00:00:00"
        },
        lineage_derived=["target_proxy"],
        name_hints=["fraud_after_24h"],
        waivers={
            "future_txn_count": {
                "user": "satish",
                "reason": "Approved for an offline retrospective analysis"
            }
        },
    )

    result["metrics"]       # list[dict], one dictionary per feature
    result["results_df"]    # Spark DataFrame containing the metrics
    result["summary"]       # run-level details

Important:
- High predictiveness is a warning signal, not proof of leakage.
- Supply accurate prediction_time and feature_availability metadata for e4.
- Supply lineage_derived/name_hints where metadata or UC lineage is available.
- Waivers are audit records; they do not alter the raw leakage_score.
"""
import json
import argparse
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

def _as_set(values: Optional[Union[Iterable[str], Mapping[str, Any]]]) -> set:
    if values is None:
        return set()
    if isinstance(values, Mapping):
        return {str(k) for k, v in values.items() if bool(v)}
    return {str(v) for v in values}


def _normalise_time(value: Any) -> Optional[pd.Timestamp]:
    """Parse a date/time value; return None when it cannot be parsed."""
    if value is None:
        return None
    try:
        ts = pd.to_datetime(value, errors="coerce", utc=True)
        if pd.isna(ts):
            return None
        return ts
    except Exception:
        return None


def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = list(X.select_dtypes(include=[np.number, "bool"]).columns)
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median"))]
        )
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1
                )),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_cols))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_model(problem_type: str, random_state: int = 42):
    if problem_type == "classification":
        return HistGradientBoostingClassifier(
            max_iter=60, max_leaf_nodes=15, learning_rate=0.1,
            random_state=random_state
        )
    return HistGradientBoostingRegressor(
        max_iter=60, max_leaf_nodes=15, learning_rate=0.1,
        random_state=random_state
    )


def _build_tree(problem_type: str, random_state: int = 42):
    if problem_type == "classification":
        return DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=5, random_state=random_state
        )
    return DecisionTreeRegressor(
        max_depth=3, min_samples_leaf=5, random_state=random_state
    )


def _infer_problem_type(y: pd.Series) -> str:
    """Infer classification for low-cardinality targets; otherwise regression."""
    non_null = y.dropna()
    if non_null.empty:
        raise ValueError("Target column contains no non-null values.")
    if (
        non_null.dtype == object
        or str(non_null.dtype).startswith("category")
        or non_null.nunique() <= 20
    ):
        return "classification"
    return "regression"


def _classification_cv(y: pd.Series, requested_folds: int, random_state: int):
    counts = y.value_counts(dropna=False)
    min_class_count = int(counts.min())
    n_splits = min(requested_folds, min_class_count)
    if n_splits < 2:
        raise ValueError(
            "Classification CV needs at least 2 examples in every target class."
        )
    return StratifiedKFold(
        n_splits=n_splits, shuffle=True, random_state=random_state
    )


def _score_name(problem_type: str, y: pd.Series) -> str:
    if problem_type == "regression":
        return "r2"
    return "roc_auc" if y.nunique() == 2 else "roc_auc_ovr"


def _cv_score(
    X: pd.DataFrame,
    y: pd.Series,
    problem_type: str,
    cv,
    random_state: int = 42,
) -> float:
    """Cross-validated score for a feature set; returns NaN on invalid folds."""
    if X.shape[1] == 0:
        # A true no-feature baseline is handled with Dummy* estimators.
        estimator = DummyClassifier(strategy="most_frequent") if problem_type == "classification" else DummyRegressor(strategy="mean")
        scoring = "accuracy" if problem_type == "classification" else "r2"
        try:
            return float(np.mean(cross_val_score(estimator, np.zeros((len(y), 1)), y, cv=cv, scoring=scoring)))
        except Exception:
            return float("nan")

    try:
        preprocessor = _make_preprocessor(X)
        model = _build_model(problem_type, random_state)
        pipe = Pipeline([("preprocess", preprocessor), ("model", model)])
        return float(np.mean(cross_val_score(
            pipe, X, y, cv=cv, scoring=_score_name(problem_type, y),
            error_score=np.nan
        )))
    except Exception:
        return float("nan")


def profile_target_leakage(
    df: DataFrame,
    target_column: str,
    problem_type: str = "auto",
    prediction_time: Any = None,
    feature_availability: Optional[Mapping[str, Any]] = None,
    lineage_derived: Optional[Union[Iterable[str], Mapping[str, Any]]] = None,
    name_hints: Optional[Union[Iterable[str], Mapping[str, Any]]] = None,
    waivers: Optional[Mapping[str, Mapping[str, Any]]] = None,
    exclude_columns: Optional[Sequence[str]] = None,
    sample_size: int = 10000,
    top_n_drop_column: int = 10,
    cv_folds: int = 5,
    warn_threshold: float = 0.50,
    fail_threshold: float = 0.80,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Calculate LEAK-01 evidence and composite leakage score per feature.

    Parameters
    ----------
    df:
        Input PySpark DataFrame.
    target_column:
        Target/label column. It is excluded from model features.
    problem_type:
        "classification", "regression", or "auto".
    prediction_time:
        Timestamp defining when the prediction is made.
    feature_availability:
        Mapping: {feature_name: availability_timestamp}. e4=1 when the
        availability timestamp is later than prediction_time, else 0.
    lineage_derived:
        Feature names known to derive from the target/source via lineage.
    name_hints:
        Feature names flagged by the caller based on target/source name hints.
        This avoids guessing target-source relationships inside the module.
    waivers:
        Mapping: {feature: {"user": ..., "reason": ...}}. A waiver requires
        both a non-empty user and reason. Timestamp is generated for audit.
    exclude_columns:
        Optional ID/audit columns to exclude from model evidence.
    sample_size:
        Maximum number of rows converted to pandas for the sample-tier model.
    top_n_drop_column:
        Maximum features receiving the more expensive drop-column CV test.
    cv_folds:
        Requested number of CV folds (reduced when class counts are small).

    Returns
    -------
    Dictionary containing:
        metrics: list of per-feature dictionaries
        results_df: Spark DataFrame of per-feature results
        summary: run configuration and sample metadata

    Formula:
        leakage_score = max(e4, 0.6*e5, 0.4*e1 + 0.3*e2 + 0.3*e3)

    e1:
        Classification: adjusted univariate AUC, clipped to [0, 1].
        Regression: univariate cross-validated R², clipped to [0, 1].
    e2:
        Feature's share of total shallow-tree impurity importance.
    e3:
        Drop-column impact relative to model lift over a naive baseline,
        clipped to [0, 1]. Calculated only for the top suspicious features.
    e4:
        1 if feature availability is later than prediction time, otherwise 0.
        If either time is missing/unparseable, e4 is 0 and evidence notes this.
    e5:
        1 if supplied name hint or lineage evidence flags the feature, else 0.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in DataFrame.")
    if problem_type not in {"auto", "classification", "regression"}:
        raise ValueError("problem_type must be 'auto', 'classification', or 'regression'.")
    if sample_size < 50:
        raise ValueError("sample_size should be at least 50 for meaningful model checks.")

    excluded = set(exclude_columns or [])
    excluded.add(target_column)

    # Select a reproducible-ish sample without collecting the entire Spark table.
    selected_cols = [c for c in df.columns if c not in excluded]
    sampled_sdf = df.select([target_column] + selected_cols).limit(int(sample_size))
    pdf = sampled_sdf.toPandas()

    if pdf.empty:
        raise ValueError("Input DataFrame has no rows.")
    pdf = pdf.dropna(subset=[target_column]).reset_index(drop=True)
    if len(pdf) < 20:
        raise ValueError("At least 20 rows with non-null target values are required.")

    y = pdf[target_column]
    inferred = _infer_problem_type(y) if problem_type == "auto" else problem_type
    if inferred == "classification":
        # sklearn classification estimators require categorical labels; preserve
        # strings/integers as supplied, and remove unused null-target rows above.
        if y.nunique() < 2:
            raise ValueError("Classification requires at least two target classes.")
        try:
            cv = _classification_cv(y, cv_folds, random_state)
        except ValueError as exc:
            raise ValueError(f"Cannot run LEAK-01 classification check: {exc}") from exc
    else:
        y = pd.to_numeric(y, errors="coerce")
        valid = y.notna()
        pdf = pdf.loc[valid].reset_index(drop=True)
        y = y.loc[valid].reset_index(drop=True)
        if len(y) < 20:
            raise ValueError("Regression requires at least 20 rows with numeric target values.")
        cv = KFold(n_splits=min(cv_folds, len(y)), shuffle=True, random_state=random_state)

    X = pdf[[c for c in selected_cols if c in pdf.columns]].copy()
    # Drop columns that cannot be used meaningfully by the model.
    usable_cols = []
    for col in X.columns:
        if X[col].nunique(dropna=True) > 1:
            usable_cols.append(col)
    X = X[usable_cols]

    if X.empty:
        raise ValueError("No usable feature columns remain after exclusions.")

    # Avoid unsupported/unstable complex values by converting them to strings.
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = X[col].map(
                lambda value: str(value) if isinstance(value, (dict, list, tuple, set)) else value
            )

    availability = dict(feature_availability or {})
    lineage_set = _as_set(lineage_derived)
    hint_set = _as_set(name_hints)
    waiver_map = dict(waivers or {})
    pred_ts = _normalise_time(prediction_time)

    # Fit a shallow tree to estimate the share of total feature importance.
    try:
        tree_preprocessor = _make_preprocessor(X)
        X_tree = tree_preprocessor.fit_transform(X)
        tree = _build_tree(inferred, random_state)
        tree.fit(X_tree, y)
        raw_importances = np.asarray(tree.feature_importances_, dtype=float)
        # ColumnTransformer can expand/transform columns. For a stable per-original
        # feature attribution, fit one shallow tree per original feature instead.
    except Exception:
        raw_importances = np.array([])

    tree_importance: Dict[str, float] = {}
    # One-feature shallow-tree importance gives a consistent original-column
    # attribution even when categoricals are ordinal-encoded.
    for col in X.columns:
        try:
            pre = _make_preprocessor(X[[col]])
            transformed = pre.fit_transform(X[[col]])
            one_tree = _build_tree(inferred, random_state)
            one_tree.fit(transformed, y)
            # The feature's predictive split strength is represented by impurity
            # reduction (tree feature importance is 1 for a used single feature).
            # To estimate dominance, use the tree's root impurity reduction.
            root_impurity = float(one_tree.tree_.impurity[0])
            left = int(one_tree.tree_.children_left[0])
            right = int(one_tree.tree_.children_right[0])
            if left >= 0 and right >= 0 and len(y) > 0:
                n = one_tree.tree_.n_node_samples
                weighted_child = (
                    n[left] * one_tree.tree_.impurity[left]
                    + n[right] * one_tree.tree_.impurity[right]
                ) / max(n[0], 1)
                reduction = max(0.0, root_impurity - weighted_child)
            else:
                reduction = 0.0
            tree_importance[col] = reduction
        except Exception:
            tree_importance[col] = 0.0

    importance_total = sum(tree_importance.values())
    e2_map = {
        col: (tree_importance[col] / importance_total if importance_total > 0 else 0.0)
        for col in X.columns
    }

    # e1: univariate predictive strength.
    e1_map: Dict[str, float] = {}
    for col in X.columns:
        try:
            univariate_score = _cv_score(X[[col]], y, inferred, cv, random_state)
            if inferred == "classification" and y.nunique() == 2:
                # Convert AUC from [0.5, 1] to [0, 1]; account for reversed direction.
                # cross_val_score ROC AUC can be below 0.5 for a consistently reversed
                # relationship, so use distance from chance.
                e1 = min(1.0, max(0.0, (abs(univariate_score - 0.5) / 0.5)))
            elif inferred == "classification":
                # Multiclass ROC AUC baseline is not 0.5; use a bounded score.
                e1 = min(1.0, max(0.0, univariate_score if np.isfinite(univariate_score) else 0.0))
            else:
                e1 = min(1.0, max(0.0, univariate_score if np.isfinite(univariate_score) else 0.0))
            e1_map[col] = float(e1) if np.isfinite(e1) else 0.0
        except Exception:
            e1_map[col] = 0.0

    # Preliminary score selects up to top_n_drop_column candidates for e3.
    prelim = {}
    for col in X.columns:
        if isinstance(availability.get(col), bool):
            e4 = int(not availability[col])
        else:
            available_ts = _normalise_time(availability.get(col))
            e4 = int(
                pred_ts is not None
                and available_ts is not None
                and available_ts > pred_ts
            )
        # available_ts = _normalise_time(availability.get(col))
        e4 = int(pred_ts is not None and available_ts is not None and available_ts > pred_ts)
        e5 = int(col in lineage_set or col in hint_set)
        prelim[col] = max(
            e4, 0.6 * e5,
            0.4 * e1_map.get(col, 0.0) + 0.3 * e2_map.get(col, 0.0)
        )
    drop_candidates = sorted(X.columns, key=lambda c: prelim[c], reverse=True)[:max(0, int(top_n_drop_column))]

    # e3: relative drop-column impact compared with the lift over a naive predictor.
    all_score = _cv_score(X, y, inferred, cv, random_state)
    if inferred == "classification":
        naive_model = DummyClassifier(strategy="most_frequent")
        naive_scoring = "accuracy"
    else:
        naive_model = DummyRegressor(strategy="mean")
        naive_scoring = "r2"
    try:
        naive_score = float(np.mean(cross_val_score(
            naive_model, np.zeros((len(y), 1)), y, cv=cv,
            scoring=naive_scoring, error_score=np.nan
        )))
    except Exception:
        naive_score = 0.0

    e3_map = {col: 0.0 for col in X.columns}
    drop_scores = {}
    lift = all_score - naive_score if np.isfinite(all_score) and np.isfinite(naive_score) else 0.0
    for col in drop_candidates:
        X_without = X.drop(columns=[col])
        without_score = _cv_score(X_without, y, inferred, cv, random_state)
        drop_scores[col] = without_score
        impact = all_score - without_score if np.isfinite(all_score) and np.isfinite(without_score) else 0.0
        if lift > 1e-12:
            e3_map[col] = float(np.clip(impact / lift, 0.0, 1.0))
        else:
            e3_map[col] = 0.0

    metrics = []
    now_iso = datetime.now(timezone.utc).isoformat()
    for col in X.columns:
        available_ts = _normalise_time(availability.get(col))
        e4 = int(pred_ts is not None and available_ts is not None and available_ts > pred_ts)
        e5 = int(col in lineage_set or col in hint_set)
        e1 = float(e1_map.get(col, 0.0))
        e2 = float(e2_map.get(col, 0.0))
        e3 = float(e3_map.get(col, 0.0))
        score = float(np.clip(max(e4, 0.6 * e5, 0.4 * e1 + 0.3 * e2 + 0.3 * e3), 0.0, 1.0))

        evidence = []
        if e4:
            evidence.append("Feature becomes available after the defined prediction time.")
        elif col in availability and pred_ts is not None and available_ts is not None:
            evidence.append("Feature availability is not later than the prediction time.")
        elif col in availability:
            evidence.append("Availability metadata or prediction time could not be parsed.")
        else:
            evidence.append("No feature-availability timestamp supplied; temporal leakage was not confirmed.")
        if col in lineage_set:
            evidence.append("Lineage metadata flags this feature as derived from the target/source.")
        if col in hint_set:
            evidence.append("A caller-supplied feature-name hint flags a possible target/source relationship.")
        if e1 >= 0.5:
            evidence.append("Strong univariate predictive signal; investigate for proxies or post-outcome data.")
        elif e1 >= 0.2:
            evidence.append("Moderate univariate predictive signal.")
        if e2 > 0.5:
            evidence.append("Feature dominates more than half of the shallow-tree importance evidence.")
        if col in drop_candidates and e3 >= 0.5:
            evidence.append("Removing this feature removes a large share of model lift over the naive baseline.")
        if not evidence:
            evidence.append("No strong leakage evidence detected by the supplied checks.")

        if score >= fail_threshold:
            risk_band = "FAIL"
        elif score >= warn_threshold:
            risk_band = "WARN"
        else:
            risk_band = "PASS"

        waiver_input = waiver_map.get(col)
        waiver_record = None
        waived = False
        if isinstance(waiver_input, Mapping):
            user = str(waiver_input.get("user", "")).strip()
            reason = str(waiver_input.get("reason", "")).strip()
            if user and reason:
                waived = True
                waiver_record = {
                    "user": user,
                    "reason": reason,
                    "timestamp_utc": now_iso,
                }
                evidence.append(f"Waiver supplied by {user}: {reason}")

        metrics.append({
            "feature": str(col),
            "leakage_score": score,
            "e1": e1,
            "e2": e2,
            "e3": e3,
            "e4": e4,
            "e5": e5,
            "evidence": " ".join(evidence),
            "risk_band": risk_band,
            "waived": bool(waived),
            "waiver": waiver_record,
            "univariate_cv_score": float(_cv_score(X[[col]], y, inferred, cv, random_state)),
            "drop_column_cv_score": (
                float(drop_scores[col]) if col in drop_scores and np.isfinite(drop_scores[col]) else None
            ),
            "availability_time": str(availability.get(col)) if col in availability else None,
        })

    # Higher risk first.
    metrics.sort(key=lambda item: item["leakage_score"], reverse=True)

    result_sdf = df.sparkSession.createDataFrame(pd.DataFrame([
        {
            "feature": m["feature"],
            "leakage_score": float(m["leakage_score"]),
            "e1": float(m["e1"]),
            "e2": float(m["e2"]),
            "e3": float(m["e3"]),
            "e4": int(m["e4"]),
            "e5": int(m["e5"]),
            "evidence": m["evidence"],
            "risk_band": m["risk_band"],
            "waived": bool(m["waived"]),
            "waiver_user": m["waiver"]["user"] if m["waiver"] else None,
            "waiver_reason": m["waiver"]["reason"] if m["waiver"] else None,
            "waiver_timestamp_utc": m["waiver"]["timestamp_utc"] if m["waiver"] else None,
        }
        for m in metrics
    ])) if metrics else df.sparkSession.createDataFrame(
        [], "feature string, leakage_score double, e1 double, e2 double, e3 double, e4 int, e5 int, evidence string, risk_band string, waived boolean, waiver_user string, waiver_reason string, waiver_timestamp_utc string"
    )

    summary = {
        "metric_id": "LEAK-01",
        "metric_name": "Target leakage score",
        "target_column": target_column,
        "problem_type": inferred,
        "prediction_time": str(prediction_time) if prediction_time is not None else None,
        "sample_rows_used": int(len(pdf)),
        "features_scored": int(len(metrics)),
        "all_features_cv_score": float(all_score) if np.isfinite(all_score) else None,
        "naive_baseline_cv_score": float(naive_score) if np.isfinite(naive_score) else None,
        "drop_column_features_tested": list(drop_candidates),
        "warn_threshold": float(warn_threshold),
        "fail_threshold": float(fail_threshold),
        "note": "Sample-tier screening; investigate WARN/FAIL features and validate metadata/lineage.",
        "metrics": metrics,
    }

    return {
         "summary": summary,
        "results_df": result_sdf,       
    }

# ==========================================================
# PARAMETERS
# ==========================================================
def get_args(print_args = False):
    parser = argparse.ArgumentParser()

    # LEAK-01
    parser.add_argument("--leak_01_target_column", type=str, default="Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22", help="Target/label column. It is excluded from model features.")
    parser.add_argument("--leak_01_problem_type", type=str, default="regression", help="problem_type: 'classification', 'regression', or 'auto'.")
    parser.add_argument("--leak_01_prediction_time", type=str, default=None, help="Timestamp defining when the prediction is made.")
    parser.add_argument("--leak_01_feature_availability", type=json.loads, default = {
        "Mazzoni 1- Sigma Mixer current_mean": True,
        "Mazzoni 1- Sigma Mixer current_sum": True,
        "Mazzoni 1- Sigma Plodder pressure_mean10to22": True,
        "Mazzoni 1- Sigma Plodder pressure_std10to22": True,
        "Mazzoni 1- Sigma Plodder pressure_after_production": "2026-01-01 22:00:00"
    }, help = "Mapping: {feature_name: availability_timestamp}. e4=1 when the availability timestamp is later than prediction_time, else 0." )
    parser.add_argument("--leak_01_lineage_derived", nargs="+", default=[],help = "Feature names known to derive from the target/source via lineage.")
    parser.add_argument("--leak_01_name_hints", nargs="+", default=[],help = "Feature names flagged by the caller based on target/source name hints.\n This avoids guessing target-source relationships inside the module.")
    parser.add_argument("--leak_01_waivers", type=json.loads, default= {
        "Mazzoni 1- Sigma Mixer current_mean": {
            "user": "satish",
            "reason": (
                "Measurement is captured before the prediction "
                "decision and approved for production use."
            )
        },
        "Mazzoni 1- Sigma Plodder pressure_mean10to22": {
            "user": "satish",
            "reason": (
                "The feature is generated from the completed "
                "10to22 measurement window, which ends before "
                "the prediction is made."
            )
        }
    },help = 'Mapping: {feature: {"user": ..., "reason": ...}}. A waiver requires both a non-empty user and reason. Timestamp is generated for audit.')
    parser.add_argument("--leak_01_exclude_columns", nargs="+", default=[],help = "Optional ID/audit columns to exclude from model evidence.")
    parser.add_argument("--leak_01_sample_size", type=int, default=10000, help = "Maximum number of rows converted to pandas for the sample-tier model.")
    parser.add_argument("--leak_01_top_n_drop_column", type=int, default=10 ,help = "Maximum features receiving the more expensive drop-column CV test.")
    parser.add_argument("--leak_01_cv_folds", type=int, default=5,help = "Requested number of CV folds (reduced when class counts are small).")

    args, unknown_args = parser.parse_known_args()
    if print_args:
        for arg in vars(args):
            print(arg,getattr(args, arg))
    return args, unknown_args

def get_config():
    args, unknown_args = get_args()
    config = {
        "leak_01_target_column": args.leak_01_target_column,
        "leak_01_problem_type": args.leak_01_problem_type ,    
        "leak_01_prediction_time": args.leak_01_prediction_time ,
        "leak_01_feature_availability": args.leak_01_feature_availability ,
        "leak_01_lineage_derived": args.leak_01_lineage_derived ,
        "leak_01_name_hints": args.leak_01_name_hints ,
        "leak_01_waivers": args.leak_01_waivers ,    
        "leak_01_exclude_columns": args.leak_01_exclude_columns ,
        "leak_01_sample_size": args.leak_01_sample_size ,
        "leak_01_top_n_drop_column": args.leak_01_top_n_drop_column,
        "leak_01_cv_folds":args.leak_01_cv_folds 
    }
    return config


def main():    
    config = get_config()
    print(f"config = {config}")
    # path = '/Workspace/Users/sateeshfrnd@gmail.com/SAMBA/datareadiness_assesment_p/sample_datasets/data_to_model.csv'    
    path = '/Workspace/Users/sateeshfrnd@gmail.com/SAMBA/datareadiness_assesment_p/sample_datasets/TotalSummaryDatawithFormulationwithRPM1022and1625.csv'    
    df = spark.read.csv(
        path,
        header=True,
        inferSchema=True
    ) 
    # df.printSchema()   
    # df.show(5,False)

    profile_target_leakage_results = profile_target_leakage(
        df=df,
        target_column = config["leak_01_target_column"],
        problem_type = config["leak_01_problem_type"],
        prediction_time = config["leak_01_prediction_time"],
        feature_availability = config["leak_01_feature_availability"],
        lineage_derived = config["leak_01_lineage_derived"],
        name_hints = config["leak_01_name_hints"],
        waivers = config["leak_01_waivers"],
        exclude_columns = config["leak_01_exclude_columns"],
        sample_size = config["leak_01_sample_size"],
        top_n_drop_column = config["leak_01_top_n_drop_column"],
        cv_folds = config["leak_01_cv_folds"]
    )
    print(profile_target_leakage_results["results_df"])    # Spark DataFrame containing the metrics
    display(profile_target_leakage_results["results_df"])
    print("========= summary ==========")
    print(json.dumps(profile_target_leakage_results["summary"], indent=4))     # run-level details

    


    



if __name__ == "__main__":
    main() 