
"""
LEAK-01: Target Leakage Profiler
===============================

What this module does
---------------------
1. Takes a Spark DataFrame and a binary classification target.
2. Samples a bounded number of rows and converts that sample to pandas.
3. Calculates:
   e1: how predictive each feature is on its own (cross-validated ROC AUC)
   e2: each feature's share of shallow decision-tree importance
   e3: drop-column impact for selected suspicious features
   e4: whether a feature is available at prediction time
   e5: whether feature name/lineage suggests target-source derivation
4. Combines evidence into a 0-1 leakage score.
5. Returns per-feature metrics, evidence, risk band, and waiver fields.
"""

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# ---------------------------------------------------------------------
# 1. Configuration: change these defaults to suit your dataset
# ---------------------------------------------------------------------

DEFAULT_WARN_THRESHOLD = 0.50
DEFAULT_FAIL_THRESHOLD = 0.80

DEFAULT_E1_WEIGHT = 0.40
DEFAULT_E2_WEIGHT = 0.30
DEFAULT_E3_WEIGHT = 0.30

# ---------------------------------------------------------------------
# 2. Helper functions for scikit-learn preprocessing
# ---------------------------------------------------------------------

def _make_one_hot_encoder():
    """Create an encoder compatible with multiple scikit-learn versions."""
    try:
        # scikit-learn >= 1.2
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # Older scikit-learn
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _prepare_features(pdf, feature_columns):
    """
    Separate numeric and categorical features.
    Remove columns that are all-null or constant in the sample.
    """
    X = pdf[feature_columns].copy()
    usable_features = []
    numeric_features = []
    categorical_features = []

    for column in feature_columns:
        series = X[column]

        # Such a feature cannot help the model in this sample.
        if series.isna().all() or series.nunique(dropna=False) <= 1:
            continue

        usable_features.append(column)

        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
        ):
            numeric_features.append(column)
        else:
            categorical_features.append(column)
            # Convert categories to strings and give nulls a clear category.
            X[column] = (
                series.astype("string")
                .fillna("__LEAKAGE_NULL__")
                .astype(str)
            )

    return X[usable_features], usable_features, numeric_features, categorical_features


def _build_preprocessor(columns, numeric_features, categorical_features):
    """Fill numeric nulls and one-hot encode categorical features."""
    numeric_columns = [c for c in columns if c in numeric_features]
    categorical_columns = [c for c in columns if c in categorical_features]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", _make_one_hot_encoder()),
        ])
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    return ColumnTransformer(transformers=transformers, remainder="drop")


def _build_classifier(columns, numeric_features, categorical_features, random_state):
    """Build a model pipeline so preprocessing is fitted inside each CV fold."""
    return Pipeline([
        (
            "preprocessing",
            _build_preprocessor(columns, numeric_features, categorical_features),
        ),
        (
            "model",
            HistGradientBoostingClassifier(
                max_iter=50,
                max_leaf_nodes=7,
                random_state=random_state,
            ),
        ),
    ])


def _cross_validated_auc(
    X, y, columns, numeric_features, categorical_features, cv, random_state
):
    """Return mean five-fold (or configured-fold) ROC AUC."""
    if not columns:
        return 0.5

    model = _build_classifier(
        columns, numeric_features, categorical_features, random_state
    )
    scores = cross_val_score(
        model,
        X[columns],
        y,
        cv=cv,
        scoring="roc_auc",
        error_score="raise",
    )
    return float(np.mean(scores))


# ---------------------------------------------------------------------
# 3. Evidence e1: single-feature predictiveness
# ---------------------------------------------------------------------

def calculate_e1(
    X, y, feature_columns, numeric_features, categorical_features,
    cv, random_state=42
):
    """
    For each feature, train a model using only that feature.

    ROC AUC 0.5 means no better than random.
    ROC AUC 1.0 means perfect ranking.
    AUC below 0.5 is adjusted as inverse predictiveness.
    The result is scaled to [0, 1].
    """
    scores = {}

    for feature in feature_columns:
        try:
            auc = _cross_validated_auc(
                X, y, [feature], numeric_features, categorical_features,
                cv, random_state,
            )
            adjusted_auc = max(auc, 1.0 - auc)
            scores[feature] = float(np.clip((adjusted_auc - 0.5) / 0.5, 0, 1))
        except Exception:
            # None means that the metric could not be calculated.
            scores[feature] = None

    return scores

# ---------------------------------------------------------------------
# 4. Evidence e2: shallow decision-tree importance share
# ---------------------------------------------------------------------

def calculate_e2(
    X, y, feature_columns, numeric_features, categorical_features,
    random_state=42
):
    """
    Fit a shallow decision tree and aggregate one-hot encoded importance
    back to the original feature. Return each feature's share of importance.
    """
    pipeline = Pipeline([
        (
            "preprocessing",
            _build_preprocessor(
                feature_columns, numeric_features, categorical_features
            ),
        ),
        (
            "tree",
            DecisionTreeClassifier(
                max_depth=3,
                min_samples_leaf=5,
                random_state=random_state,
            ),
        ),
    ])

    pipeline.fit(X[feature_columns], y)

    importances = pipeline.named_steps["tree"].feature_importances_
    transformed_names = (
        pipeline.named_steps["preprocessing"].get_feature_names_out()
    )

    importance_by_feature = {feature: 0.0 for feature in feature_columns}

    # Map each encoded column back to its original Spark feature.
    # Longest names are checked first to avoid prefix collisions.
    for encoded_name, importance in zip(transformed_names, importances):
        matched_feature = None

        for feature in sorted(feature_columns, key=len, reverse=True):
            if (
                encoded_name == f"numeric__{feature}"
                or encoded_name.startswith(f"categorical__{feature}_")
            ):
                matched_feature = feature
                break

        if matched_feature is not None:
            importance_by_feature[matched_feature] += float(importance)

    total_importance = sum(importance_by_feature.values())

    if total_importance == 0:
        return {feature: 0.0 for feature in feature_columns}

    return {
        feature: importance / total_importance
        for feature, importance in importance_by_feature.items()
    }


# ---------------------------------------------------------------------
# 5. Evidence e3: drop-column impact
# ---------------------------------------------------------------------

def calculate_e3(
    X, y, feature_columns, numeric_features, categorical_features,
    cv, random_state, top_features, full_model_auc
):
    """
    For selected suspicious features:
      1. Measure full-model AUC.
      2. Remove one feature and measure AUC again.
      3. Calculate lost lift relative to the full model's lift above 0.5.

    Features not selected for this expensive test get None (not tested).
    """
    scores = {feature: None for feature in feature_columns}
    full_lift = max(full_model_auc - 0.5, 0.0)

    for feature in top_features:
        remaining_features = [
            column for column in feature_columns if column != feature
        ]

        if not remaining_features:
            scores[feature] = 0.0
            continue

        try:
            reduced_auc = _cross_validated_auc(
                X, y, remaining_features, numeric_features,
                categorical_features, cv, random_state,
            )
            lost_lift = full_model_auc - reduced_auc

            if full_lift > 0:
                scores[feature] = float(
                    np.clip(lost_lift / full_lift, 0, 1)
                )
            else:
                scores[feature] = 0.0
        except Exception:
            scores[feature] = None

    return scores


# ---------------------------------------------------------------------
# 6. Evidence e4 and e5: metadata checks
# ---------------------------------------------------------------------

def calculate_e4(feature, feature_availability):
    """
    feature_availability values:
      True  -> confirmed available at prediction time -> e4 = 0
      False -> confirmed unavailable at prediction time -> e4 = 1
      missing key -> unknown -> e4 = None
    """
    if feature not in feature_availability:
        return None

    value = feature_availability[feature]
    if not isinstance(value, bool):
        raise ValueError(
            f"Availability for {feature!r} must be True, False, or omitted."
        )

    return 0.0 if value else 1.0


def calculate_e5(feature, lineage_derived, name_hints):
    """
    Return e5 = 1 if configured name hints or lineage metadata indicate
    derivation from the target's source; otherwise return 0.

    In production, populate lineage_derived from a trusted lineage source.
    """
    lineage_hint = lineage_derived.get(feature, False) is True
    name_hint = any(hint.lower() in feature.lower() for hint in name_hints)
    return 1.0 if (lineage_hint or name_hint) else 0.0


# ---------------------------------------------------------------------
# 7. Score and risk band
# ---------------------------------------------------------------------

def calculate_leakage_score(
    e1, e2, e3, e4, e5,
    e1_weight=DEFAULT_E1_WEIGHT,
    e2_weight=DEFAULT_E2_WEIGHT,
    e3_weight=DEFAULT_E3_WEIGHT,
):
    """
    LEAK-01 formula:
    max(e4, 0.6*e5, 0.4*e1 + 0.3*e2 + 0.3*e3)

    Unknown evidence is not proof of safety. For the numeric score, unknown
    evidence is omitted/treated as zero contribution; the evidence fields
    must still be reviewed.
    """
    statistical_score = (
        e1_weight * (e1 if e1 is not None else 0.0)
        + e2_weight * (e2 if e2 is not None else 0.0)
        + e3_weight * (e3 if e3 is not None else 0.0)
    )

    candidates = [0.6 * e5, statistical_score]

    if e4 is not None:
        candidates.append(e4)

    return float(max(candidates))


def assign_risk_band(score, warn_threshold=DEFAULT_WARN_THRESHOLD,
                     fail_threshold=DEFAULT_FAIL_THRESHOLD):
    if score >= fail_threshold:
        return "FAIL"
    if score >= warn_threshold:
        return "WARN"
    return "PASS"


# ---------------------------------------------------------------------
# 8. Waiver validation
# ---------------------------------------------------------------------

def validate_waiver(feature, waivers):
    """
    A waiver must include a reviewer/user and justification.
    The waiver does not lower the leakage score or risk band.
    """
    waiver = waivers.get(feature)

    if waiver is None:
        return {
            "waived": False,
            "waiver_user": None,
            "waiver_reason": None,
            "waiver_timestamp": None,
        }

    if not isinstance(waiver, dict):
        raise ValueError(f"Waiver for {feature!r} must be a dictionary.")

    user = str(waiver.get("user", "")).strip()
    reason = str(waiver.get("reason", "")).strip()

    if not user or not reason:
        raise ValueError(
            f"Waiver for {feature!r} requires non-empty 'user' and 'reason'."
        )

    return {
        "waived": True,
        "waiver_user": user,
        "waiver_reason": reason,
        "waiver_timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------
# 9. Main reusable function: orchestrates all evidence calculations
# ---------------------------------------------------------------------
def profile_target_leakage(
    df: DataFrame,
    target_column: str,
    feature_columns=None,
    prediction_time=None,
    feature_availability=None,
    lineage_derived=None,
    waivers=None,
    name_hints=None,
    sample_size=10_000,
    top_n_drop_column=10,
    cv_folds=5,
    random_state=42,
    warn_threshold=DEFAULT_WARN_THRESHOLD,
    fail_threshold=DEFAULT_FAIL_THRESHOLD,
):
    """
    Profile leakage risk for each feature in a PySpark DataFrame.

    Parameters
    ----------
    df:
        Input Spark DataFrame.
    target_column:
        Name of binary classification target (must be supplied by caller).
    feature_columns:
        Feature names. If omitted, all columns except target are selected.
        Exclude identifiers such as 'TrialBusinessId_ of Sigma' by default
        in your calling code unless their use is justified.
    prediction_time:
        Human-readable definition of prediction point.
    feature_availability:
        {feature_name: True/False}; True means available by prediction time.
        Omit keys where availability is unknown.
    lineage_derived:
        {feature_name: True/False}; True means confirmed target-source lineage.
    waivers:
        {feature_name: {'user': 'reviewer', 'reason': 'justification'}}.
    name_hints:
        Name fragments that indicate target-source-derived fields.
    sample_size:
        Maximum sample rows collected to pandas.
    top_n_drop_column:
        Number of suspicious features tested with the expensive e3 check.
    cv_folds:
        Number of stratified CV folds (default 5).

    Returns
    -------
    dict containing dataset-level metadata and a sorted list of feature metrics.
    """
    feature_availability = feature_availability or {}
    lineage_derived = lineage_derived or {}
    waivers = waivers or {}
    name_hints = name_hints or []

    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found.")

    if feature_columns is None:
        feature_columns = [c for c in df.columns if c != target_column]

    feature_columns = list(dict.fromkeys(feature_columns))

    if target_column in feature_columns:
        raise ValueError("Target column cannot also be a feature.")

    missing_columns = set(feature_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Feature columns not found: {sorted(missing_columns)}")

    if not feature_columns:
        raise ValueError("At least one feature column is required.")

    if sample_size < 1 or cv_folds < 2 or top_n_drop_column < 0:
        raise ValueError("Invalid sample_size, cv_folds, or top_n_drop_column.")

    if not (0 <= warn_threshold < fail_threshold <= 1):
        raise ValueError("Thresholds must satisfy 0 <= warn < fail <= 1.")

    # Sample on Spark, then collect only the bounded sample to the driver.
    sample_pdf = (
        df.select(*(feature_columns + [target_column]))
        .where(F.col(target_column).isNotNull())
        .orderBy(F.rand(random_state))
        .limit(sample_size)
        .toPandas()
    )

    if sample_pdf.empty:
        raise ValueError("No rows with a non-null target were found.")

    # Do not drop duplicate rows: duplicates may be legitimate observations.
    y = sample_pdf[target_column]

    if y.nunique() != 2:
        raise ValueError(
            "This version requires a binary target with exactly two classes "
            "in the sample."
        )

    if y.value_counts().min() < cv_folds:
        raise ValueError(
            f"Each target class needs at least {cv_folds} rows in the sample."
        )

    X, usable_features, numeric_features, categorical_features = (
        _prepare_features(sample_pdf, feature_columns)
    )

    if not usable_features:
        raise ValueError("No varying, non-null features remain in the sample.")

    cv = StratifiedKFold(
        n_splits=cv_folds, shuffle=True, random_state=random_state
    )

    # Calculate e1 and e2 for all usable features.
    e1_scores = calculate_e1(
        X, y, usable_features, numeric_features, categorical_features,
        cv, random_state,
    )

    e2_scores = calculate_e2(
        X, y, usable_features, numeric_features, categorical_features,
        random_state,
    )

    # Full model score is needed as the reference for e3.
    full_model_auc = _cross_validated_auc(
        X, y, usable_features, numeric_features, categorical_features,
        cv, random_state,
    )

    # Choose the top suspicious features for drop-column testing.
    ranked_features = sorted(
        usable_features,
        key=lambda feature: (
            e1_scores[feature] if e1_scores[feature] is not None else -1,
            e2_scores.get(feature, 0.0),
        ),
        reverse=True,
    )
    selected_for_e3 = ranked_features[:top_n_drop_column]

    e3_scores = calculate_e3(
        X, y, usable_features, numeric_features, categorical_features,
        cv, random_state, selected_for_e3, full_model_auc,
    )

    # Combine evidence and create one output record per feature.
    feature_results = []

    for feature in usable_features:
        e1 = e1_scores.get(feature)
        e2 = e2_scores.get(feature)
        e3 = e3_scores.get(feature)
        e3_tested = feature in selected_for_e3

        e4 = calculate_e4(feature, feature_availability)
        e5 = calculate_e5(feature, lineage_derived, name_hints)

        score = calculate_leakage_score(e1, e2, e3, e4, e5)
        risk_band = assign_risk_band(
            score, warn_threshold, fail_threshold
        )

        evidence = []

        if e1 is not None and e1 >= 0.5:
            evidence.append("Strong individual predictive signal.")
        if e2 is not None and e2 > 0.5:
            evidence.append("Feature accounts for over 50% of tree importance.")
        if e3 is not None and e3 >= 0.5:
            evidence.append("Removing feature substantially reduces model lift.")
        if e3 is None:
            evidence.append("Drop-column impact was not measured successfully.")
        if e4 == 1.0:
            evidence.append("Feature is confirmed unavailable at prediction time.")
        elif e4 is None:
            evidence.append("Feature availability at prediction time is unknown.")
        if any(hint.lower() in feature.lower() for hint in name_hints):
            evidence.append("Feature name matches a configured target-source hint.")
        if lineage_derived.get(feature, False) is True:
            evidence.append("Lineage metadata indicates target-source derivation.")
        if not evidence:
            evidence.append("No configured strong leakage evidence was detected.")

        waiver_info = validate_waiver(feature, waivers)

        feature_results.append({
            "feature": feature,
            "leakage_score": round(score, 4),
            "e1": None if e1 is None else round(e1, 4),
            "e2": None if e2 is None else round(e2, 4),
            "e3": None if e3 is None else round(e3, 4),
            "e3_tested": e3_tested,
            "e4": e4,
            "e5": e5,
            "evidence": evidence,
            "risk_band": risk_band,
            **waiver_info,
        })

    feature_results.sort(
        key=lambda item: item["leakage_score"], reverse=True
    )

    return {
        "metric_id": "LEAK-01",
        "metric": "Target leakage score",
        "target_column": target_column,
        "prediction_time": prediction_time,
        "sample_rows": len(sample_pdf),
        "full_model_cv_auc": round(full_model_auc, 4),
        "features_evaluated": len(feature_results),
        "features": feature_results,
        "notes": [
            "Binary classification only.",
            "e3 is measured only for the top configured suspicious features.",
            "e3=None means not measured or calculation failed; inspect e3_tested.",
            "Unknown e4 means availability is unverified, not safe.",
            "A waiver is recorded but does not change score or risk band.",
            "Review identifier and time-window features using domain knowledge.",
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
    # df.printSchema()   
    # df.show(5,False)

    target_column = "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22"
    identifier_columns = ["TrialBusinessId_ of Sigma"]
    feature_columns = [
        column for column in df.columns
        if column != target_column and column not in identifier_columns
    ]

    waivers = {
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
    }
    feature_availability = {
        "Mazzoni 1- Sigma Mixer current_mean": True,
        "Mazzoni 1- Sigma Mixer current_sum": True,
        "Mazzoni 1- Sigma Plodder pressure_mean10to22": True,
        "Mazzoni 1- Sigma Plodder pressure_std10to22": True,
        "Mazzoni 1- Sigma Plodder pressure_after_production": False
    }
    lineage_derived = {
        "Mazzoni 1- Sigma Mixer current_mean": False,
        "Mazzoni 1- Sigma Mixer current_sum": False,
        "Mazzoni 1- Sigma Plodder pressure_mean10to22": False,
        "product_quality_result": True
    }
    name_hints = {
        "Mazzoni 1- Sigma Mixer current_mean": False,
        "Mazzoni 1- Sigma Mixer current_sum": False,
        "final_quality_result": True,
        "target_quality_flag": True
    }
    result = profile_target_leakage(
        df=df,
        target_column=target_column,
        feature_columns=feature_columns,
        prediction_time="After the 10to22 measurement window ends",
        feature_availability=feature_availability,
        lineage_derived=lineage_derived,
        name_hints=name_hints,
        waivers=waivers,
        sample_size=10_000,
        top_n_drop_column=10,
        cv_folds=5,
    )
    print(result)
    for item in result["features"]:
        print(
            item["feature"],
            item["leakage_score"],
            item["risk_band"],
            item["evidence"],
        )

if __name__ == "__main__":
    main()  