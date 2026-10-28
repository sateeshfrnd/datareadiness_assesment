
import json
import numpy as np
import pandas as pd

from pyspark.sql import functions as F
from pyspark.sql.types import (
    NumericType, DoubleType, FloatType
)
from scipy.linalg import qr

def assess_vif(
    df,
    feature_cols=None,
    sample_fraction=1.0,
    sample_limit=100_000,
    warn_threshold=5.0,
    fail_threshold=10.0,
    signal_scores=None,
    eliminate=False,
    random_seed=42,
    tolerance=1e-10,
):
    """
    RED-01: Variance Inflation Factor assessment.

    Parameters
    ----------
    df : pyspark.sql.DataFrame
        Input DataFrame.
    feature_cols : list[str] or None
        Numeric feature columns. None selects all numeric columns.
    sample_fraction : float
        Fraction of rows to sample (0, 1].
    sample_limit : int or None
        Maximum number of rows used in the matrix calculation.
    signal_scores : dict[str, float] or None
        Optional SIG-04 score for each feature.
    eliminate : bool
        Whether to suggest iterative feature elimination.
    """

    signal_scores = signal_scores or {}

    # 1. Select numeric feature columns
    schema_types = {
        field.name: field.dataType
        for field in df.schema.fields
    }

    if feature_cols is None:
        feature_cols = [
            name for name, dtype in schema_types.items()
            if isinstance(dtype, NumericType)
            and not isinstance(dtype, (FloatType, DoubleType))
                or (
                    isinstance(dtype, (FloatType, DoubleType))
                    and name in df.columns
                )
        ]

    # Explicitly exclude non-numeric columns even if supplied
    feature_cols = [
        c for c in feature_cols
        if c in schema_types
        and isinstance(schema_types[c], NumericType)
        and not isinstance(schema_types[c], type(None))
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns were found.")

    # 2. Sample rows for the assessment
    sample_df = df.select(*feature_cols)

    if sample_fraction < 1.0:
        sample_df = sample_df.sample(
            withReplacement=False,
            fraction=sample_fraction,
            seed=random_seed
        )

    if sample_limit is not None:
        sample_df = sample_df.limit(sample_limit)

    # 3. Convert the bounded sample to Pandas
    pdf = sample_df.toPandas()

    if len(pdf) < 3:
        raise ValueError(
            "At least 3 sample rows are recommended for VIF."
        )

    # Ensure invalid numeric values are treated as missing
    pdf = pdf.apply(pd.to_numeric, errors="coerce")
    pdf = pdf.replace([np.inf, -np.inf], np.nan)

    # 4. Remove all-null and constant columns
    removed_features = []
    usable = []

    for col in feature_cols:
        series = pdf[col]
        non_null = series.dropna()

        if non_null.empty or non_null.nunique() <= 1:
            removed_features.append(col)
        else:
            usable.append(col)

    pdf = pdf[usable].copy()

    if len(usable) < 2:
        raise ValueError(
            "VIF requires at least two non-constant numeric features."
        )

    # 5. Median imputation
    medians = pdf.median()
    pdf = pdf.fillna(medians)

    # 6. Standardization
    means = pdf.mean()
    stds = pdf.std(ddof=0)

    # A second constant check protects against numeric edge cases
    valid = stds[stds > tolerance].index.tolist()
    removed_features.extend(
        [c for c in pdf.columns if c not in valid]
    )

    pdf = pdf[valid]
    means = pdf.mean()
    stds = pdf.std(ddof=0)

    standardized = (pdf - means) / stds

    if standardized.shape[1] < 2:
        raise ValueError(
            "Fewer than two usable features remain after preprocessing."
        )

    # 7. Pearson correlation matrix R
    R = standardized.corr(method="pearson").to_numpy()
    names = standardized.columns.tolist()

    # 8. Pseudoinverse and rank check
    R_inv = np.linalg.pinv(R, rcond=tolerance)
    rank = np.linalg.matrix_rank(R, tol=tolerance)
    singular = rank < len(names)

    # VIF_j = diagonal element of inverse correlation matrix
    vif_values = np.diag(R_inv).copy()

    # 9. Detect dependent columns using pivoted QR
    # The dependent columns are relative to the selected independent
    # basis; they are not necessarily the only possible dependencies.
    _, R_qr, pivots = qr(
        standardized.to_numpy(),
        mode="economic",
        pivoting=True
    )

    diag = np.abs(np.diag(R_qr))
    qr_tol = (
        tolerance * max(standardized.shape)
        * diag.max()
        if len(diag) and diag.max() > 0
        else tolerance
    )
    qr_rank = int(np.sum(diag > qr_tol))
    dependent_idx = list(pivots[qr_rank:])
    dependent_features = [names[i] for i in dependent_idx]

    # 10. Report results
    rows = []

    for i, col in enumerate(names):
        vif = float(vif_values[i])

        # A singular matrix means the pseudoinverse diagonal alone
        # must not be interpreted as a conventional finite VIF.
        if singular and col in dependent_features:
            vif = float("inf")

        if np.isinf(vif) or vif > fail_threshold:
            status = "FAIL"
        elif vif > warn_threshold:
            status = "WARN"
        else:
            status = "PASS"

        r2 = (
            1.0 - (1.0 / vif)
            if np.isfinite(vif) and vif > 0
            else 1.0
        )
        r2 = min(1.0, max(0.0, r2))

        rows.append({
            "feature": col,
            "vif": vif,
            "r2_on_others": r2,
            "status": status,
            "signal_score": signal_scores.get(col)
        })

    result = pd.DataFrame(rows)

    # 11. Optional iterative elimination
    suggested_drop_list = []
    retained = names.copy()

    if eliminate:
        while len(retained) > 1:
            current = standardized[retained].to_numpy()
            current_R = np.corrcoef(current, rowvar=False)
            current_R = np.atleast_2d(current_R)

            current_rank = np.linalg.matrix_rank(
                current_R, tol=tolerance
            )
            current_inv = np.linalg.pinv(
                current_R, rcond=tolerance
            )
            current_vif = np.diag(current_inv)

            # If singular, identify a dependent feature to remove.
            if current_rank < len(retained):
                _, R_step, piv_step = qr(
                    current, mode="economic", pivoting=True
                )
                d = np.abs(np.diag(R_step))
                qtol = (
                    tolerance * max(current.shape) * d.max()
                    if len(d) and d.max() > 0 else tolerance
                )
                r = int(np.sum(d > qtol))
                candidates = [retained[i] for i in piv_step[r:]]

                if not candidates:
                    candidates = retained

                # Prefer dropping the feature with the lower signal score.
                drop = min(
                    candidates,
                    key=lambda c: (
                        signal_scores.get(c, float("inf"))
                        if c in signal_scores
                        else float("inf"),
                        -current_vif[retained.index(c)]
                    )
                )
            else:
                max_vif = float(np.max(current_vif))

                if max_vif <= warn_threshold:
                    break

                # Highest VIF first; lower signal score breaks ties.
                max_candidates = [
                    retained[i]
                    for i, v in enumerate(current_vif)
                    if np.isclose(v, max_vif)
                ]
                drop = min(
                    max_candidates,
                    key=lambda c: signal_scores.get(
                        c, float("inf")
                    )
                )

            suggested_drop_list.append(drop)
            retained.remove(drop)

    return {
        "per_feature": result.sort_values(
            "vif", ascending=False
        ).reset_index(drop=True),
        "correlation_matrix": pd.DataFrame(
            R, index=names, columns=names
        ),
        "removed_constant_features": removed_features,
        "singular": singular,
        "matrix_rank": rank,
        "feature_count": len(names),
        "singular_sets": (
            [dependent_features] if dependent_features else []
        ),
        "suggested_drop_list": suggested_drop_list,
        "suggested_retained_features": retained
        if eliminate else names,
        "sample_row_count": len(pdf),
    }

import numpy as np
import pandas as pd

def make_json_serializable(obj):
    """Convert NumPy/Pandas values into JSON-compatible Python values."""

    if isinstance(obj, dict):
        return {
            str(k): make_json_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        if not np.isfinite(obj):
            return None
        return float(obj)

    if isinstance(obj, np.ndarray):
        return make_json_serializable(obj.tolist())

    if obj is None or isinstance(obj, (str, bool, int, float)):
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    if pd.isna(obj):
        return None

    return obj

def get_red_01_assesment_metrics(df, pred05_result, exclude_cols,sig04_result=None):

    if sig04_result is None:
        signal_scores = None
    else:
        signal_scores = {
            item["feature"]: item["signal_score"]
            for item in sig04_result["feature_metrics"]
            if item.get("signal_score") is not None
        }
    # Select numeric features that exist in the original DataFrame
    schema_types = {
        field.name: field.dataType
        for field in df.schema.fields
    }

    vif_features = [
        col_name
        for col_name in pred05_result
        if col_name in schema_types
        and isinstance(schema_types[col_name], NumericType)
        and col_name not in exclude_cols
    ]

    if len(vif_features) < 2:
        raise ValueError(
            "RED-01 requires at least two eligible numeric features."
        )

    # Pass only PRED-05-selected features to RED-01
    vif_df = df.select(*vif_features)
    result = assess_vif(
        df=vif_df,
        feature_cols=vif_features,
        sample_fraction=1.0,
        sample_limit=100_000,
        warn_threshold=5.0,
        fail_threshold=10.0,
        signal_scores=signal_scores,
        eliminate=False
    )
    print(result)
    return make_json_serializable(
        {
            "per_feature_dict" : result["per_feature"].to_dict(orient="records"),
            "correlation matrix:" : result["correlation_matrix"].to_dict(),
            "removed_constant_features" : result["removed_constant_features"],
            "singular" : result["singular"],
            "matrix_rank" : result["matrix_rank"],
            "feature_count" : result["feature_count"],
            "singular_sets" : result["singular_sets"],
            "suggested_drop_list" : result["suggested_drop_list"],
            "suggested_retained_features" : result["suggested_retained_features"],
            "sample_row_count" : result["sample_row_count"],
        }
    )

     # per_feature_dict = result["per_feature"].to_dict(orient="records")
    # print(json.dumps(per_feature_dict, indent=4))
    
    # print("correlation matrix:", result["correlation_matrix"])
    # # print("correlation matrix:", result["correlation_matrix"].to_dict())
    # print(json.dumps(result["correlation_matrix"].to_dict(), indent=4))

    # print("Singular matrix:", result["singular"])
    # print("Potential dependent sets:", result["singular_sets"])
    # print("Constant features removed:",
    #     result["removed_constant_features"])
    # print(f'suggested_drop_list = {result["suggested_drop_list"]}')
    # print(f'suggested_retained_features = {result["suggested_retained_features"]}')
    # print(f'sample_row_count = {result["sample_row_count"]}')




def main():
    # path = "abfss://data-readiness@bnlwestgunileverbd00036.dfs.core.windows.net/Source/data_to_model_20261014.csv"  
    path = '/Workspace/Users/sateeshfrnd@gmail.com/SAMBA/datareadiness_assesment_p/sample_datasets/data_to_model.csv'    
    df = spark.read.csv(
        path,
        header=True,
        inferSchema=True
    ) 
    # df.printSchema()   
    # df.show(5,False)

    # PRED-05 output: eligible feature names
    pred05_result = [
        "C12",
        "C14",
        "C16",
        "Mean_Iodine_value",
        "LongChain",
        "LongChainSat"
    ]

    # Exclude identifiers, even if mistakenly returned by PRED-05
    exclude_cols = {"_c0", "UID", "RootRecipeCode"}

    # numeric_cols = [
    #     field.name
    #     for field in df.schema.fields
    #     if isinstance(field.dataType, NumericType)
    #     and field.name not in exclude_cols
    # ]

    # print("Numeric feature count:", len(numeric_cols))
    # print("Numeric features:", numeric_cols)

    # result = assess_vif(
    #     df=df,
    #     feature_cols=numeric_cols,
    #     sample_fraction=1.0,
    #     sample_limit=100_000,
    #     warn_threshold=5.0,
    #     fail_threshold=10.0,
    #     eliminate=False
    # )
    result = get_red_01_assesment_metrics(df, pred05_result, exclude_cols)
    print("=======================================")
    print(result)
    print("=======================================")
    # # display(result["per_feature"])
    # per_feature_dict = result["per_feature"].to_dict(orient="records")
    # print(json.dumps(per_feature_dict, indent=4))
    
    # print("correlation matrix:", result["correlation_matrix"])
    # # print("correlation matrix:", result["correlation_matrix"].to_dict())
    # print(json.dumps(result["correlation_matrix"].to_dict(), indent=4))

    # print("Singular matrix:", result["singular"])
    # print("Potential dependent sets:", result["singular_sets"])
    # print("Constant features removed:",
    #     result["removed_constant_features"])
    # print(f'suggested_drop_list = {result["suggested_drop_list"]}')
    # print(f'suggested_retained_features = {result["suggested_retained_features"]}')
    # print(f'sample_row_count = {result["sample_row_count"]}')

    print(json.dumps(result, indent=4))



if __name__ == "__main__":
    main()    