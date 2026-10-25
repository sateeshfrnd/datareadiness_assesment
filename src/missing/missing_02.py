import json
import numpy as np
import pandas as pd

from scipy.stats import chi2, mannwhitneyu, chi2_contingency
# from statsmodels.stats.multitest import multipletests

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

from pyspark.sql import functions as F
from pyspark.sql.types import (
    NumericType, DoubleType, FloatType
)


MAX_SAMPLE_ROWS = 50_000
MISSING_THRESHOLD_PCT = 1.0
MCAR_ALPHA = 0.05
MAR_AUC_THRESHOLD = 0.65
TARGET_ASSOC_ALPHA = 0.01

# Identifier should not be used as a model predictor.
ID_COLUMNS = ["TrialBusinessId_ of Sigma"]

def prepare_miss02_data(df, max_rows=MAX_SAMPLE_ROWS):
    total_rows = df.count()

    if total_rows == 0:
        raise ValueError("Input DataFrame contains no rows.")

    numeric_columns = [
        field.name
        for field in df.schema.fields
        if isinstance(field.dataType, NumericType)
    ]

    # Handle NaN as missing for floating-point features.
    missing_exprs = []
    for c in df.columns:
        dtype = df.schema[c].dataType

        is_missing = F.col(c).isNull()

        if isinstance(dtype, (DoubleType, FloatType)):
            is_missing = is_missing | F.isnan(F.col(c))

        missing_exprs.append(
            F.sum(F.when(is_missing, 1).otherwise(0)).alias(c)
        )

    counts = df.agg(*missing_exprs).first().asDict()

    profile = pd.DataFrame([
        {
            "feature": c,
            "missing_k": int(counts[c] or 0),
            "missing_pct": 100.0 * (counts[c] or 0) / total_rows
        }
        for c in df.columns
    ])

    eligible = profile.loc[
        profile["missing_pct"] >= MISSING_THRESHOLD_PCT,
        "feature"
    ].tolist()

    # Only sample if necessary. Prefer an existing MISS-01
    # stratified sample if your pipeline already produces one.
    if total_rows > max_rows:
        sampled_df = df.orderBy(F.rand(seed=42)).limit(max_rows)
    else:
        sampled_df = df

    pdf = sampled_df.select(*df.columns).toPandas()

    # Normalize NaN and infinite numeric values.
    for c in numeric_columns:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
        pdf[c] = pdf[c].replace([np.inf, -np.inf], np.nan)

    return {
        "pdf": pdf,
        "profile": profile,
        "numeric_columns": numeric_columns,
        "eligible_columns": eligible,
        "total_rows": total_rows,
        "sample_rows": len(pdf)
    }

def littles_mcar_test(pdf, numeric_columns, max_iter=200, tol=1e-6):
    # Little's test: numeric features with at least one missing value.
    cols = [
        c for c in numeric_columns
        if c in pdf.columns and pdf[c].isna().any()
    ]

    x = pdf[cols].replace([np.inf, -np.inf], np.nan)
    x = x.dropna(axis=1, how="all")
    x = x.loc[:, x.nunique(dropna=True) > 1]

    if x.shape[1] < 2:
        return {
            "little_chi2": None,
            "little_df": None,
            "little_pvalue": None,
            "global_mechanism": "NA",
            "reason": "Fewer than two usable numeric features."
        }

    if not x.isna().any().any():
        return {
            "little_chi2": None,
            "little_df": None,
            "little_pvalue": None,
            "global_mechanism": "NA",
            "reason": "No missing values in the tested features."
        }

    a = x.to_numpy(dtype=float)
    observed = np.isfinite(a)
    n, p = a.shape

    # Remove rows with no observed values in the tested features.
    keep = observed.any(axis=1)
    a = a[keep]
    observed = observed[keep]
    n = len(a)

    if n <= p:
        return {
            "little_chi2": None,
            "little_df": None,
            "little_pvalue": None,
            "global_mechanism": "NA",
            "reason": "Insufficient rows relative to feature count."
        }

    col_means = np.nanmean(a, axis=0)
    filled = np.where(observed, a, col_means)
    mu = filled.mean(axis=0)
    cov = np.atleast_2d(np.cov(filled, rowvar=False))
    cov += np.eye(p) * 1e-6

    # EM-style estimation for the multivariate normal mean/covariance.
    converged = False

    for _ in range(max_iter):
        sum_x = np.zeros(p)
        sum_xx = np.zeros((p, p))

        for row, mask in zip(a, observed):
            o = np.where(mask)[0]
            m = np.where(~mask)[0]

            ex = mu.copy()
            ex[o] = row[o]
            exx = np.outer(ex, ex)

            if len(m) > 0 and len(o) > 0:
                soo = cov[np.ix_(o, o)]
                smo = cov[np.ix_(m, o)]
                som = cov[np.ix_(o, m)]
                smm = cov[np.ix_(m, m)]

                inv = np.linalg.pinv(soo)
                ex[m] = (
                    mu[m] + smo @ inv @ (row[o] - mu[o])
                )

                exx = np.outer(ex, ex)
                conditional_cov = smm - smo @ inv @ som
                exx[np.ix_(m, m)] += conditional_cov

            elif len(o) == 0:
                exx = cov + np.outer(mu, mu)

            sum_x += ex
            sum_xx += exx

        new_mu = sum_x / n
        new_cov = sum_xx / n - np.outer(new_mu, new_mu)
        new_cov = (new_cov + new_cov.T) / 2
        new_cov += np.eye(p) * 1e-8

        change = max(
            np.max(np.abs(new_mu - mu)),
            np.max(np.abs(new_cov - cov))
        )

        mu, cov = new_mu, new_cov

        if change < tol:
            converged = True
            break

    # Group rows by distinct observed/missing pattern.
    patterns = {}
    for i, mask in enumerate(observed):
        key = tuple(mask.tolist())
        patterns.setdefault(key, []).append(i)

    statistic = 0.0
    df = -p

    for pattern, indices in patterns.items():
        o = np.where(pattern)[0]
        p_j = len(o)

        # Degrees of freedom count distinct patterns, not rows.
        df += p_j

        if p_j == 0:
            continue

        group = a[np.ix_(indices, o)]
        group_mean = group.mean(axis=0)
        delta = group_mean - mu[o]
        cov_o = cov[np.ix_(o, o)]

        statistic += len(indices) * float(
            delta @ np.linalg.pinv(cov_o) @ delta
        )

    if df <= 0 or not np.isfinite(statistic):
        return {
            "little_chi2": float(statistic),
            "little_df": int(df),
            "little_pvalue": None,
            "global_mechanism": "NA",
            "reason": "Non-positive degrees of freedom or invalid statistic."
        }

    pvalue = float(chi2.sf(statistic, df))

    return {
        "little_chi2": float(statistic),
        "little_df": int(df),
        "little_pvalue": pvalue,
        "global_mechanism": (
            "MCAR-consistent" if pvalue >= MCAR_ALPHA
            else "Not MCAR-consistent"
        ),
        "em_converged": converged,
        "patterns_tested": len(patterns),
        "features_tested": x.columns.tolist()
    }


def run_miss02(
    df,
    miss01_result,
    target_columns
):
    prepared = prepare_miss02_data(df)
    pdf = prepared["pdf"]
    profile = prepared["profile"]
    numeric_columns = prepared["numeric_columns"]
    eligible_columns = prepared["eligible_columns"]

    display(profile.sort_values("missing_pct", ascending=False))

    print("Rows:", prepared["total_rows"])
    print("Sample rows:", prepared["sample_rows"])
    print("Eligible features:", len(eligible_columns))

    # Use MISS-01 output to identify eligible features.
    if miss01_result != None:
        miss01_columns = miss01_result["column_missingness"]

        eligible_columns = [
            item["feature"]
            for item in miss01_columns
            if item["missing_pct"] >= 1.0
        ]
    little = littles_mcar_test(pdf, numeric_columns)
    print(little)

    per_column = []
    for feature in eligible_columns:
        associations = test_missingness_associations(pdf, feature)

        auc_result = predict_missingness_auc(pdf, feature)

        target_results = {
            t: next(
                (
                    a for a in associations
                    if a["feature"] == t
                ),
                None
            )
            for t in target_columns
            if t in pdf.columns and t != feature
        }

        target_assoc_p = {
            t: (r["pvalue"] if r else None)
            for t, r in target_results.items()
        }

        associated_vars = [
            a["feature"] for a in associations
            if a.get("associated", False)
        ][:10]

        significant_target = any(
            p is not None and p < TARGET_ASSOC_ALPHA
            for p in target_assoc_p.values()
        )

        auc = auc_result["auc"]

        if significant_target:
            label = "Informative missingness; investigate possible MNAR"
        elif auc is not None and auc >= MAR_AUC_THRESHOLD:
            label = "MAR evidence"
        elif (
            little["global_mechanism"] == "MCAR-consistent"
            and not associated_vars
        ):
            label = "MCAR-consistent; evidence is limited"
        elif associated_vars:
            label = "Possible MAR; associated variables detected"
        else:
            label = "Inconclusive"

        per_column.append({
            "feature": feature,
            "missing_pct": float(
                prepared["profile"].loc[
                    prepared["profile"]["feature"] == feature,
                    "missing_pct"
                ].iloc[0]
            ),
            "mechanism_label": label,
            "missing_model_auc": auc,
            "auc_status": auc_result["status"],
            "top_associated_vars": associated_vars,
            "target_assoc_p": target_assoc_p,
            "associations": associations
        })

    return {
        "metric_id": "MISS-02",
        "metric_name": "MCAR vs MAR test",
        "sample_rows": prepared["sample_rows"],
        **little,
        "per_column": per_column,
        "status": "NA" if not eligible_columns else "COMPLETED"
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
    missing_metrics = MISSING_METRICS
    miss01_result = MISSING_METRICS["missing_metrics"]["miss-01"] #["column_missingness"]
    # print(miss01_result)

    # Your four observed Sigma Mixer measurements
    TARGET_COLUMNS = [
        "Mazzoni 1- Sigma Mixer current_mean",
        "Mazzoni 1- Sigma Mixer current_sum",
        "Mazzoni 1- Sigma Mixer current_median",
        "Mazzoni 1- Sigma Mixer current_std",
    ]
    miss02_result = run_miss02(
        df,
        miss01_result,
        target_columns=TARGET_COLUMNS
    )

    print("Global result:")
    print({
        k: v for k, v in miss02_result.items()
        if k != "per_column"
    })

    print("\nPer-column results:")
    # display(pd.DataFrame(miss02_result["per_column"]))
    print(miss02_result)
    print(json.dumps(miss02_result, indent=4))



MISSING_METRICS= {"missing_metrics": {
    "miss-01": {
      "metric_id": "MISS-01",
      "metric_name": "Missing rate per column",
      "n_rows": 148,
      "rows_with_any_missing": 6,
      "column_missingness": [
        {
          "feature": "TrialBusinessId_ of Sigma",
          "missing_k": 0,
          "missing_pct": 0.0
        },
        {
          "feature": "Mazzoni 1- Sigma Mixer current_mean",
          "missing_k": 0,
          "missing_pct": 0.0
        },
        {
          "feature": "Mazzoni 1- Sigma Mixer current_sum",
          "missing_k": 0,
          "missing_pct": 0.0
        },
        {
          "feature": "Mazzoni 1- Sigma Mixer current_median",
          "missing_k": 0,
          "missing_pct": 0.0
        },
        {
          "feature": "Mazzoni 1- Sigma Mixer current_std",
          "missing_k": 0,
          "missing_pct": 0.0
        },
        {
          "feature": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1 - Flowrate throughput_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "missing_k": 6,
          "missing_pct": 0.040541
        },
        {
          "feature": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1 - Flowrate throughput_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "missing_k": 6,
          "missing_pct": 0.040541
        },
        {
          "feature": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1 - Flowrate throughput_max10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "missing_k": 6,
          "missing_pct": 0.040541
        },
        {
          "feature": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        },
        {
          "feature": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "missing_k": 5,
          "missing_pct": 0.033784
        }
      ],
      "co_missing": [
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1 - Flowrate throughput_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1 - Flowrate throughput_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_max10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_max10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_max10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_max10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 1.0
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_mean10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_mean10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_std10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_std10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1 - Flowrate throughput_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_min10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled Water flow_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Chilled water SV on/off indication_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Cone Heater temp_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder I/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder O/L cooling water temp_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder Temperature at Cone Inlet_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder main motor Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder plate pressure_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final plodder worm RPM_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Level in Final Plodder hopper_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Plodder vaccum pump Start/stop Indication_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Vaccum in final plodder hopper_min10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Billet Temperature at Plodder exit_max10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1 - Flowrate throughput_max10to22",
          "b": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "b": "Mazzoni 1- Final plodder Chilled Water flow_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "b": "Mazzoni 1- Final plodder Chilled water SV on/off indication_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "b": "Mazzoni 1- Final plodder Cone Heater temp_max10to22",
          "corr": 0.9097
        },
        {
          "a": "Mazzoni 1- Final Plodder Worm motor current_max10to22",
          "b": "Mazzoni 1- Final plodder I/L cooling water temp_max10to22",
          "corr": 0.9097
        }
      ],
      "detected_sentinels": {}
    }
  }}
if __name__ == "__main__":
    main()   


