
"""
A single, self-contained PySpark module that:

SIG-04 : Single composite score per feature combining MI, ANOVA / association and rank correlation on a common absolute [0, 1] scale, with ranking and a noise-floor comparison.
"""

import json

DEFAULT_WEIGHTS = {
    "s_mi": 0.4,
    "s_f": 0.3,
    "s_rho": 0.3,
}


def calculate_signal_score(
    feature: str,
    s_mi: float = None,
    s_f: float = None,
    s_rho: float = None,
    is_shadow: bool = False,
    other_features: list = None,
    weights: dict = None,
):
    """
    Calculate SIG-04 feature-level and step-level metrics.

    other_features:
        List of dictionaries containing the scores for other
        actual and shadow features.

    Returns:
        Dictionary containing feature_metrics and step_metrics.
    """

    weights = weights or DEFAULT_WEIGHTS

    score_columns = ["s_mi", "s_f", "s_rho"]

    if set(weights) != set(score_columns):
        raise ValueError(
            f"Weights must contain exactly {score_columns}"
        )

    if any(w < 0 for w in weights.values()):
        raise ValueError("Weights cannot be negative")

    if sum(weights.values()) <= 0:
        raise ValueError("At least one weight must be positive")

    # Include the current feature and the other features.
    records = [
        {
            "feature": feature,
            "s_mi": s_mi,
            "s_f": s_f,
            "s_rho": s_rho,
            "is_shadow": is_shadow,
        }
    ]
    records.extend(other_features or [])

    # Calculate a composite score for each feature.
    for record in records:
        numerator = 0.0
        denominator = 0.0

        for col in score_columns:
            value = record.get(col)
            weight = weights[col]

            if value is None:
                continue

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{col} must be between 0 and 1"
                )

            if weight > 0:
                numerator += value * weight
                denominator += weight

        score = (
            numerator / denominator
            if denominator > 0
            else None
        )

        record["signal_score"] = score

        if score is None:
            record["band"] = "unscored"
        elif score >= 0.30:
            record["band"] = "strong"
        elif score >= 0.10:
            record["band"] = "moderate"
        else:
            record["band"] = "weak"

    # Separate actual features and shadow features.
    actual = [
        r for r in records
        if not r["is_shadow"]
    ]
    shadows = [
        r for r in records
        if r["is_shadow"]
    ]

    valid_shadow_scores = [
        r["signal_score"]
        for r in shadows
        if r["signal_score"] is not None
    ]

    noise_floor = (
        max(valid_shadow_scores)
        if valid_shadow_scores
        else None
    )

    # Dense rank, descending. Tied scores share the same rank.
    valid_scores = sorted(
        {
            r["signal_score"]
            for r in actual
            if r["signal_score"] is not None
        },
        reverse=True,
    )

    rank_lookup = {
        score: rank
        for rank, score in enumerate(valid_scores, start=1)
    }

    n_valid = len(valid_scores)

    feature_metrics = []

    for record in actual:
        score = record["signal_score"]

        rank = (
            rank_lookup.get(score)
            if score is not None
            else None
        )

        # Best = 100, worst = 0.
        percentile = (
            100.0
            if n_valid == 1 and score is not None
            else (
                (n_valid - rank) / (n_valid - 1) * 100.0
                if score is not None and n_valid > 1
                else None
            )
        )

        above_noise = (
            score > noise_floor
            if score is not None and noise_floor is not None
            else None
        )

        feature_metrics.append({
            "feature": record["feature"],
            "signal_score": score,
            "rank": rank,
            "percentile": percentile,
            "band": record["band"],
            "above_noise": above_noise,
        })

    # Step-level metrics.
    valid_actual = [
        r for r in actual
        if r["signal_score"] is not None
    ]

    max_signal_score = (
        max(r["signal_score"] for r in valid_actual)
        if valid_actual
        else None
    )

    n_strong = sum(
        r["band"] == "strong" for r in valid_actual
    )

    n_moderate = sum(
        r["band"] == "moderate" for r in valid_actual
    )

    n_above_noise = sum(
        r["signal_score"] > noise_floor
        for r in valid_actual
        if noise_floor is not None
    )

    # Status based on the SIG-04 thresholds.
    if max_signal_score is None:
        status = "FAIL"
    elif max_signal_score < 0.10:
        status = "FAIL"
    elif max_signal_score < 0.30:
        status = "WARN"
    elif n_above_noise == 0:
        status = "WARN"
    else:
        status = "PASS"

    return {
        "metric_id": "SIG-04",
        "metric_name": "Normalised signal score and rank",
        "feature_metrics": feature_metrics,
        "step_metrics": {
            "max_signal_score": max_signal_score,
            "n_strong": n_strong,
            "n_moderate": n_moderate,
            "n_above_noise": n_above_noise,
            "noise_floor": noise_floor,
            "status": status,
        },
    }


def main():
    sig_04_metrics = calculate_signal_score(
        feature="Mazzoni 1- Sigma Mixer current_mean",
        s_mi=0.80,
        s_f=0.60,
        s_rho=0.70,
        is_shadow=False,
        other_features=[
            {
                "feature": "Mazzoni 1- Sigma Mixer current_median",
                "s_mi": 0.40,
                "s_f": 0.50,
                "s_rho": 0.30,
                "is_shadow": False,
            },
            {
                "feature": "Mazzoni 1- Sigma Mixer current_sum",
                "s_mi": 0.80,
                "s_f": 0.60,
                "s_rho": 0.70,
                "is_shadow": False,
            },
        ],
    )    
    print(json.dumps(sig_04_metrics, indent=4))


if __name__ == "__main__":
    main()    
