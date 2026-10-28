import datetime
import json

from pyspark.sql import functions as F
# --- Settings -------------------------------------------------------------

MISSING_THRESHOLD = 10
NEAR_CONSTANT_THRESHOLD = 95
HARD_STOP_THRESHOLD = 50
CO_MISSING_MIN_CORR = 0.5,
INCLUDE_ZERO_MISSING = True
# ===========================================================================
# Core computation
# ===========================================================================
def compute_missing(df, config, sentinels=None):
    """Compute per-column missingness, the row histogram and co-missing pairs."""
    if sentinels is None:
        sentinels = detect_sentinels(df)

# ===========================================================================
# Orchestration
# ===========================================================================
def run_missingness_profile(df):
    """Run the whole missingness profile and return the final metrics DataFrame."""
    result = compute_missing(df)



def main():
    # path = "abfss://data-readiness@bnlwestgunileverbd00036.dfs.core.windows.net/Source/data_to_model_20261014.csv"  
    path = '/Workspace/Users/sateeshfrnd@gmail.com/SAMBA/datareadiness_assesment_p/sample_datasets/data_to_model.csv'    
    df = spark.read.csv(
        path,
        header=True,
        inferSchema=True
    )     
    df.show()
    # invalid_values_usable_features_metrics = run_invalid_values_usable_features(df)
    # print(invalid_values_usable_features_metrics)
    run_missingness_profile(df)

if __name__ == "__main__":
    main()