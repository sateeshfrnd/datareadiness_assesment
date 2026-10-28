import yaml
from collections import Counter

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

############# Constants ###############
path = "abfss://data-readiness@bnlwestgunileverbd00036.dfs.core.windows.net/Source/data_to_model.csv"
data_profile_config_path = '/Workspace/Users/sambasivarao.katragadda@unilever.com/DataReadinessAssessment/assessment_steps/03_Data_Profiling/test.yml'

############# Utillity Functions ###############
def get_spark(app_name):
    return SparkSession.builder.appName(app_name).getOrCreate()


def load_config(config_path):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

############# Data Profiling Utillity Functions ###############

def get_duplicate_record(df):
    duplicate_results = []
    total_rows = df.count() # total count
    distinct_rows = df.distinct().count() # Unique records
    duplicate_rows = total_rows - distinct_rows
    duplicate_results.append(
            ('schema',
            "Duplicate Rows",
            duplicate_rows,
            "PASS" if duplicate_rows > 0 else "FAIL",
            "INFO"
            )
        )
    # Check duplicate column names
    column_counts = Counter(df.columns)
    duplicate_columns = [
        col_name
        for col_name, cnt in column_counts.items()
        if cnt > 1
    ]
    # Append result to DQ report
    duplicate_results.append(
        (
            'schema',
            "Duplicate Columns",
            len(duplicate_columns),
            "PASS" if len(duplicate_columns) > 0 else "FAIL",
            "INFO"
        )
    )
    return duplicate_results

def check_Empty_rows_colums(df):
    dq_results = []
    condition = None
    for c in df.columns:
        expr = F.col(c).isNull() | (F.trim(F.col(c).cast("string")) == "")
        if condition is None:
            condition = expr
    else:
        condition = condition & expr
    empty_rows = df.filter(condition).count()
    dq_results.append(
        (
            'schema',
            "Empty Rows",
            empty_rows,
            "PASS" if empty_rows > 0 else "FAIL",
            "INFO"
        )
    )

    empty_columns = []

    for c in df.columns:
        non_empty_count = (
            df.filter(
                F.col(c).isNotNull() &
                (F.trim(F.col(c).cast("string")) != "")
            ).count()
        )

        if non_empty_count == 0:
            empty_columns.append(c)

    dq_results.append(
        (
            'schema',
            "Empty Columns",
            len(empty_columns),
            "PASS" if len(empty_columns) > 0 else "FAIL",
            "INFO"
        )
    )
    return dq_results

def check_schema(spark, df, data_profiling):
    expected_schema = data_profiling["schema"]


    data_type_checks = []
    actual_schema = dict(df.dtypes)
    actual_num_cols = len(df.columns)
    
    validation_results = []
    mismatched_columns = []

    for col_name, expected_dtype in expected_schema.items():
        actual_dtype = actual_schema.get(col_name, "Column Missing")
        is_match = actual_dtype == expected_dtype
        if not is_match:
            mismatched_columns.append(col_name)
        validation_results.append(
            (
                col_name,
                expected_dtype,
                actual_dtype,
                is_match,
                "Matched" if is_match else "Mismatched"
            )
        )

    # uncomment if you want to see each column check
    # schema_df = spark.createDataFrame(
    #     validation_results,
    #     [
    #         "Column_Name",
    #         "Expected_DataType",
    #         "Actual_DataType",
    #         "Boolean_Check",
    #         "Status"
    #     ]
    # )
    # display(schema_df)
    print(f'mismatched_columns = {len(mismatched_columns)}')


    data_type_checks.append(
        (
        'schema',
        "Data Type Validation",
        len(mismatched_columns),
        "FAIL" if len(mismatched_columns) > 0 else "PASS",
        "WARN"
        )
    )    
    return data_type_checks

def check_missing_values(spark, df, data_profiling):
    missing_threshold = data_profiling["thresholds"]["missing_threshold"]
    print(f'missing_threshold = {missing_threshold}')

    results = []
    total_nulls = 0
    missing_values = []
    for c in df.columns:
        null_cnt = df.filter(F.col(c).isNull()).count()

        na_cnt = df.filter(
            F.lower(F.trim(F.col(c).cast("string"))) == "na"
        ).count()

        n_a_cnt = df.filter(
            F.lower(F.trim(F.col(c).cast("string"))) == "n/a"
        ).count()

        nan_cnt = df.filter(
            F.lower(F.trim(F.col(c).cast("string"))) == "nan"
        ).count()

        blank_cnt = df.filter(
            F.trim(F.col(c).cast("string")) == ""
        ).count()

        # Total issues for the column
        column_total = (
            null_cnt +
            na_cnt +
            n_a_cnt +
            nan_cnt +
            blank_cnt
        )

        if column_total > 0 :
            total_nulls = total_nulls + 1

        missing_values.append(
            (
                c,
                "Null Value Check",
                total_nulls,
                "FAIL" if total_nulls > missing_threshold else "PASS",
                "WARN"
            )
        )        
    return missing_values


def check_constant_near_constant(spark, df, data_profiling):
    near_constant_threshold = data_profiling["thresholds"]["near_constant_threshold"]
    print(f'near_constant_threshold = {near_constant_threshold}')

    cons_check_results = []
    cons_non_cons_check_results = []

    total_rows = df.count()

    for c in df.columns:
        # Constant Column Check
        distinct_count = df.select(F.countDistinct(F.col(c))).first()[0]
        if distinct_count == 1:
            cons_check_results.append(
                (
                    c,
                    "Constant Columns",
                    distinct_count,
                    'PASS',
                    "INFO"
                )
            )
        else:
            # Near Constant Column Check (95% threshold)
            top_count = (
                df.groupBy(c)
                .count()
                .orderBy(F.desc("count"))
                .first()
            )
            if top_count is not None:
                dominant_count = top_count["count"]
                dominant_percentage = (dominant_count * 100.0) / total_rows
                cons_non_cons_check_results.append(
                (
                    c,
                    "Near Constant Columns",
                    dominant_count,
                    "FAIL" if dominant_percentage > near_constant_threshold else "PASS",
                    "INFO"
                )
            )
    return cons_check_results, cons_non_cons_check_results
               

        





def run_data_profiling(spark, df, data_profile_config_path):
    data_profile_configs = load_config(data_profile_config_path)
    data_profiling = data_profile_configs['data_profiling']
    # print(f'data_profile_configs : {data_profile_configs}')
    data_profiling_results = []

    duplicate_metrics = get_duplicate_record(df)  
    # print(f'duplicate_metrics = {duplicate_metrics}')
    data_profiling_results.append(duplicate_metrics)

    empty_metrics = check_Empty_rows_colums(df)
    # print(f'empty_metrics = {empty_metrics}')
    data_profiling_results.append(empty_metrics)

    schema_metrics = check_schema(spark, df, data_profiling)
    # print(schema_metrics)
    data_profiling_results.append(schema_metrics)

    missing_values_metrics = check_missing_values(spark, df, data_profiling)
    # print(missing_values_metrics)
    data_profiling_results.append(missing_values_metrics)

    cons_check_results, cons_non_cons_check_results = check_constant_near_constant(spark, df, data_profiling)
    data_profiling_results.append(cons_check_results)
    data_profiling_results.append(cons_non_cons_check_results)

    print(data_profiling_results)
    display(
        spark.createDataFrame(
            [item for sublist in data_profiling_results for item in sublist],
            ["column","metric", "value", "status",'severity']
        )
    )





  

if __name__ ==  '__main__':
    spark = get_spark('data_profiling')
    df = spark.read.csv(
        path,
        header=True,
        inferSchema=True
    )    
    run_data_profiling(spark, df, data_profile_config_path)

