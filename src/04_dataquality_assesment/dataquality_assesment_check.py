
import argparse
import csv
import sys
from pathlib import Path

import dqa_config
import dqa_engine as engine

DEFAULT_CONFIG = '/Workspace/Users/sateeshfrnd@gmail.com/SAMBA/datareadiness_assesment/config/data_to_model.yml'

def run(config_path: str | Path = DEFAULT_CONFIG) -> list[dict]:
    print(f"Running with config: {config_path}")
    cfg = dqa_config.load_config(config_path)
    # print(cfg.target)
    spark = engine.get_spark()
    spark.sparkContext.setLogLevel("ERROR")
    df = (spark.read.option("header", True).option("inferSchema", True)
          .csv(str(cfg.data_path)).cache())
    df.show()
    n_rows = df.count()
    print(n_rows)

    n_rows = df.count()
    ctx = engine.classify_columns(df, cfg.target, cfg.id_columns)

    metrics = []
    metrics += engine.dr01_structure(df, n_rows)
    metrics += engine.dr02_target(df, n_rows, cfg.target)
    metrics += engine.dr03_predictor_quality(df, n_rows, ctx)
    metrics += engine.dr04_missingness(df, n_rows, cfg.target)

    dr05 = engine.dr05_sample_size(df, n_rows, ctx)
    metrics += dr05
    usable = next((v for _, m, v in dr05 if m == "usable_rows"), float(n_rows))

    sig_metrics, _ = engine.dr06_signal_and_dr08_leakage(df, ctx)
    metrics += sig_metrics
    metrics += engine.dr07_redundancy(df, ctx, cfg.redundancy_pairs)

    _, strategy = engine.dr10_evaluation_readiness(ctx, usable)

    try:
        metrics += engine.dr11_baseline(df, ctx)
    except Exception as exc:
        metrics.append(("DR-11", "baseline_r2", float("nan")))
        print("[warn] baseline modelling failed: " + str(exc))

    scored = dqa_config.score(metrics, cfg.thresholds)
    _report(scored, n_rows, ctx, strategy, cfg)
    spark.stop()
    return scored


def _report(scored, n_rows, ctx, strategy, cfg):
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with (cfg.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["check", "metric", "value", "status",
                                           "severity", "blocks", "note"])
        w.writeheader()
        w.writerows(scored)

    findings = [r for r in scored if r["status"] != "PASS"]
    hard = [r for r in scored if r["status"] == "HARD_STOP"]
    fails = [r for r in scored if r["status"] == "FAIL"]
    warns = [r for r in scored if r["status"] == "WARN"]

    if hard:
        decision = "NO-GO"
        reason = "hard stop: " + ", ".join(sorted({r["metric"] for r in hard}))
    elif fails:
        decision = "CONDITIONAL GO"
        reason = str(len(fails)) + " failure(s) to resolve"
    else:
        decision = "GO"
        reason = (str(len(warns)) + " warning(s)") if warns else "clean"

    n_cols = len(ctx["numeric"]) + len(ctx["categorical"])
    print("\nMOT Data Readiness (PYSPARK DEMO) - " + cfg.data_path.name +
          " (" + str(n_rows) + " rows x " + str(n_cols) + " cols, target=" +
          cfg.target + ", " + cfg.problem_type + ")\n")

    header = f"{'CHECK':6} {'METRIC':46} {'VALUE':>12}  {'STATUS':9} NOTE"
    print(header)
    print("-" * len(header))
    for r in sorted(findings, key=lambda r: r["check"]):
        val = f"{r['value']:.4f}" if isinstance(r["value"], float) else str(r["value"])
        print(f"{r['check']:6} {r['metric'][:46]:46} {val:>12}  {r['status']:9} {r['note']}")
    if not findings:
        print("(no findings above Information)")

    info = {(r["check"], dqa_config.family(r["metric"])): r["value"] for r in scored}
    print("\nContext:")
    for key, label in [
        (("DR-01", "row_count"), "rows"),
        (("DR-01", "column_count"), "columns"),
        (("DR-05", "usable_rows"), "usable rows (target present)"),
        (("DR-11", "baseline_r2"), "baseline R2 (lin reg, 80/20)"),
    ]:
        if key in info:
            print(f"  {label:34}: {info[key]}")
    print(f"  {'recommended evaluation':34}: {strategy}")

    counts = {s: sum(1 for r in scored if r["severity"] == s)
              for s in ("Information", "Warning", "Failure", "Hard Stop")}
    print("\n" + "=" * len(header))
    print("FINAL RECOMMENDATION: " + decision + "  (" + reason + ")")
    print("Severity counts: " + str(counts))
    print("Outputs: " + str(cfg.output_dir) + "  (results.csv)")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run the data readiness assessment.")
    # parser.add_argument("-f", "--file", dest="config_path", default=DEFAULT_CONFIG,
    #                     help="Path to the config YAML file.")
    # args = parser.parse_args()
    # print(args)
    # run(args.config_path)
    run(DEFAULT_CONFIG)