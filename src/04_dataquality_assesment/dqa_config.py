# Config loading and threshold scoring for the SPARK_DRA demo. Self-contained
# (standard library only; no Spark and no external engine dependency).

import json
from pathlib import Path

SEV_INFO = "Information"
SEV_WARN = "Warning"
SEV_FAIL = "Failure"
SEV_STOP = "Hard Stop"
SEV_RANK = {SEV_INFO: 0, SEV_WARN: 1, SEV_FAIL: 2, SEV_STOP: 3}
STATUS = {SEV_INFO: "PASS", SEV_WARN: "WARN", SEV_FAIL: "FAIL", SEV_STOP: "HARD_STOP"}

OP_ALIASES = {
    "gt": "gt", "gte": "gte", "ge": "gte",
    "lt": "lt", "lte": "lte", "le": "lte",
    "eq": "eq", "ne": "ne", "neq": "ne",
    "between": "between",
}
SEV_ALIASES = {
    "info": SEV_INFO, "information": SEV_INFO,
    "warn": SEV_WARN, "warning": SEV_WARN,
    "fail": SEV_FAIL, "failure": SEV_FAIL,
    "stop": SEV_STOP, "hard_stop": SEV_STOP, "hardstop": SEV_STOP, "critical": SEV_STOP,
}


def compare(value, op, bound):
    if op == "between":
        lo, hi = [float(x) for x in str(bound).split(",")]
        return not (lo <= value <= hi)
    b = float(bound)
    if op == "gt":
        return value > b
    if op == "gte":
        return value >= b
    if op == "lt":
        return value < b
    if op == "lte":
        return value <= b
    if op == "eq":
        return value == b
    if op == "ne":
        return value != b
    raise ValueError("Unsupported operator: " + str(op))


class DatasetConfig:
    def __init__(self, data_path, target, problem_type, id_columns,
                 redundancy_pairs, output_dir, thresholds):
        self.data_path = data_path
        self.target = target
        self.problem_type = problem_type
        self.id_columns = id_columns
        self.redundancy_pairs = redundancy_pairs
        self.output_dir = output_dir
        self.thresholds = thresholds


def _load_raw(path):
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in (".yml", ".yaml"):
        try:
            import yaml
        except ImportError:
            raise SystemExit(
                "Reading " + path.name + " needs pyyaml ('pip install pyyaml'); "
                "or use the equivalent .json config."
            )
        return yaml.safe_load(text)
    if suffix == ".json":
        return json.loads(text)
    raise SystemExit("Unsupported config extension: " + suffix + " (use .yml/.yaml/.json)")


def _parse_thresholds(raw):
    rules = []
    for i, t in enumerate(raw or []):
        op_key = str(t["op"]).strip().lower()
        sev_key = str(t["severity"]).strip().lower()
        if op_key not in OP_ALIASES or sev_key not in SEV_ALIASES:
            raise SystemExit("threshold #" + str(i + 1) + " has an invalid op/severity: " + str(t))
        op = OP_ALIASES[op_key]
        sev = SEV_ALIASES[sev_key]
        value = t["value"]
        if isinstance(value, (list, tuple)):
            value_str = ",".join(str(v) for v in value)
        else:
            value_str = str(value)
        blocks = bool(t.get("blocks", sev == SEV_STOP))
        rules.append((str(t["check"]), str(t["metric"]), op, value_str, sev, blocks))
    return rules


def load_config(path):
    path = Path(path)
    raw = _load_raw(path)
    ds = raw.get("dataset", {}) or {}
    base = path.parent

    data_path = Path(ds.get("data_path", "data_to_model.csv"))
    if not data_path.is_absolute():
        data_path = base / data_path
    out_dir = Path(ds.get("output_dir", "dra_output"))
    if not out_dir.is_absolute():
        out_dir = base / out_dir
    pairs = [tuple(p) for p in ds.get("redundancy_pairs", [])]

    return DatasetConfig(
        data_path,
        ds["target"],
        ds.get("problem_type", "regression"),
        list(ds.get("id_columns", [])),
        pairs,
        out_dir,
        _parse_thresholds(raw.get("thresholds", [])),
    )


def family(metric):
    return metric.split(":", 1)[0]


def score(metrics, thresholds):
    out = []
    for check_id, name, value in metrics:
        fam = family(name)
        rules = [t for t in thresholds if t[0] == check_id and t[1] == fam]
        breached = [
            t for t in rules
            if isinstance(value, (int, float)) and value == value and compare(value, t[2], t[3])
        ]
        if not breached:
            sev = SEV_INFO
            status = "PASS"
            blocks = False
            note = "no threshold" if not rules else "within thresholds"
        else:
            worst = max(breached, key=lambda t: SEV_RANK[t[4]])
            sev = worst[4]
            status = STATUS[sev]
            blocks = worst[5] or (sev == SEV_STOP)
            note = worst[2] + " " + worst[3]
        out.append({
            "check": check_id,
            "metric": name,
            "value": round(value, 4) if isinstance(value, (int, float)) else value,
            "status": status,
            "severity": sev,
            "blocks": blocks,
            "note": note,
        })
    return out
