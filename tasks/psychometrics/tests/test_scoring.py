#!/usr/bin/env python3
"""Check that each task scores the way it is meant to.

For every task: the generating model scores 1.0, every rival scores 0.0 and
names the stage it failed at, and a set of malformed or dishonest submissions
are rejected.

    uv run --with numpy --with pandas --with scipy --with semopy \
      python tests/test_scoring.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import sys
import warnings
from pathlib import Path

import pandas as pd
import semopy

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

_BIASED = ["HSNS1", "HSNS10", "HSNS5", "HSNS8"]
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "generators"))

from psychometrics.score import score_model_criteria  # noqa: E402

TASKS = [
    ("generators/level_1/gen_l1_t01_hsns_structure.py",
     "environments/level_1/tasks_json/task_01.json",
     "artifacts/level_1/task_01/data.csv", "two_correlated_factors"),
    ("generators/level_1/gen_l1_t02_dd_structure.py",
     "environments/level_1/tasks_json/task_02.json",
     "artifacts/level_1/task_02/data.csv", "bifactor_general_plus_specifics"),
    ("generators/level_1/gen_l1_t03_local_dependence.py",
     "environments/level_1/tasks_json/task_03.json",
     "artifacts/level_1/task_03/data.csv",
     "unidimensional_with_correlated_residuals"),
    ("generators/level_1/gen_l1_t04_invariant_combination.py",
     "environments/level_1/tasks_json/task_04.json",
     "artifacts/level_1/task_04/data.csv", "DD, bifactor (CORRECT)"),
    ("generators/level_1/gen_l1_t05_defensible_comparisons.py",
     "environments/level_1/tasks_json/task_05.json",
     "artifacts/level_1/task_05/data.csv", "correct"),
    ("generators/level_1/gen_l1_t06_latent_relationships.py",
     "environments/level_1/tasks_json/task_06.json",
     "artifacts/level_1/task_06/data.csv", "joint latent model (CORRECT)"),
    ("generators/level_1/gen_l1_t07_cross_country_replication.py",
     "environments/level_1/tasks_json/task_07.json",
     "artifacts/level_1/task_07/data.csv", "correct"),
    ("generators/level_1/gen_l1_t08_score_justification.py",
     "environments/level_1/tasks_json/task_08.json",
     "artifacts/level_1/task_08/data.csv", "correct"),
]


def load_generator(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_submission(spec, X, items):
    """The submission an agent would make if it genuinely fitted `spec`."""
    if "gender" in items:
        return _group_submission(spec, X, items, _BIASED)
    model = semopy.Model(spec)
    model.fit(X[items])
    ins = model.inspect(std_est=True)
    load = ins[ins.op == "~"].copy()
    load["abs"] = pd.to_numeric(load["Est. Std"], errors="coerce").abs()
    best = load.loc[load.groupby("lval")["abs"].idxmax()]
    cov = ins[(ins.op == "~~") & (ins.lval != ins.rval)
              & ins.lval.isin(["F1", "F2"]) & ins.rval.isin(["F1", "F2"])]
    phi = (float(pd.to_numeric(cov["Est. Std"], errors="coerce").iloc[0])
           if len(cov) else None)
    return {"model_syntax": spec,
            "loadings": {r.lval: round(float(r.abs), 3) for r in best.itertuples()},
            "factor_correlation": phi}


def _group_submission(spec, X, items, truth=None):
    """Level 2: the model, the difference it implies, and the rejected instrument."""
    used = [c for c in X.columns if c in spec]
    model = semopy.Model(spec)
    model.fit(X[used + ["gender"]] if "gender" not in used else X[used])
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & (ins.rval == "gender")
               & ~ins.lval.isin(X.columns)]
    value = (float(pd.to_numeric(rows["Est. Std"], errors="coerce").iloc[0])
             if len(rows) else 0.0)
    return {"model_syntax": spec, "latent_difference": round(value, 4),
            "biased_items": list(truth or [])}


def run_task(gen_path, task_path, data_path, expected_winner):
    gen = load_generator(gen_path)
    params = json.loads((ROOT / task_path).read_text())[0]["scoring_params"]
    items = params["items"]
    data = pd.read_csv(ROOT / data_path, sep="\t")
    keep = params["syntax_whitelist"]["items"]
    for col, val in params.get("subset", {}).items():
        data = data[data[col].isin(val)] if isinstance(val, list) else data[data[col] == val]
    X = data[[c for c in keep if c in data.columns]]
    X = X[(X != 0).all(axis=1)].astype(float)

    failures = []
    print(f"\n{gen.TASK_ID}")
    print(f"  {'submission':38s} {'binary':>6s} {'partial':>7s}  reason")
    # Some tasks answer with a model, others with a judgement; ask the generator.
    if hasattr(gen, "candidate_submissions"):
        cases = gen.candidate_submissions(X)
    else:
        cases = {name: build_submission(spec, X, items)
                 for name, spec in gen.candidate_models().items()}
    for name, submission in cases.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        want = 1.0 if name == expected_winner else 0.0
        ok = result["score_binary"] == want
        failures += [] if ok else [f"{name}: expected {want}, got {result['score_binary']}"]
        print(f"  {name:38s} {result['score_binary']:6.1f} "
              f"{result['score_partial']:7.2f}  {result['reason']}")

    good = cases.get("correct") or build_submission(gen.reference_syntax(), X, items)
    pooled_X = pd.read_csv(ROOT / data_path, sep="\t")
    if "gender" in items:
        pooled_X = pooled_X[pooled_X.gender.isin([1, 2])]
    pooled_X = pooled_X[items]
    pooled_X = pooled_X[(pooled_X != 0).all(axis=1)].astype(float)
    key = ("scoring" if "scoring" in good
           else "replication" if "replication" in good
           else "correlations" if "correlations" in good
           else "comparisons" if "comparisons" in good
           else "latent_difference" if "gender" in items else "loadings")
    fake = ({k: "total_only" for k in good["scoring"]} if key == "scoring"
            else {i: {c: "exact" for c in v} for i, v in good["replication"].items()}
            if key == "replication"
            else {k: 0.3 for k in good["correlations"]} if key == "correlations"
            else {k: True for k in good["comparisons"]} if key == "comparisons"
            else 0.9 if key == "latent_difference" else {k: 0.55 for k in items})
    adversarial = {
        **({} if key in ("comparisons", "correlations", "replication",
                         "scoring") else
           {"pooled sample (no US filter)":
            build_submission(gen.reference_syntax(), pooled_X, items)}),
        f"{key} fabricated": {**good, key: fake},
        f"{key} omitted": {k: v for k, v in good.items() if k != key},
        "code injection": {**good, "model_syntax": "import os"},
        "unknown variable":
            {**good, "model_syntax": f"F1 =~ {'+'.join(items)}+GHOST"},
        "not JSON": "the model is two factors",
    }
    print(f"  {'-' * 70}")
    for name, submission in adversarial.items():
        result = score_model_criteria(submission, params, base_dir=ROOT)
        ok = result["score_binary"] == 0.0
        failures += [] if ok else [f"adversarial {name} scored {result['score_binary']}"]
        print(f"  {name:38s} {result['score_binary']:6.1f} "
              f"{result['score_partial']:7.2f}  {result['reason']}")
    return failures


def main():
    failures = []
    for task in TASKS:
        failures += run_task(*task)
    print()
    if failures:
        for f in failures:
            print(f"FAIL  {f}")
        return 1
    print("all tasks score as intended")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
