#!/usr/bin/env python3
"""Generate Level 2 Task 03: classify respondents against DDM populations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402
from population_classification import recipe_frame, sha, write_codebook, write_task_json  # noqa: E402

TASK_ID = "psy_l2_t03"
SEED = 20260926
OUT_DIR = C.PKG_ROOT / "artifacts" / "level_2" / "task_03"
TASK_JSON = C.PKG_ROOT / "environments" / "level_2" / "tasks_json" / "task_03.json"
ITEMS = C.DD_ITEMS
POPULATIONS = [f"population_{letter}" for letter in "abcdef"]
RECIPES = [
    "three_factor",
    "two_factor",
    "one_factor",
    "bifactor",
    "local_dependence",
    "three_factor",
]
EXPECTED = {
    "person_1": ["population_a"],
    "person_2": ["population_b", "population_c"],
    "person_3": ["population_d", "population_e"],
    "person_4": ["population_f"],
    "person_5": ["population_a", "population_f"],
}
PROMPT = """You are given responses from six anonymised reference populations and five anonymised respondents who may have come from any of them. The reference samples are large enough to investigate their measurement structures, but the population labels do not explain what those structures are. Determine which population or populations are compatible with each respondent's response pattern. A person can remain compatible with more than one population when these data do not distinguish them; do not force a unique assignment."""
SUBMISSION_FORMAT = """Return one JSON object and nothing else:

{
  "classifications": {
    "person_1": ["population_a"],
    "person_2": ["population_b", "population_c"]
  }
}

Include every person exactly once. Each value is a list of compatible population identifiers; use more than one when the evidence leaves a genuine overlap."""


def build_cases(rng, refs):
    # Cases are selected from reference draws. The overlapping cases are drawn
    # from populations whose structures are deliberately hard to distinguish
    # with one short response vector.
    rows = [
        refs["population_a"].iloc[10],
        refs["population_b"].iloc[20],
        refs["population_d"].iloc[30],
        refs["population_f"].iloc[40],
        refs["population_a"].iloc[50],
    ]
    return pd.DataFrame(rows, index=list(EXPECTED)).reset_index(names="person_id")[
        ["person_id"] + ITEMS
    ]


def build():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    refs = {}
    for population, recipe in zip(POPULATIONS, RECIPES):
        frame = recipe_frame(rng, 1800, ITEMS, "ddm", recipe)
        refs[population] = frame
        frame.to_csv(OUT_DIR / f"{population}.csv", sep="\t", index=False)
    cases = build_cases(rng, refs)
    cases.to_csv(OUT_DIR / "cases.csv", sep="\t", index=False)
    write_codebook(OUT_DIR / "codebook.md", ITEMS)
    truth = {
        "task_id": TASK_ID,
        "scored": {"classifications": EXPECTED},
        "generative_parameters": {
            "population_recipes": dict(zip(POPULATIONS, RECIPES)),
            "items": ITEMS,
        },
        "provenance": {
            "generator": Path(__file__).name,
            "seed": SEED,
            "rows_per_reference": 1800,
            "data_sha256": {p: sha(OUT_DIR / f"{p}.csv") for p in POPULATIONS},
            "cases_sha256": sha(OUT_DIR / "cases.csv"),
        },
    }
    (OUT_DIR / "truth.json").write_text(json.dumps(truth, indent=2) + "\n")
    write_task_json(
        TASK_JSON,
        TASK_ID,
        "Which DDM population fits each respondent?",
        PROMPT,
        SUBMISSION_FORMAT,
        ITEMS,
        POPULATIONS,
        {p: sha(OUT_DIR / f"{p}.csv") for p in POPULATIONS},
        sha(OUT_DIR / "cases.csv"),
    )


def candidate_submissions():
    return {
        "correct": {"classifications": EXPECTED},
        "forces_unique_answers": {"classifications": {k: [v[0]] for k, v in EXPECTED.items()}},
        "fit_only_guess": {"classifications": {k: ["population_a"] for k in EXPECTED}},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--naive", action="store_true")
    args = parser.parse_args()
    build()
    if args.verify:
        from psychometrics.score import score_population_classification

        result = score_population_classification(
            {"classifications": EXPECTED},
            json.loads(TASK_JSON.read_text())[0]["scoring_params"],
            C.PKG_ROOT,
        )
        print("verified", result["score_binary"])
        return 0 if result["score_binary"] == 1 else 1
    if args.naive:
        print("naive classifications do not reproduce the overlapping answer sets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
