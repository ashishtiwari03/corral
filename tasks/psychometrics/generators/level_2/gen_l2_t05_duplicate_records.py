#!/usr/bin/env python3
"""Generate Level 2 Task 05, duplicate records against chance collisions.

Part of the sample was delivered twice under fresh identifiers, and the block
that was repeated sits high on both traits, so the correlation between them is
inflated. Thousands of other rows share an answer pattern purely by chance,
because ten five-point items cannot tell 12,000 people apart. Removing every
repeated pattern overshoots further than leaving them all in.

    python gen_l2_t05_duplicate_records.py
    python gen_l2_t05_duplicate_records.py --verify
    python gen_l2_t05_duplicate_records.py --naive
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import common as C  # noqa: E402

TASK_ID = "psy_l2_t05_duplicate_records"
SEED = 20260925
OUT_DIR = C.PKG_ROOT / "artifacts" / "level_2" / "task_05"
TASK_JSON = C.PKG_ROOT / "environments" / "level_2" / "tasks_json" / "task_05.json"

ITEMS = C.HSNS_ITEMS
F1_ITEMS = ["HSNS1", "HSNS4", "HSNS5", "HSNS6", "HSNS8", "HSNS10"]
F2_ITEMS = ["HSNS2", "HSNS3", "HSNS7", "HSNS9"]
LOADINGS = {
    "HSNS1": 0.55,
    "HSNS4": 0.30,
    "HSNS5": 0.70,
    "HSNS6": 0.48,
    "HSNS8": 0.71,
    "HSNS10": 0.66,
    "HSNS2": 0.76,
    "HSNS3": 0.58,
    "HSNS7": 0.69,
    "HSNS9": 0.52,
}
PHI = 0.35
N_RESPONDENTS = 12_000

# A respondent is re-delivered with probability rising in the sum of both
# traits, so the repeated block is the high-scoring end of the sample.
REDELIVERY_CENTRE, REDELIVERY_SLOPE = 1.6, 3.0

# What a re-delivered row keeps, and what it is given fresh.
IDENTITY = ITEMS + ["age", "accuracy"]
REISSUED = ["participant_id", "session_id", "recorded_on"]

DIAGNOSIS = "both_present"
CORRELATION_TOLERANCE = 0.02

PROMPT = (
    "You are given responses to the Hypersensitive Narcissism Scale collected by a survey "
    "panel. Thousands of rows share an answer pattern with at least one other row.\n\n"
    "Two things can produce that. Records may have been delivered more than once, in which "
    "case the repeats are not independent respondents and inflate whatever is estimated from "
    "them. Or two different people may simply have chosen the same answers, which ten "
    "five-point items make far more likely than it first appears.\n\n"
    "Work out which is happening here, and report the correlation between the two traits "
    "that the data actually support. State which columns together identify a record that is "
    "the same respondent as another; the analysis is repeated with one row kept per "
    "combination of those columns.\n"
)

SUBMISSION_FORMAT = """\
A single JSON object:

{
  "model_syntax": "F1 =~ ...\\nF2 =~ ...\\nF1 ~~ F2",
  "diagnosis": "duplicate_records_only|chance_collisions_only|both_present",
  "duplicate_key": ["<column>", "..."]
}

`duplicate_key` is the set of columns that, taken together, mark two rows as the
same respondent. One row is kept per distinct combination of those columns, the
submitted model is refitted on what remains, and the trait correlation is read
from that refit.
"""


def reference_syntax():
    """Two correlated traits."""
    return f"F1 =~ {'+'.join(F1_ITEMS)}\nF2 =~ {'+'.join(F2_ITEMS)}\nF1 ~~ F2"


def respondents(rng, n):
    """Draw n distinct respondents."""
    eta = rng.multivariate_normal([0.0, 0.0], [[1.0, PHI], [PHI, 1.0]], n)
    out = {}
    for item, loading in LOADINGS.items():
        index = 0 if item in F1_ITEMS else 1
        y = loading * eta[:, index] + rng.normal(0, np.sqrt(1 - loading**2), n)
        out[item] = C.categorize(y, C.THRESHOLDS[item])
    frame = pd.DataFrame(out)
    frame["age"] = np.clip(rng.normal(35, 11, n).round(), 18, 78).astype(int)
    frame["gender"] = rng.choice([1, 2], size=n, p=[0.61, 0.39])
    frame["accuracy"] = np.clip(rng.beta(8, 1.5, n) * 100, 1, 100).round().astype(int)
    frame["country"] = "US"
    frame["_trait_sum"] = eta.sum(axis=1)
    return frame


def deliver(rng, people):
    """Re-deliver the high-scoring end of the sample under fresh identifiers."""
    z = (people._trait_sum - people._trait_sum.mean()) / people._trait_sum.std()
    chance = 1 / (1 + np.exp(-(z - REDELIVERY_CENTRE) * REDELIVERY_SLOPE))
    repeated = people[rng.random(len(people)) < chance.values]
    delivered = pd.concat([people, repeated], ignore_index=True)
    delivered = delivered.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

    n = len(delivered)
    delivered["participant_id"] = [f"R{i:06d}" for i in range(1, n + 1)]
    delivered["session_id"] = [f"S{i:06d}" for i in rng.permutation(n) + 1]
    start = np.datetime64("2026-02-02")
    delivered["recorded_on"] = (start + rng.integers(0, 54, n).astype("timedelta64[D]")).astype(str)
    return delivered.drop(columns="_trait_sum"), len(repeated)


def analysis_sample(df):
    """Responses, as delivered."""
    return df[ITEMS].astype(float)


def deduplicate(df, key):
    """One row per distinct combination of `key`."""
    return df[~df.duplicated(subset=list(key), keep="first")]


def factor_correlation(X):
    """The correlation between the two traits under the reference model."""
    model = C.fit(reference_syntax(), X, ITEMS)
    estimates = model.inspect(std_est=True)
    cov = estimates[
        (estimates.op == "~~") & (estimates.lval != estimates.rval) & ~estimates.lval.isin(ITEMS)
    ]
    return round(float(cov["Est. Std"].iloc[0]), 3)


def population_matrix(rng):
    """Correlations a correctly specified model reproduces, from a large draw."""
    big = respondents(rng, 200_000)
    return np.corrcoef(big[ITEMS].values.T.astype(float)).round(3)


def write_codebook(path, df):
    path.write_text(
        "# Codebook - panel delivery\n\n"
        "Tab-separated, one row per delivered record. Responses run from 1 (Disagree) to "
        "5 (Agree). Every item was answered; there are no missing responses.\n\n"
        "| item | text |\n|---|---|\n"
        + "\n".join(f"| `{item}` | {C.ITEM_TEXT[item]} |" for item in ITEMS)
        + "\n\n| variable | description |\n|---|---|\n"
        "| `participant_id` | identifier assigned by the panel on delivery |\n"
        "| `session_id` | identifier assigned when the record was transmitted |\n"
        "| `recorded_on` | date the record was transmitted |\n"
        "| `age` | respondent age in years |\n"
        "| `gender` | 1 = Male, 2 = Female |\n"
        "| `accuracy` | self-rated response accuracy, 0-100 |\n"
        "| `country` | collection country; all rows are US |\n\n"
        f"Rows delivered: {len(df):,}.\n"
    )


def build_truth(floor, pop, data_sha, rows, n_distinct, n_repeated, corrected):
    return {
        "task_id": TASK_ID,
        "scored": {
            "diagnosis": DIAGNOSIS,
            "distinct_respondents": n_distinct,
            "repeats_remaining": 0,
            "corrected_factor_correlation": corrected,
        },
        "scoring_reference": {
            "reference_model_syntax": reference_syntax(),
            "reference_role": "floor",
            "reference_criteria": floor,
            "population_correlation_matrix": pop,
            "item_order": ITEMS,
        },
        "generative_parameters": {
            "trait_correlation": PHI,
            "distinct_respondents": n_distinct,
            "records_delivered_twice": n_repeated,
            "redelivery": {"centre": REDELIVERY_CENTRE, "slope": REDELIVERY_SLOPE},
            "identity_columns": IDENTITY,
            "reissued_columns": REISSUED,
        },
        "provenance": C.provenance(Path(__file__).name, SEED, rows, data_sha),
    }


def build_task_json(data_sha):
    contract = C.scoring_contract(
        "artifacts/level_2/task_05/truth.json",
        ITEMS,
        [
            {
                "key": "diagnosis",
                "fn": "score_label",
                "truth_key": "scored.diagnosis",
                "criterion": "anomaly_mechanism",
            },
            {
                "key": "retained_respondents",
                "fn": "score_scalar",
                "truth_key": "scored.distinct_respondents",
                "derive_from": "dedup_by_key",
                "tol": 0,
                "criterion": "record_identity",
            },
            {
                "key": "repeats_remaining",
                "fn": "score_scalar",
                "truth_key": "scored.repeats_remaining",
                "derive_from": "dedup_by_key",
                "tol": 0,
                "criterion": "record_identity",
            },
            {
                "key": "corrected_factor_correlation",
                "fn": "score_scalar",
                "truth_key": "scored.corrected_factor_correlation",
                "derive_from": "dedup_by_key",
                "tol": CORRELATION_TOLERANCE,
                "criterion": "trait_relationship",
            },
        ],
    )
    contract["identity_columns"] = IDENTITY
    return [
        {
            "id": TASK_ID,
            "name": "Duplicate records or chance collisions?",
            "uuid": "1d2c5a7e-3f44-4c8b-9a10-6b5e2f7c9d31",
            "keywords": ["psychometrics", "data integrity", "duplicate records", "data cleaning"],
            "metrics": ["binary", "partial"],
            "level": 2,
            "description": PROMPT,
            "submission_format": SUBMISSION_FORMAT,
            "initial_input": {
                "dataset": "data.csv",
                "codebook": "codebook.md",
                "data_sha256": data_sha,
            },
            "tools": [],
            "scoring_function": "score_model_criteria",
            "scoring_params": contract,
        }
    ]


def candidate_submissions(X):
    """The intended answer and the ways a cleaning decision goes wrong."""
    del X
    correct = {
        "model_syntax": reference_syntax(),
        "diagnosis": DIAGNOSIS,
        "duplicate_key": IDENTITY,
    }
    return {
        "correct": correct,
        "leaves the data as delivered": {
            **correct,
            "diagnosis": "chance_collisions_only",
            "duplicate_key": IDENTITY + REISSUED,
        },
        "treats every repeated pattern as a duplicate": {
            **correct,
            "diagnosis": "duplicate_records_only",
            "duplicate_key": ITEMS,
        },
        "right key, wrong account of it": {**correct, "diagnosis": "duplicate_records_only"},
        "keys on the answers and age only": {**correct, "duplicate_key": ITEMS + ["age"]},
    }


def verify(delivered, n_repeated):
    """Confirm the two mechanisms are both present and only one key separates them."""
    as_delivered = factor_correlation(analysis_sample(delivered))
    routes = {
        "leave every row in": delivered,
        "drop every row sharing an answer pattern": delivered[
            ~delivered.duplicated(subset=ITEMS, keep=False)
        ],
        "keep one row per answer pattern": deduplicate(delivered, ITEMS),
        "keep one per answers + age + accuracy": deduplicate(delivered, IDENTITY),
    }
    pattern_size = delivered.groupby(ITEMS)[ITEMS[0]].transform("size")
    sharing = int((pattern_size > 1).sum())
    chance = sharing - 2 * n_repeated

    print(f"delivered rows: {len(delivered):,}, distinct respondents: {N_RESPONDENTS:,}")
    print(f"  rows sharing an answer pattern with another row: {sharing:,}")
    print(f"    accounted for by re-delivery: {2 * n_repeated:,}")
    print(f"    two different people, same answers: {chance:,}")
    print(f"\n  trait correlation, by what gets removed (true value {PHI}):")
    estimates = {}
    for label, frame in routes.items():
        estimates[label] = factor_correlation(analysis_sample(frame))
        print(f"    {label:42s} {estimates[label]:+.3f}  (n={len(frame):,})")

    correct = estimates["keep one per answers + age + accuracy"]
    return C.report(
        [
            ("both mechanisms are present in the data", n_repeated > 500 and chance > 150),
            ("the re-delivered block inflates the correlation", as_delivered - correct > 0.04),
            (
                "removing every repeated pattern overshoots further than leaving them in",
                abs(estimates["drop every row sharing an answer pattern"] - correct)
                > abs(as_delivered - correct),
            ),
            (
                "keeping one row per answer pattern still loses real respondents",
                len(routes["keep one row per answer pattern"]) < N_RESPONDENTS,
            ),
            (
                "the answers with age and accuracy recover every respondent exactly",
                len(routes["keep one per answers + age + accuracy"]) == N_RESPONDENTS,
            ),
            ("and recover the true correlation", abs(correct - PHI) <= CORRELATION_TOLERANCE),
        ]
    )


def naive(delivered):
    """Confirm the obvious cleaning move lands further from the truth than none."""
    as_delivered = factor_correlation(analysis_sample(delivered))
    deduped = factor_correlation(analysis_sample(deduplicate(delivered, ITEMS)))
    dropped = factor_correlation(
        analysis_sample(delivered[~delivered.duplicated(subset=ITEMS, keep=False)])
    )
    print(f"  true trait correlation                      {PHI:+.3f}")
    print(f"  analysing the file as delivered             {as_delivered:+.3f}")
    print(f"  keeping one row per answer pattern          {deduped:+.3f}")
    print(f"  dropping every row in a repeated pattern    {dropped:+.3f}")
    print("\n  the repeated patterns are not all duplicates, so deduplicating on")
    print("  the answers alone deletes real respondents and overshoots.")
    return C.report(
        [
            ("the file as delivered overstates the correlation", as_delivered - PHI > 0.04),
            (
                "dropping every repeated pattern lands further away than doing nothing",
                abs(dropped - PHI) > abs(as_delivered - PHI),
            ),
            (
                "and keeping one row per answer pattern still misses",
                abs(deduped - PHI) > CORRELATION_TOLERANCE,
            ),
        ]
    )


def main():
    action = C.mode()
    rng = np.random.default_rng(SEED)
    people = respondents(rng, N_RESPONDENTS)
    delivered, n_repeated = deliver(rng, people)
    if action == "verify":
        return verify(delivered, n_repeated)
    if action == "naive":
        return naive(delivered)
    pop = population_matrix(np.random.default_rng(SEED + 1001))
    floor = C.evaluate_model(reference_syntax(), analysis_sample(delivered), ITEMS, pop)
    C.write_artifacts(
        OUT_DIR,
        TASK_JSON,
        delivered,
        lambda sha: build_truth(
            floor,
            pop.tolist(),
            sha,
            len(delivered),
            N_RESPONDENTS,
            n_repeated,
            factor_correlation(analysis_sample(deduplicate(delivered, IDENTITY))),
        ),
        build_task_json,
    )
    write_codebook(OUT_DIR / "codebook.md", delivered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
