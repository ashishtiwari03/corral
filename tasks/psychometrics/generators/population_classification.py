"""Shared simulation and artifact helpers for Level 2 population tasks."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import common as C


def ordinal_block(rng, n, items, factors, phi, loadings, thresholds):
    """Draw ordinal responses from a Gaussian latent-factor model."""
    eta = rng.multivariate_normal(np.zeros(len(factors)), phi, n)
    index = {name: i for i, name in enumerate(factors)}
    out = {}
    for item in items:
        terms = sum((loading * eta[:, index[factor]] for factor, loading in loadings[item]), 0.0)
        explained = sum(loading**2 for _, loading in loadings[item])
        out[item] = C.categorize(
            terms + rng.normal(0, np.sqrt(max(1.0 - explained, 0.08)), n),
            thresholds[item],
        )
    return pd.DataFrame(out)[items]


def bifactor_block(rng, n, items, specific_of, general_loading, specific_loading, thresholds):
    g = rng.normal(size=n)
    specifics = {name: rng.normal(size=n) for name in dict.fromkeys(specific_of.values())}
    out = {}
    for item in items:
        specific = specifics[specific_of[item]]
        residual = np.sqrt(max(1 - general_loading[item] ** 2 - specific_loading[item] ** 2, 0.08))
        score = general_loading[item] * g + specific_loading[item] * specific
        out[item] = C.categorize(score + rng.normal(0, residual, n), thresholds[item])
    return pd.DataFrame(out)[items]


def recipe_frame(rng, n, items, instrument, recipe):
    """Generate one hidden reference population without naming its model."""
    if instrument == "ddm":
        groups = {"P": C.DD_ITEMS[0:4], "N": C.DD_ITEMS[4:8], "M": C.DD_ITEMS[8:12]}
        base = {item: [(factor, 0.68)] for factor, members in groups.items() for item in members}
        if recipe == "three_factor":
            factors, phi, loadings = ["P", "N", "M"], np.full((3, 3), 0.22), base
            np.fill_diagonal(phi, 1.0)
            return ordinal_block(rng, n, items, factors, phi, loadings, C.THRESHOLDS)
        if recipe == "two_factor":
            loadings = {
                item: [("A" if item.startswith(("DDP", "DDN")) else "B", 0.70)] for item in items
            }
            return ordinal_block(
                rng, n, items, ["A", "B"], [[1, 0.30], [0.30, 1]], loadings, C.THRESHOLDS
            )
        if recipe == "one_factor":
            loadings = {item: [("G", 0.62)] for item in items}
            return ordinal_block(rng, n, items, ["G"], [[1]], loadings, C.THRESHOLDS)
        if recipe == "bifactor":
            specific = {item: factor for factor, members in groups.items() for item in members}
            return bifactor_block(
                rng,
                n,
                items,
                specific,
                {item: 0.43 for item in items},
                {item: 0.50 for item in items},
                C.THRESHOLDS,
            )
        if recipe == "local_dependence":
            frame = ordinal_block(
                rng,
                n,
                items,
                ["P", "N", "M"],
                np.array([[1, 0.22, 0.22], [0.22, 1, 0.22], [0.22, 0.22, 1]]),
                base,
                C.THRESHOLDS,
            )
            # A small shared response tendency creates dependence not represented by the factors.
            shared = rng.random(n) < 0.22
            frame.loc[shared, ["DDP1", "DDP2"]] = np.minimum(
                frame.loc[shared, ["DDP1", "DDP2"]] + 1, 5
            )
            return frame
        loadings = {
            item: [
                ("P" if item.startswith("DDP") else "N" if item.startswith("DDN") else "M", 0.62)
            ]
            for item in items
        }
        return ordinal_block(
            rng,
            n,
            items,
            ["P", "N", "M"],
            np.array([[1, 0.22, 0.22], [0.22, 1, 0.22], [0.22, 0.22, 1]]),
            loadings,
            C.THRESHOLDS,
        )

    f1 = ["HSNS1", "HSNS4", "HSNS5", "HSNS6", "HSNS8", "HSNS10"]
    base = {
        item: [("F1" if item in f1 else "F2", loading)]
        for item, loading in {
            "HSNS1": 0.55,
            "HSNS4": 0.42,
            "HSNS5": 0.70,
            "HSNS6": 0.58,
            "HSNS8": 0.68,
            "HSNS10": 0.66,
            "HSNS2": 0.70,
            "HSNS3": 0.58,
            "HSNS7": 0.69,
            "HSNS9": 0.52,
        }.items()
    }
    phi = np.array([[1, 0.35], [0.35, 1.0]])
    if recipe == "one_factor":
        return ordinal_block(
            rng, n, items, ["G"], [[1]], {item: [("G", 0.62)] for item in items}, C.THRESHOLDS
        )
    if recipe == "cross_loading":
        base["HSNS9"] = [("F2", 0.52), ("F1", 0.28)]
        return ordinal_block(rng, n, items, ["F1", "F2"], phi, base, C.THRESHOLDS)
    if recipe == "bifactor":
        specific = {item: ("S1" if item in f1 else "S2") for item in items}
        return bifactor_block(
            rng,
            n,
            items,
            specific,
            {item: 0.43 for item in items},
            {item: 0.47 for item in items},
            C.THRESHOLDS,
        )
    if recipe == "residual_dependence":
        frame = ordinal_block(rng, n, items, ["F1", "F2"], phi, base, C.THRESHOLDS)
        shared = rng.random(n) < 0.25
        frame.loc[shared, ["HSNS5", "HSNS10"]] = np.minimum(
            frame.loc[shared, ["HSNS5", "HSNS10"]] + 1, 5
        )
        return frame
    if recipe == "altered_relation":
        return ordinal_block(
            rng, n, items, ["F1", "F2"], [[1, 0.78], [0.78, 1]], base, C.THRESHOLDS
        )
    return ordinal_block(rng, n, items, ["F1", "F2"], phi, base, C.THRESHOLDS)


def write_codebook(path, items):
    lines = [
        "# Codebook - anonymised population comparison",
        "",
        "Each response is rated from 1 (Disagree) to 5 (Agree). A zero would mean a missing response; the generated case and reference files used here are complete.",
        "",
        "The reference files are labelled only by anonymous population identifiers. Their measurement structures are not supplied: infer what is useful from the responses.",
        "",
        "| item | text |",
        "|---|---|",
        *[f"| {item} | {C.ITEM_TEXT[item]} |" for item in items],
        "",
    ]
    path.write_text("\n".join(lines))


def write_task_json(
    path, task_id, name, prompt, submission_format, items, populations, data_sha, cases_sha
):
    contract = {
        "task_type": "population_classification",
        "truth_path": f"artifacts/level_2/task_{task_id[-2:]}/truth.json",
        "items": items,
        "populations": populations,
        "reference_datasets": {name: f"{name}.csv" for name in populations},
        "cases": "cases.csv",
        "codebook": "codebook.md",
        "case_data_sha256": cases_sha,
    }
    path.write_text(
        json.dumps(
            [
                {
                    "id": task_id,
                    "name": name,
                    "uuid": f"{task_id}-0000-4000-8000-000000000001",
                    "keywords": ["psychometrics", "population classification", "model comparison"],
                    "metrics": ["binary", "partial"],
                    "level": 2,
                    "description": prompt,
                    "submission_format": submission_format,
                    "initial_input": {
                        "reference_datasets": contract["reference_datasets"],
                        "cases": "cases.csv",
                        "codebook": "codebook.md",
                        "data_sha256": data_sha,
                        "case_data_sha256": cases_sha,
                    },
                    "tools": [],
                    "scoring_function": "score_population_classification",
                    "scoring_params": contract,
                }
            ],
            indent=2,
        )
        + "\n"
    )


def sha(path):
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()
