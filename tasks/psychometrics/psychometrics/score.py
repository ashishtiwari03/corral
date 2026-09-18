"""Scoring for the psychometrics environment.

A submission gives a model, not a number, so scoring is not a comparison of
scalars. It runs in three stages. All three must pass, and they are never
added up into a weighted total.

  Stage 1, constraints. Is the model usable at all? A negative variance, two
  factors correlated above .90, or a factor no item really loads on make a
  model invalid rather than merely worse. Such a model is dropped here,
  however well it fits.

  Stage 2, comparison. The model that generated the data sets a floor. The
  submission has to be at least as good as it on every measure - three of fit,
  one of how many parameters it spends, and one of how close the correlations
  it implies come to the truth - allowing a small margin for sampling noise.
  Beating the floor is fine and never counts against a submission.

  Stage 3, claims. The numbers the agent reported, checked against the values
  the data were generated from. Not against what the reference model happens to
  estimate, so a better model is rewarded rather than penalised.

Results:

    score_binary   1.0 only if all three stages pass
    score_partial  share of applicable checks passed; for diagnosis only
    checks_vector  {name: "PASS" | "FAIL" | "n/a"}
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

# --------------------------------------------------------------------------
# Submitted-syntax validation
# --------------------------------------------------------------------------
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ALLOWED_OPS = ("=~", "~~", "~")


class InvalidSubmission(ValueError):
    """The submitted model cannot be accepted as input."""


def validate_syntax(spec: str, items: list[str], max_factors: int = 6) -> str:
    """Check a submitted model before fitting it.

    Only known variables and a few operators are allowed, so a submission
    cannot smuggle in code or refer to columns it was not given.

    lavaan syntax is declarative rather than executable, but agent output is
    still untrusted input: only the listed observed variables, the three
    structural operators and numeric fixings are permitted.
    """
    if not isinstance(spec, str) or not spec.strip():
        raise InvalidSubmission("empty model specification")
    if len(spec) > 8000:
        raise InvalidSubmission("model specification too long")
    for bad in ("import", "exec", "eval", "__", "open(", "system", ";"):
        if bad in spec:
            raise InvalidSubmission(f"forbidden token in specification: {bad!r}")

    lines = [ln.strip() for ln in spec.splitlines() if ln.strip()]
    if not lines:
        raise InvalidSubmission("no statements in specification")

    latents: set[str] = set()
    for line in lines:
        if not any(op in line for op in _ALLOWED_OPS):
            raise InvalidSubmission(f"line has no permitted operator: {line!r}")
        if "=~" in line:
            latents.add(line.split("=~")[0].strip())
    if len(latents) > max_factors:
        raise InvalidSubmission(
            f"{len(latents)} latent variables exceeds the limit of {max_factors}"
        )

    known = set(items) | latents
    for tok in _TOKEN.findall(spec):
        if tok not in known:
            raise InvalidSubmission(f"unknown variable in specification: {tok!r}")
    return spec


# --------------------------------------------------------------------------
# Fitting
# --------------------------------------------------------------------------
def observed_in_spec(spec: str, known: list[str]) -> set:
    """The data columns a model actually refers to."""
    return {tok for tok in _TOKEN.findall(spec) if tok in set(known)}


def evaluate_model(
    spec: str, X: pd.DataFrame, items: list[str], pop: np.ndarray | None = None
) -> tuple[dict, Any]:
    """Fit a model and measure it. Returns (measures, fitted model)."""
    import semopy

    model = semopy.Model(spec)
    model.fit(X[items])
    stats = semopy.calc_stats(model)

    sigma = model.calc_sigma()[0]
    order = list(model.vars["observed"])
    idx = [order.index(it) for it in items]
    sigma = sigma[np.ix_(idx, idx)]
    scale = np.sqrt(np.diag(sigma))
    implied = sigma / np.outer(scale, scale)

    empirical = np.corrcoef(X[items].values.T)
    iu = np.triu_indices(len(items), 1)
    chi2 = float(stats["chi2"].iloc[0])
    n_par = len(model.param_vals)

    criteria = {
        "df": float(stats["DoF"].iloc[0]),
        "chi2": chi2,
        "CFI": float(stats["CFI"].iloc[0]),
        "RMSEA": float(stats["RMSEA"].iloc[0]),
        "SRMR": float(np.sqrt(((empirical[iu] - implied[iu]) ** 2).mean())),
        "BIC": float(chi2 + n_par * np.log(len(X))),
        "n_free_parameters": n_par,
    }
    if pop is not None:
        criteria["sigma_max_abs_deviation"] = float(np.abs(implied[iu] - pop[iu]).max())
        criteria["sigma_rms_deviation"] = float(np.sqrt(((implied[iu] - pop[iu]) ** 2).mean()))
    # Matched to the precision the reference criteria are stored at, so both
    # sides of every comparison are measured the same way.
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in criteria.items()}, model


# --------------------------------------------------------------------------
# Tier 1 - constraints
# --------------------------------------------------------------------------
def check_constraints(model, criteria: dict, params: dict) -> tuple[dict, list[str]]:
    ins = model.inspect(std_est=True)
    loadings = ins[ins.op == "~"]
    lstd = pd.to_numeric(loadings["Est. Std"], errors="coerce")
    variances = ins[(ins.op == "~~") & (ins.lval == ins.rval)]
    covariances = ins[(ins.op == "~~") & (ins.lval != ins.rval)]
    # In semopy both measurement loadings and structural regressions use `~`,
    # so anything observed on the right-hand side is a covariate, not a factor.
    latents = set(loadings.rval.unique()) - set(model.vars["observed"])
    factor_cov = covariances[covariances.lval.isin(latents) & covariances.rval.isin(latents)]
    std_err = pd.to_numeric(ins["Std. Err"], errors="coerce")

    phi_max = params.get("phi_max", 0.90)
    min_load = params.get("min_salient_loading", 0.30)
    min_per = params.get("min_salient_per_factor", 2)
    sign_at = params.get("sign_reversal_at", -0.10)
    max_se = params.get("max_standard_error", 10.0)

    degenerate: list[str] = []
    for factor in sorted(latents):
        sub = lstd[
            (loadings.rval.values == factor) & loadings.lval.isin(model.vars["observed"]).values
        ]
        if len(sub) and (sub.abs() >= min_load).sum() < min_per:
            degenerate.append(factor)
        if len(sub) and (sub < sign_at).any():
            degenerate.append(f"{factor}~sign_reversal")

    results = {
        "converged": True,
        "no_negative_variance": not (
            pd.to_numeric(variances["Estimate"], errors="coerce") < 0
        ).any(),
        "positive_df": criteria["df"] > 0,
        "finite_standard_errors": bool(std_err.notna().any() and std_err.dropna().lt(max_se).all()),
        "no_redundant_factor": (
            not (pd.to_numeric(factor_cov["Est. Std"], errors="coerce").abs() > phi_max).any()
            if len(factor_cov)
            else True
        ),
        "no_collapsed_factor": not degenerate,
    }
    return results, degenerate


# --------------------------------------------------------------------------
# Tier 2 - Pareto dominance over a floor
# --------------------------------------------------------------------------
def check_dominance(criteria: dict, floor: dict, spec: list[dict]) -> dict:
    out = {}
    for item in spec:
        key, direction, eps = item["key"], item["direction"], item["eps"]
        if key not in criteria or key not in floor:
            out[key] = None  # not applicable
            continue
        out[key] = (
            criteria[key] >= floor[key] - eps
            if direction == "higher"
            else criteria[key] <= floor[key] + eps
        )
    return out


# --------------------------------------------------------------------------
# Tier 3 - factual claims
# --------------------------------------------------------------------------
def _partition(mapping: dict) -> set[frozenset]:
    groups: dict[Any, set] = {}
    for item, factor in mapping.items():
        groups.setdefault(factor, set()).add(item)
    return {frozenset(g) for g in groups.values()}


def primary_assignment(model) -> dict:
    """Which factor each item belongs to.

    Read from the submitted model rather than asked for. An item that loads on
    two factors is assigned to the stronger one.
    """
    ins = model.inspect(std_est=True)
    loadings = ins[ins.op == "~"].copy()
    loadings["abs"] = pd.to_numeric(loadings["Est. Std"], errors="coerce").abs()
    strongest = loadings.loc[loadings.groupby("lval")["abs"].idxmax()]
    return {row.lval: row.rval for row in strongest.itertuples()}


def biased_items_from_fit(model, items, covariate: str = "gender") -> set:
    """Items the model says are answered differently between groups.

    The model already lets the covariate, usually gender, predict the trait. A
    direct path to an item on top of that is a claim that the item behaves
    differently for reasons the trait does not explain.
    """
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & (ins.rval == covariate) & ins.lval.isin(items)]
    return {row.lval for row in rows.itertuples()}


def latent_group_difference(model, items, covariate: str = "gender") -> float | None:
    """The largest group difference the model still puts on a trait.

    Read from the refitted model, so it reflects what the submission actually
    estimated rather than what it claimed. Returns None when the model gives
    the covariate no path to any trait, which leaves the question unanswered.
    """
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & (ins.rval == covariate) & ~ins.lval.isin(items)]
    values = pd.to_numeric(rows["Est. Std"], errors="coerce").abs().dropna()
    return float(values.max()) if len(values) else None


def factor_composition(model, items) -> dict:
    """Which items each factor covers.

    A submission can name its factors anything: they are matched to the answers
    by the items they cover, not by name.
    """
    ins = model.inspect(std_est=True)
    rows = ins[(ins.op == "~") & ins.lval.isin(items)]
    out: dict[str, set] = {}
    for row in rows.itertuples():
        out.setdefault(row.rval, set()).add(row.lval)
    return {name: frozenset(its) for name, its in out.items()}


def _match_correlations(reported, target, model, items, tol) -> bool:
    """Check reported correlations between factors, matched by items.

    reported  [[factor, factor, correlation], ...]
    target    [{"a": [items], "b": [items], "r": value}, ...]
    tol       how far a reported value may be from the answer
    """
    if not isinstance(reported, list) or not target:
        return False
    composition = factor_composition(model, items)
    want = {frozenset((frozenset(e["a"]), frozenset(e["b"]))): e["r"] for e in target}
    seen = {}
    for entry in reported:
        try:
            a, b, value = entry[0], entry[1], float(entry[2])
        except (TypeError, ValueError, IndexError):
            return False
        if a not in composition or b not in composition:
            return False
        seen[frozenset((composition[a], composition[b]))] = value
    if set(seen) != set(want):
        return False
    return all(abs(seen[k] - want[k]) <= tol for k in want)


def residual_pairs_from_fit(model, items) -> set:
    """Pairs of items the model says agree beyond the trait.

    These are the `~~` terms between two observed items: the model's claim that
    those two items agree for a reason the common factors do not explain.
    """
    ins = model.inspect(std_est=True)
    rows = ins[
        (ins.op == "~~") & (ins.lval != ins.rval) & ins.lval.isin(items) & ins.rval.isin(items)
    ]
    return {frozenset((row.lval, row.rval)) for row in rows.itertuples()}


def check_claims(
    submission: dict,
    truth: dict,
    spec: list[dict],
    n_latents: int,
    model=None,
    items: list[str] | None = None,
) -> dict:
    out: dict[str, bool | None] = {}
    for item in spec:
        key, fn = item["key"], item["fn"]
        if item.get("derive_from") == "refit_primary_loadings":
            reported = primary_assignment(model)
        elif item.get("derive_from") == "refit_residual_covariances":
            reported = residual_pairs_from_fit(model, items)
        elif item.get("derive_from") == "refit_covariate_paths":
            reported = biased_items_from_fit(model, items, item.get("covariate", "gender"))
        elif item.get("derive_from") == "refit_latent_group_difference":
            reported = latent_group_difference(model, items, item.get("covariate", "gender"))
        else:
            reported = submission.get(key)
        target = _dig(truth, item.get("truth_key", ""))

        if item.get("applicable_if") == "model_has_two_oblique_factors" and n_latents != 2:
            out[key] = None
            continue
        if reported is None and not item.get("derive_from"):
            out[key] = None if key == "factor_correlation" else False
            continue

        if fn == "score_vector":
            try:
                out[key] = all(
                    abs(float(reported[k]) - v) <= item["tol"] for k, v in target.items()
                )
            except (KeyError, TypeError, ValueError):
                out[key] = False
        elif fn == "score_scalar":
            try:
                out[key] = abs(float(reported) - float(target)) <= item["tol"]
            except (TypeError, ValueError):
                out[key] = False
        elif fn == "score_label_panel":

            def _flat(obj, prefix=""):
                flat = {}
                for k, v in (obj or {}).items():
                    if isinstance(v, dict):
                        flat.update(_flat(v, f"{prefix}{k}."))
                    else:
                        flat[f"{prefix}{k}"] = v
                return flat

            want, got = _flat(target), _flat(reported)
            out[key] = bool(want) and want == got
        elif fn == "score_boolean_panel":
            try:
                out[key] = all(bool(reported[k]) == bool(v) for k, v in target.items())
            except (KeyError, TypeError):
                out[key] = False
        elif fn == "score_item_set":
            out[key] = set(reported) == set(target or [])
        elif fn == "score_pair_set":
            if item.get("match") == "composition":
                # Pairs of factors, named by the submission and matched to the
                # truth by which items each one covers.
                composition = factor_composition(model, items)
                want = {frozenset((frozenset(a), frozenset(b))) for a, b in (target or [])}
                try:
                    got = {frozenset((composition[a], composition[b])) for a, b in (reported or [])}
                except (KeyError, TypeError, ValueError):
                    got = None
            else:
                want = {frozenset(pair) for pair in (target or [])}
                try:
                    got = (
                        reported
                        if isinstance(reported, set)
                        else {frozenset(pair) for pair in (reported or [])}
                    )
                except TypeError:
                    got = None
            out[key] = got == want
        elif fn == "score_correlations_by_composition":
            out[key] = _match_correlations(reported, target, model, items, item.get("tol", 0.06))
        elif fn == "score_partition":
            out[key] = (
                _partition(reported) == _partition(target) if isinstance(reported, dict) else False
            )
        else:
            out[key] = None
    return out


def _dig(obj: dict, dotted: str):
    for part in filter(None, dotted.split(".")):
        if not isinstance(obj, dict) or part not in obj:
            return None
        obj = obj[part]
    return obj


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def score_model_criteria(submission: str | dict, params: dict, base_dir: str | Path = ".") -> dict:
    """Score one submission.

    submission  the agent's JSON answer, or the object already parsed
    params      the task's scoring rules
    base_dir    where the dataset and answers live

    Returns the full result. score_binary is the metric.
    """
    base = Path(base_dir)
    if isinstance(submission, str):
        try:
            submission = json.loads(submission.strip().strip("`").removeprefix("json"))
        except json.JSONDecodeError:
            return _zero("submission is not valid JSON")

    truth = json.loads((base / params["truth_path"]).read_text())
    ref = truth["scoring_reference"]
    floor = ref["reference_criteria"]
    pop = np.asarray(ref["population_correlation_matrix"], dtype=float)
    items = params["items"]

    known = params["syntax_whitelist"]["items"]
    data = pd.read_csv(base / params["truth_path"].rsplit("/", 1)[0] / params["dataset"], sep="\t")
    for col, val in params.get("subset", {}).items():
        data = data[data[col].isin(val)] if isinstance(val, list) else data[data[col] == val]
    X = data[[c for c in known if c in data.columns]]
    X = X[(X != 0).all(axis=1)].astype(float)
    # A dataset may carry a recording fault that the task is about. Repairing it
    # here keeps every submission judged on the same responses, whether or not
    # the submission noticed.
    for name, total in ref.get("repairs", {}).get("reverse_scored", {}).items():
        if name in X.columns:
            X[name] = total - X[name]

    try:
        spec = validate_syntax(
            submission.get("model_syntax", ""), known, params["syntax_whitelist"]["max_factors"]
        )
        # Which variables the model analyses is itself an answer when the task
        # leaves the choice open, and the later stages assume that set.
        if set(observed_in_spec(spec, known)) != set(items):
            return _result(
                0.0,
                {"instrument": "FAIL"},
                {},
                "TIER3 instrument: the model does not analyse the " "expected set of variables",
            )
        criteria, model = evaluate_model(spec, X, items, pop)
    except InvalidSubmission as exc:
        return _zero(f"invalid specification: {exc}")
    except Exception as exc:  # noqa: BLE001
        return _zero(f"model failed to fit: {type(exc).__name__}: {exc}")

    checks: dict[str, str] = {}

    constraints, degenerate = check_constraints(
        model, criteria, _merge(params["tier_1_constraints"])
    )
    checks.update({k: _fmt(v) for k, v in constraints.items()})
    if not all(constraints.values()):
        return _result(
            0.0,
            checks,
            criteria,
            "TIER1 "
            + ",".join(k for k, v in constraints.items() if not v)
            + (f" {degenerate}" if degenerate else ""),
        )

    dominance = check_dominance(criteria, floor, params["tier_2_comparative"])
    checks.update({k: _fmt(v) for k, v in dominance.items()})
    if not all(v for v in dominance.values() if v is not None):
        return _result(
            0.0,
            checks,
            criteria,
            "TIER2 " + ",".join(k for k, v in dominance.items() if v is False),
        )

    n_latents = len({ln.split("=~")[0].strip() for ln in spec.splitlines() if "=~" in ln})
    claims = check_claims(
        submission, truth, params["tier_3_claims"], n_latents, model=model, items=items
    )
    checks.update({k: _fmt(v) for k, v in claims.items()})
    if not all(v for v in claims.values() if v is not None):
        return _result(
            0.0, checks, criteria, "TIER3 " + ",".join(k for k, v in claims.items() if v is False)
        )

    return _result(1.0, checks, criteria, "all tiers pass", _recorded(criteria))


def _merge(specs: list[dict]) -> dict:
    out: dict = {}
    for s in specs:
        out.update({k: v for k, v in s.items() if k != "key"})
    return out


def _fmt(v) -> str:
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def _recorded(criteria: dict) -> dict:
    """Extra numbers kept for later analysis. Never affects the score."""
    from scipy.stats import chi2 as chi2_dist

    record = {}
    if criteria.get("df", 0) > 0:
        record["chi_square_p"] = float(1 - chi2_dist.cdf(criteria["chi2"], criteria["df"]))
        record["chi_square_df"] = criteria["df"]
    return record


def _result(
    score: float, checks: dict, criteria: dict, reason: str, recorded: dict | None = None
) -> dict:
    applicable = [v for v in checks.values() if v != "n/a"]
    return {
        "score_binary": score,
        "score_partial": (
            sum(v == "PASS" for v in applicable) / len(applicable) if applicable else 0.0
        ),
        "checks_vector": checks,
        "criteria": criteria,
        "reason": reason,
        "recorded": recorded or {},
    }


def _zero(reason: str) -> dict:
    return {
        "score_binary": 0.0,
        "score_partial": 0.0,
        "checks_vector": {},
        "criteria": {},
        "reason": reason,
    }
