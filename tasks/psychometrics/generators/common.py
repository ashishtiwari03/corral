"""Shared building blocks for the psychometrics task generators.

Each task generator supplies its own generating model and prompt; everything
that is the same across tasks - the response scale, the survey scaffolding, the
model-fitting used to build the scoring reference, and the artifact writing -
lives here.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PKG_ROOT = Path(__file__).resolve().parents[1]

HSNS_ITEMS = [f"HSNS{i}" for i in range(1, 11)]
DD_ITEMS = ["DDP1", "DDP2", "DDP3", "DDP4", "DDN1", "DDN2", "DDN3", "DDN4",
            "DDM1", "DDM2", "DDM3", "DDM4"]
ALL_ITEMS = HSNS_ITEMS + DD_ITEMS
COLUMNS = ALL_ITEMS + ["age", "gender", "accuracy", "country"]

# Respondents per country, following the imbalance typical of open web surveys.
N_BY_COUNTRY = {
    "US": 27000, "GB": 6200, "CA": 3800, "AU": 3000, "DE": 830,
    "IN": 660, "SE": 630, "PH": 560, "PL": 520, "BR": 510,
}
ITEM_MISSING_RATE = 0.012

# Category thresholds on the latent response variate, four per item. Chosen to
# give asymmetric, realistic-looking Likert distributions.
THRESHOLDS = {
    "HSNS1": [-1.701, -0.923, -0.487, 0.509], "HSNS2": [-0.948, -0.253, 0.109, 0.984],
    "HSNS3": [-1.351, -0.568, -0.189, 0.737], "HSNS4": [-1.057, -0.032, 0.44, 1.18],
    "HSNS5": [-1.32, -0.443, 0.013, 0.902], "HSNS6": [-2.084, -1.288, -0.752, 0.326],
    "HSNS7": [-1.451, -0.669, -0.25, 0.876], "HSNS8": [-1.247, -0.383, 0.016, 0.873],
    "HSNS9": [-1.591, -0.875, -0.445, 0.606], "HSNS10": [-0.712, 0.158, 0.538, 1.229],
    "DDP1": [-0.659, 0.026, 0.337, 1.06], "DDP2": [-0.625, 0.171, 0.554, 1.209],
    "DDP3": [-1.069, -0.35, 0.063, 0.925], "DDP4": [-1.559, -0.969, -0.507, 0.434],
    "DDN1": [-1.607, -1.037, -0.569, 0.509], "DDN2": [-1.241, -0.539, -0.008, 0.941],
    "DDN3": [-1.114, -0.371, 0.083, 0.888], "DDN4": [-0.884, 0.046, 0.526, 1.379],
    "DDM1": [-1.011, -0.296, 0.116, 1.007], "DDM2": [-1.565, -0.934, -0.591, 0.612],
    "DDM3": [-1.286, -0.523, -0.17, 0.865], "DDM4": [-0.81, 0.057, 0.58, 1.396],
}

ITEM_TEXT = {
    "HSNS1": "I can become entirely absorbed in thinking about my personal affairs, my health, my cares or my relations to others.",
    "HSNS2": "My feelings are easily hurt by ridicule or the slighting remarks of others.",
    "HSNS3": "When I enter a room I often become self conscious and feel that the eyes of others are upon me.",
    "HSNS4": "I dislike sharing the credit of an achievement with others.",
    "HSNS5": "I feel that I have enough on my hands without worrying about other people's troubles.",
    "HSNS6": "I feel that I am temperamentally different from most people.",
    "HSNS7": "I often interpret the remarks of others in a personal way.",
    "HSNS8": "I easily become wrapped up in my own interests and forget the existence of others.",
    "HSNS9": "I dislike being with a group unless I know that I am appreciated by at least one of those present.",
    "HSNS10": "I am secretly \"put out\" or annoyed when other people come to me with their troubles, asking me for my time and sympathy.",
    "DDM1": "I tend to manipulate others to get my way.",
    "DDM2": "I have used deceit or lied to get my way.",
    "DDM3": "I have used flattery to get my way.",
    "DDM4": "I tend to exploit others towards my own end.",
    "DDP1": "I tend to lack remorse.",
    "DDP2": "I tend to not be too concerned with morality or the morality of my actions.",
    "DDP3": "I tend to be callous or insensitive.",
    "DDP4": "I tend to be cynical.",
    "DDN1": "I tend to want others to admire me.",
    "DDN2": "I tend to want others to pay attention to me.",
    "DDN3": "I tend to seek prestige or status.",
    "DDN4": "I tend to expect special favors from others.",
}


# --------------------------------------------------------------------------
# Simulation
# --------------------------------------------------------------------------
def categorize(ystar, tau):
    """Cut a continuous latent response into 1-5 categories at the thresholds."""
    return np.searchsorted(np.asarray(tau), ystar).astype(int) + 1


def correlated_block(n, rng, loadings, phi_matrix, factor_names, taus,
                     scale=1.0, cross=None, resid_corr=None, dif=None, female=None):
    """Draw `n` ordinal responses from a correlated-factors model.

    Latent traits are drawn from a multivariate normal with correlation
    `phi_matrix`; each item's latent response is a loading-weighted trait plus
    normal error scaled so the response has unit variance; the result is cut at
    the item's thresholds.

    loadings    {item: (factor_name, loading)}
    scale       multiplies every loading, for generating a weaker population
    cross       (item, factor, loading) giving one item a secondary loading
    resid_corr  ((item, item), covariance) correlating two items' residuals
    dif         {item: threshold shift} applied where `female` is True
    """
    eta = rng.multivariate_normal(np.zeros(len(factor_names)), phi_matrix, size=n)
    index = {f: i for i, f in enumerate(factor_names)}
    shared = rng.normal(size=n)
    out = {}
    for item, (factor, lam) in loadings.items():
        lam *= scale
        common = lam * eta[:, index[factor]]
        explained = lam ** 2
        if cross and item == cross[0]:
            extra = cross[2] * scale
            common = common + extra * eta[:, index[cross[1]]]
            explained += extra ** 2 + 2 * extra * lam * phi_matrix[
                index[factor], index[cross[1]]]
        if resid_corr and item in resid_corr[0]:
            common = common + np.sqrt(resid_corr[1]) * shared
            explained += resid_corr[1]
        ystar = common + rng.normal(0, np.sqrt(max(1 - explained, 1e-6)), n)
        tau = taus[item]
        if dif and item in dif and female is not None:
            shifted = [t + dif[item] for t in tau]
            out[item] = np.where(female, categorize(ystar, shifted),
                                 categorize(ystar, tau))
        else:
            out[item] = categorize(ystar, tau)
    return pd.DataFrame(out)


def bifactor_block(n, rng, general, specific, specific_of, items, taus):
    """Draw `n` ordinal responses from a general factor plus orthogonal specifics.

    general      {item: loading on the general factor}
    specific     {item: loading on its specific factor}
    specific_of  {item: name of its specific factor}
    """
    g = rng.normal(size=n)
    # dict.fromkeys keeps first-appearance order; iterating a set here would
    # shuffle the draws between processes, because string hashing is randomised.
    s = {name: rng.normal(size=n) for name in dict.fromkeys(specific_of.values())}
    out = {}
    for item in items:
        gl, sl = general[item], specific[item]
        ystar = (gl * g + sl * s[specific_of[item]]
                 + rng.normal(0, np.sqrt(max(1 - gl ** 2 - sl ** 2, 1e-6)), n))
        out[item] = categorize(ystar, taus[item])
    return pd.DataFrame(out)[items]


def demographics(n, rng, country):
    """Age, gender, self-rated accuracy and country for one block of respondents."""
    return pd.DataFrame({
        "age": np.clip(rng.lognormal(np.log(22), 0.38, n).round(), 13, 89).astype(int),
        "gender": rng.choice([1, 2, 3, 0], size=n, p=[0.61, 0.37, 0.01, 0.01]),
        "accuracy": np.clip(rng.beta(6, 1.4, n) * 100, 1, 100).round().astype(int),
        "country": country,
    })


def finalize(frames, rng, seed):
    """Shuffle the country blocks together and apply item-level missingness."""
    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=seed)
    df[ALL_ITEMS] = df[ALL_ITEMS].mask(
        rng.random((len(df), len(ALL_ITEMS))) < ITEM_MISSING_RATE, 0)
    return df[COLUMNS].reset_index(drop=True)


def analysis_sample(df, items, country="US"):
    """Respondents from one country with complete responses on `items`."""
    X = df[df.country == country][items]
    return X[(X != 0).all(axis=1)].astype(float)


# --------------------------------------------------------------------------
# Model fitting
# --------------------------------------------------------------------------
def evaluate_model(spec, X, items, pop):
    """Fit one specification and return its fit, parsimony and accuracy criteria.

    `pop` is the population correlation matrix a perfectly specified model would
    reproduce; the returned sigma deviations say how close this model comes.
    """
    import semopy

    model = semopy.Model(spec)
    model.fit(X[items])
    stats = semopy.calc_stats(model)

    sigma = model.calc_sigma()[0]
    order = list(model.vars["observed"])
    pick = [order.index(i) for i in items]
    sigma = sigma[np.ix_(pick, pick)]
    scale = np.sqrt(np.diag(sigma))
    implied = sigma / np.outer(scale, scale)
    empirical = np.corrcoef(X[items].values.T)
    upper = np.triu_indices(len(items), 1)

    chi2 = float(stats["chi2"].iloc[0])
    n_par = len(model.param_vals)
    return {
        "df": float(stats["DoF"].iloc[0]),
        "chi2": chi2,
        "CFI": float(stats["CFI"].iloc[0]),
        "RMSEA": float(stats["RMSEA"].iloc[0]),
        "SRMR": float(np.sqrt(((empirical[upper] - implied[upper]) ** 2).mean())),
        "BIC": float(chi2 + n_par * np.log(len(X))),
        "n_free_parameters": n_par,
        "sigma_max_abs_deviation": float(np.abs(implied[upper] - pop[upper]).max()),
        "sigma_rms_deviation": float(
            np.sqrt(((implied[upper] - pop[upper]) ** 2).mean())),
    }


def primary_loadings_from_fit(spec, X, items):
    """Largest absolute standardised loading per item, as a fitted model gives it."""
    import semopy

    model = semopy.Model(spec)
    model.fit(X[items])
    ins = model.inspect(std_est=True)
    load = ins[ins.op == "~"].copy()
    load["abs"] = pd.to_numeric(load["Est. Std"], errors="coerce").abs()
    best = load.loc[load.groupby("lval")["abs"].idxmax()]
    return {row.lval: float(row.abs) for row in best.itertuples()}


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
def write_codebook(path, df):
    """Write the respondent-facing description of the dataset.

    Deliberately documents the items and the response scale only. Which items
    belong to which instrument, and how they group into subscales, is what the
    task asks the agent to work out.
    """
    lines = [
        "# Codebook - online personality survey",
        "",
        "Tab-separated, one row per respondent. Item responses are ratings on a "
        "five-point scale: 1 = Disagree, 3 = Neutral, 5 = Agree. **0 = missed.**",
        "",
        "Items are taken from published self-report personality scales and were "
        "presented to respondents in a single block.",
        "",
        "| item | text |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in ITEM_TEXT.items()],
        "",
        "| variable | description |",
        "|---|---|",
        "| `age` | entered as free text |",
        "| `gender` | 1 = Male, 2 = Female, 3 = Other, 0 = missed |",
        "| `accuracy` | self-rated accuracy of own responses, 0-100 |",
        "| `country` | ISO country code |",
        "",
        f"Rows: {len(df):,}. Countries: " +
        ", ".join(f"{c} ({n:,})" for c, n in df.country.value_counts().items()) + ".",
        "",
    ]
    path.write_text("\n".join(lines))


def provenance(generator, seed, rows, data_sha):
    """Record of how the dataset was produced, stored inside truth.json."""
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      cwd=PKG_ROOT, text=True).strip()
    except Exception:                                          # noqa: BLE001
        rev = "unknown"
    return {
        "generator": generator, "seed": seed,
        "generated": date.today().isoformat(), "git_rev": rev,
        "rows": rows, "data_sha256": data_sha, "external_data_used": None,
        "versions": {"python": sys.version.split()[0],
                     "numpy": np.__version__, "pandas": pd.__version__},
    }


def scoring_contract(truth_path, items, tier_3_claims):
    """The scoring block of a task JSON.

    Stages one and two are identical for every model-discovery task; only the
    claims a task asks about differ.
    """
    return {
        "truth_path": truth_path,
        "dataset": "data.csv",
        "subset": {"country": "US"},
        "items": items,
        "scorer_estimator": "ML",
        "refit_submitted_syntax": True,
        "syntax_whitelist": {"items": items, "operators": ["=~", "~~", "~"],
                             "max_factors": 6},
        "method": "constraints_then_pareto_then_claims",
        "aggregation": "all_tiers_must_pass",
        "emit": ["score_binary", "score_partial", "checks_vector"],
        # Feasibility. Violating any of these makes a model invalid rather than
        # merely worse, so it is rejected before any comparison.
        "tier_1_constraints": [
            {"key": "converged"},
            {"key": "no_negative_variance"},
            {"key": "positive_df"},
            {"key": "finite_standard_errors", "max_standard_error": 10.0},
            {"key": "no_redundant_factor", "phi_max": 0.90},
            {"key": "no_collapsed_factor", "min_salient_loading": 0.30,
             "min_salient_per_factor": 2, "sign_reversal_at": -0.10},
        ],
        # Pareto dominance over the generating model, which acts as a floor: the
        # submission must be no worse on any criterion, and may be better.
        "reference_role": "floor",
        "tier_2_comparative": [
            {"key": "CFI", "direction": "higher", "eps": 0.005,
             "criterion": "global_fit"},
            {"key": "RMSEA", "direction": "lower", "eps": 0.005,
             "criterion": "global_fit"},
            {"key": "SRMR", "direction": "lower", "eps": 0.005,
             "criterion": "global_fit"},
            {"key": "BIC", "direction": "lower", "eps": 10.0,
             "criterion": "parsimony"},
            {"key": "sigma_max_abs_deviation", "direction": "lower",
             "eps": 0.010, "criterion": "accuracy"},
        ],
        "tier_3_claims": tier_3_claims,
        "recorded_fields": ["chi_square_p", "chi_square_df"],
    }


def write_artifacts(out_dir, task_json_path, df, truth_fn, task_json_fn):
    """Write the dataset, codebook, ground truth and task definition.

    `truth_fn` and `task_json_fn` each take the dataset's SHA-256, which is only
    known once the file has been written.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    task_json_path.parent.mkdir(parents=True, exist_ok=True)

    data_path = out_dir / "data.csv"
    df.to_csv(data_path, sep="\t", index=False)
    sha = data_sha256(out_dir)
    write_codebook(out_dir / "codebook.md", df)
    (out_dir / "truth.json").write_text(json.dumps(truth_fn(sha), indent=2))
    task_json_path.write_text(json.dumps(task_json_fn(sha), indent=2))

    print(f"\n{len(df):,} rows -> {data_path}")
    for name in ("codebook.md", "truth.json"):
        print(f"           {out_dir / name}")
    print(f"           {task_json_path}")


def data_sha256(out_dir):
    """SHA-256 of the written dataset, recorded for reproducibility."""
    return hashlib.sha256((out_dir / "data.csv").read_bytes()).hexdigest()
