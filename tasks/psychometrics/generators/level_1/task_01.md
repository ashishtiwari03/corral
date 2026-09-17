# Level 1 / Task 01 — HSNS factor structure

**Ask.** Select US participants, identify plausible models for the HSNS,
fit and compare them, submit the best one.

**Generator.** `gen_l1_t01_hsns_structure.py` — simulates the survey, fits the
scoring reference, writes `artifacts/level_1/task_01/` and the task JSON.

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t01_hsns_structure.py [--verify|--naive]
```

## Generating model

Two correlated factors, φ = .35, over 10 ordinal items (N = 27,000 US, 43,710 total).

| factor | items | loadings |
|---|---|---|
| egocentrism | HSNS1, 4, 5, 6, 8, 10 | .55, **.30**, .70, .48, .71, .66 |
| oversensitivity | HSNS2, 3, 7, 9 | .76, .58, .69, .52 |

Plus two small terms that keep the two-factor model an *approximation* of the
population rather than an exact reproduction: a .15 cross-loading of HSNS9 on
egocentrism, and a .10 residual correlation between HSNS5 and HSNS10.

Also generated, not part of this task: a Dirty Dozen block (three correlated
factors), non-US participants from a weaker population (loadings ×0.72, φ = .68),
and a small gender threshold shift on HSNS3 and HSNS8 for Level 2.

## Why it is not a one-liner

Fitting every candidate and ranking by fit gives the wrong answer:

| model | CFI | RMSEA | BIC | Σ dev. | outcome |
|---|---|---|---|---|---|
| unidimensional | .7045 | .1213 | 12553 | .334 | clearly rejected |
| two_orthogonal_factors | .9414 | .0540 | 2679 | .208 | worse on everything |
| **two_correlated_factors** | .9868 | .0260 | 796 | .074 | **the generating model** |
| three_correlated_factors | .9919 | .0210 | 602 | .065 | better fit, φ = .904 |
| bifactor | .9994 | .0067 | 354 | .009 | best fit, specific factor collapses |

- χ² rejects the generating model at p < .0001 (CFI = .987). Selecting on the χ²
  p-value discards the right answer.
- Two rivals beat it on **every** fit index. Both are inadmissible — one has a
  factor correlation of .904, the other a specific factor with loadings of .02,
  .00 and two sign reversals. Fit alone cannot decide this.
- HSNS4 loads .30, which tempts a third factor or item deletion.
- Skipping the US filter shifts the reported φ from .374 to .446.

## Scored

Three stages, all must pass; see `psychometrics/score.py`.

1. **Constraints** — converged, no negative variance, φ ≤ .90, no collapsed
   factor, identified. Rejects the two trap models before any comparison.
2. **Comparative** — Pareto dominance over the generating model as a floor:
   CFI, RMSEA, SRMR, BIC, Σ deviation. Beating it is accepted; a model adding
   the cross-loading, or the cross-loading and the residual correlation, scores
   full marks.
3. **Claims** — the reported loadings (±.08) and factor correlation (±.06), plus
   the item assignment derived from the re-fit.

The submission is three fields: `model_syntax`, `loadings`, `factor_correlation`.
Everything else the scorer needs it computes by re-fitting the specification.
The reported estimates come from the agent's own analysis sample, which is what
makes the US filtering step consequential: fitting the pooled sample yields
φ = .446 and loadings out by up to .119, both outside tolerance.

Recorded but never scored: χ² p-value and df.
