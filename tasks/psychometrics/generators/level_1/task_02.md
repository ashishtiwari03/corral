# Level 1 / Task 02 — Dirty Dozen factor structure

**Ask.** Select US participants, identify plausible models for the Dirty Dozen,
fit and compare them, submit the best one.

**Generator.** `gen_l1_t02_dd_structure.py`

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t02_dd_structure.py [--verify|--naive]
```

## Generating model

A general factor plus three orthogonal specific factors, over 12 ordinal items
(N = 27,000 US). Every item measures a broad antagonistic disposition and, on
top of that, the narrower trait its subscale is named for.

| item block | general loadings | specific loadings |
|---|---|---|
| DDM1–4 | .58 .52 .48 .60 | .48 .44 .42 .50 |
| DDP1–4 | .55 .50 .53 .38 | .52 .48 .55 .35 |
| DDN1–4 | .42 .40 .50 .52 | .58 .56 .45 .30 |

Non-US respondents have **no general factor** — three merely correlated, weakly
measured traits (loadings ≈ .50, φ .20–.35). The HSNS block is present but is
not part of this task.

## Why it is not a one-liner

| model | df | CFI | RMSEA | BIC | Σ dev. | outcome |
|---|---|---|---|---|---|---|
| unidimensional | 54 | .743 | .117 | 17642 | .248 | clearly rejected |
| three_correlated_factors | 51 | **.991** | **.023** | 939 | .060 | passes every cutoff, and is wrong |
| second_order | 51 | .991 | .023 | 939 | .060 | **equivalent to the row above** |
| **bifactor** | 42 | .9995 | .006 | 438 | .012 | **the generating model** |

- The nominal three-subscale structure — the one the instrument was published
  with, and the one almost every study fits — returns CFI .991 and RMSEA .023.
  Nothing in its own fit table says anything is missing. The evidence only
  appears in the *comparison*: BIC 939 vs 438, Σ deviation .060 vs .012.
- `second_order` is not merely similar to `three_correlated_factors`, it is the
  same model. With exactly three first-order factors the second-order structure
  imposes no constraint, so no fit index can ever separate them. Both receive an
  identical verdict.
- This task inverts Task 01. There, the best-fitting model was an illusion and
  the error was over-fitting. Here the error is under-fitting, so an agent that
  generalises "prefer the simpler model" from Task 01 fails.
- Skipping the US filter shifts the reported loadings by up to .114.

## Scored

Three stages, all must pass; see `psychometrics/score.py`. Stages 1 and 2 are
identical to every other model-discovery task. Stage 3 checks the reported
loadings (±.08) and the item assignment derived from the re-fit.

`factor_correlation` is not asked for: the generating model's factors are
orthogonal, so the check reports `n/a` and leaves the denominator.

Submission: `model_syntax`, `loadings`.
