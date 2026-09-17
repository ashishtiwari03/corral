# Level 1 / Task 03 — HSNS: one trait, or two?

**Ask.** Identical to Task 01's, word for word. Same instrument, same question,
different data, different answer.

**Generator.** `gen_l1_t03_local_dependence.py`

```bash
uv run --with numpy --with pandas --with scipy --with semopy --with factor_analyzer \
  python generators/level_1/gen_l1_t03_local_dependence.py [--verify|--naive]
```

## Generating model

**One** trait behind all ten items. Two pairs of items additionally share a
component unrelated to the trait — respondents answer them alike because the
items are near-paraphrases:

| pair | shared | wording |
|---|---|---|
| HSNS2 ↔ HSNS7 | .38 | both about "the remarks of others" |
| HSNS5 ↔ HSNS10 | .33 | both about other people's troubles |

Loadings run .49–.82. HSNS1 and HSNS8 are set high (.82, .80) on purpose — see
below. Non-US respondents are measured less well and have no such dependence.

## The decoy

**HSNS1** ("absorbed in thinking about my personal affairs") and **HSNS8**
("wrapped up in my own interests") are the most similar-reading pair in the
instrument and share **nothing**. Nothing is done to create the decoy; it exists
because the item text is real. Their high loadings make them the second most
correlated pair in the matrix anyway:

```
top raw correlations:  HSNS2-HSNS7 0.680   HSNS1-HSNS8 0.581   HSNS5-HSNS10 0.533
residual correlations: HSNS2-HSNS7 +0.264  HSNS5-HSNS10 +0.282   HSNS1-HSNS8 +0.029
```

Rank the raw correlations and you free one right pair, the decoy, and miss a
real one. Look at what is left over once the single trait is accounted for and
the answer is unambiguous — a tenfold gap.

## Why it is not a one-liner

| model | CFI | BIC | Σ dev. | outcome |
|---|---|---|---|---|
| unidimensional | .835 | 10213 | .282 | misses the duplication |
| two_correlated_factors | .870 | 8103 | .222 | what parallel analysis suggests |
| bifactor | .941 | 3925 | .244 | specific factor collapses |
| **unidim + the 2 real pairs** | .9998 | 265 | .010 | **the generating model** |
| unidim + decoy also freed | .9998 | 279 | .012 | fits fine, wrong pair set |
| unidim + top-2 raw correlations | .950 | 4067 | .263 | misses HSNS5–HSNS10 |

Parallel analysis on the US sample reports **two factors** — the duplication is
enough to push a second eigenvalue past the threshold. Skipping the US filter
shifts the reported loadings by up to .125.

## Scored

Three stages; see `psychometrics/score.py`. Note where the work happens compared
with Task 01: there the rivals die in stage 1 on admissibility, here they die in
stage 2 on fit, and the decoy survives both and dies in stage 3.

Stage 3 adds one claim, **which item pairs agree beyond the trait**, compared as
an unordered set. It is read off the `~~` terms of the re-fitted model rather
than asked for, so the submission keeps the same fields as every other task.

| submission | derived pairs | result |
|---|---|---|
| the 2 real pairs | {(2,7), (5,10)} | pass |
| decoy freed too | {(2,7), (5,10), (1,8)} | fail |
| top-2 raw correlations | {(2,7), (1,8)} | fail |
| none | {} | fail |
