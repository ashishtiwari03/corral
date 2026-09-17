# Level 1 / Task 06 — How do the two instruments' dimensions relate?

**Ask.** Examine how the dimensions of one instrument relate to the dimensions of
the other, accounting for the fact that no dimension is measured perfectly, and
say which pairs are too closely related to be treated as distinct.

**Generator.** `gen_l1_t06_latent_relationships.py`

## Generating model

Five correlated dimensions: two from the HSNS (vulnerability, egocentrism) and
three from the Dirty Dozen. **Egocentrism is measured poorly** (loadings .40–.48)
while vulnerability is measured well (.58–.76) — so relationships involving
egocentrism shrink furthest when the instruments are related through scale scores.

**Egocentrism and Dark-Triad narcissism correlate .86** — very nearly the same
construct.

## Why it is not a one-liner

```
pair             true  joint model  scale scores  shrinkage
VULN-MACH        0.28        0.289         0.214     -0.066
VULN-PSYCH       0.40        0.388         0.274     -0.126
VULN-NARC        0.30        0.306         0.231     -0.069
EGO-MACH         0.52        0.525         0.338     -0.182
EGO-PSYCH        0.34        0.338         0.212     -0.128
EGO-NARC         0.86        0.866         0.570     -0.290
```

- The joint latent model recovers every relationship. Scale scores miss **all six**,
  so an analysis that ignores measurement error fails on the numbers alone — no
  judgement of the method is needed.
- The shrinkage is **uneven**, from −.07 to −.29, because it depends on how well
  each dimension is measured. The result is not a shrunken picture but a
  *distorted* one: scale scores say these traits all relate at .21–.34, fairly
  uniformly, when the real range is .28 to .86.
- **The pair that is not distinguishable looks distinguishable.** EGO–NARC is .86
  in truth and .57 through scale scores, so the naive analysis licenses treating
  two near-identical dimensions as separate traits.

A ranking reversal was tried and rejected: producing one requires the true gap
between two relationships to be about .02, at which point the true ordering is
not substantively meaningful either.

## Scored

- **correlations** — all six cross-instrument relationships, ±.06.
- **not_distinguishable** — which pairs sit at or above .80. Both missing the
  redundant pair and over-flagging extra ones fail.

Dimensions are matched to the truth by **which items they cover**, so an agent
names its factors whatever it likes. The scorer reads the composition from the
re-fitted model.

Submission: `model_syntax`, `correlations`, `not_distinguishable`.
