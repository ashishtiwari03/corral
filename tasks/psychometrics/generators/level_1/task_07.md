# Level 1 / Task 07 — Which instrument travels?

**Ask.** Calibrate a measurement model for each instrument on the US sample, then
classify how far each carries over to six other countries: `exact`,
`approximate`, `substantive_only`, or `none`.

**Generator.** `gen_l1_t07_cross_country_replication.py`

## Generating model

| country | N | HSNS | Dirty Dozen |
|---|---|---|---|
| US | 24000 | two factors, φ = .35 | general factor + three specifics |
| GB | 6200 | unchanged → `exact` | unchanged → `exact` |
| CA | 3800 | loadings ×0.85 → `approximate` | general ×0.5 → `substantive_only` |
| AU | 3000 | unchanged → `exact` | general ×0.3 → `substantive_only` |
| IN | 660 | loadings ×0.85 → `approximate` | no general factor → `substantive_only` |
| BR | 510 | φ → .95 → `substantive_only` | no general factor → `substantive_only` |
| DE | 830 | loadings ×0.85 → `approximate` | scrambled, weakened → `none` |

## Why it is not a one-liner

**The instrument that measures better at home is the one that does not travel.**
In the US the Dirty Dozen's structure is beyond dispute — its general factor beats
the three-subscale alternative by 578 BIC units. It holds in exactly one other
country.

**And it never stops fitting.** Judging replication the obvious way, by asking
whether the calibrated model still fits, gets four of six countries wrong:

```
country  HSNS CFI   DD CFI   verdict from fit alone
CA         1.0014   0.9951   Dirty Dozen replicates   (truth: substantive_only)
AU         0.9971   1.0065   Dirty Dozen replicates   (truth: substantive_only)
IN         0.9936   1.0028   Dirty Dozen replicates   (truth: substantive_only)
BR         1.0103   0.9890   Dirty Dozen replicates   (truth: substantive_only)
```

The US model fits those countries perfectly well. It has simply stopped being the
*best* model there — the three-subscale structure now wins on BIC, because the
general factor is a feature of the US sample. Only a per-country model
*comparison* shows this; a per-country fit check cannot.

This is why an earlier design failed: a bifactor fitted to data with no general
factor still returns a healthy-looking general factor with excellent fit, because
the two structures are alternative parameterisations of the same shared variance.
There is nothing to see unless you compare them.

**Both columns vary,** so neither can be filled in by pattern: the HSNS is
`exact` twice, `approximate` three times and `substantive_only` once; the Dirty
Dozen is `exact` once, `substantive_only` four times and `none` once.

Every country's verdict was checked over 20 seeds and never flips; the tightest
margin is Canada's, at 21 BIC units minimum.

## Scored

- **Stages 1–2** re-fit the submitted model on the US sample. An agent that
  calibrates the wrong US model fails here and never reaches the countries —
  correctly, since replication of a model that does not hold at home is not a
  finding. This is also why `model_syntax` is required: Brazil is
  `substantive_only` under a two-factor HSNS and `exact` under a one-factor one,
  so the labels are meaningless without knowing what was tested.
- **Stage 3** checks the twelve labels. All must be right.

Nothing else is asked for. The labels cannot be produced without fitting in every
country, so no auxiliary quantity is needed to force the work — and asking for
one (a factor correlation, say) would point straight at the diagnostic that
cracks Brazil.

Submission: `model_syntax`, `replication`.
