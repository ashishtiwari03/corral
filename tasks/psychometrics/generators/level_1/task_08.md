# Level 1 / Task 08 — What may each instrument's scores be used for?

**Ask.** Select US participants, model each instrument, and say what each one
entitles you to score: `total_only`, `subscales_only`, `total_and_subscales`
or `none`.

**Generator.** `gen_l1_t08_score_justification.py`

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t08_score_justification.py [--verify|--naive]
```

## Generating model

| | structure | α | ω_h | ω_total | answer |
|---|---|---|---|---|---|
| HSNS, 10 items | one factor, λ .40–.52, plus one residual pair | .699 | **.729** | .729 | `total_only` |
| Dirty Dozen, 12 items | general factor λ .22–.46 under three specifics λ .62–.80 | **.781** | **.379** | .909 | `subscales_only` |

Outside the United States the HSNS splits into two weakly correlated factors
(λ .60, φ .25) and the Dirty Dozen collapses onto a single dominant general
factor (λ_g .70, λ_s .25).

## Why it is not a one-liner

- **α ranks the instruments backwards.** By the coefficient almost every paper
  reports, the Dirty Dozen is the better-measured instrument, .781 against .699.
  Its total score is the one that cannot be interpreted: only 38% of that
  score's variance comes from anything the twelve items share, and the rest is
  three distinguishable traits a single sum silently blends. The HSNS has the
  worse α and the sound total. Both the ranking and the size of the gap invert.
- **The Dirty Dozen total is genuinely reliable**, ω_total .909. Reaching for ω
  instead of α is the textbook correction and it still passes the total. Only
  ω_h separates them, so reliability and interpretability have to come apart in
  the analyst's hands.
- **`total_and_subscales` is the tempting middle.** The subscales are excellent
  (ω_total .876–.880, of which .69–.73 is specific variance) and the total looks
  fine by α, so keeping both is the natural compromise. It is wrong.
- **The model gate does real work.** Three correlated factors reaches CFI .986
  and would lead to the right verdict, but BIC 1634 against the bifactor's 487
  rejects it at stage 2. The general-to-specific ratio varies within each
  subscale, which is what stops the bifactor being a relabelling of three
  correlated factors.
- **A second HSNS factor is redundant, not merely unhelpful.** It fits slightly
  better than one factor — the residual pair pulls that way — and returns
  φ = .956, so stage 1 rejects it.
- **Skipping the US filter breaks both verdicts.** Pooled, the HSNS stops
  looking unidimensional (CFI .989 → .894) and the Dirty Dozen's general factor
  becomes substantial (ω_h .379 → .621, α .781 → .844).

## Scored

Three stages, all must pass; see `psychometrics/score.py`. Stage 3 checks the
two labels and the item partition derived from the re-fit.

Neither α nor ω is asked for. Knowing which coefficient answers the question is
the task.

Submission: `model_syntax`, `scoring`.
