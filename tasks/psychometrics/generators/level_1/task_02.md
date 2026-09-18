# Task 02 — how many traits does the Dirty Dozen measure?

## The task

Task is to identify the theoretically plausible psychometric models for the Dirty Dozen. Select the US respondents, find the 12 Dirty Dozen items, and work out how many traits lie behind them. Submit that model and its loadings.

## The data

The 12 Dirty Dozen items were drawn from one broad trait plus a narrow trait per subscale. Every item carries both:

| | broad loadings | narrow loadings |
|---|---|---|
| Machiavellianism | .58 .52 .48 .60 | .48 .44 .42 .50 |
| psychopathy | .55 .50 .53 .38 | .52 .48 .55 .35 |
| narcissism | .42 .40 .50 .52 | .58 .56 .45 .30 |

Put in deliberately: non-US respondents with **no broad trait at all**, only three related traits (loadings ≈.50, correlated .20–.35), so pooling the countries recovers neither picture.

## The answer

One broad trait that every item reflects, plus a narrower trait for each of the three subscales. Each item therefore reflects two things at once.

## Traps

- **The published structure passes every cutoff.** The three subscales the questionnaire was published with, and that nearly every study fits, look fine on their own. We need to compare more than fit metrics to compare models.
- **Two candidate models are the same model.** Writing the three subscales as sharing one higher-level trait is mathematically identical to three related traits when there are exactly three. No fit measure can ever separate them,
  so both get the same verdict.
- **The wrong sample shifts the answer.** Using every country moves the reported loadings by up to .11.

## Scoring

Stage 2 does the work: the published structure is a perfectly legitimate model, so it survives stage 1, and only the comparison against the true model rejects it. Stage 3 checks the loadings and which items group together.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t02_dd_structure.py [--verify|--naive]
```

`--verify` prints the numbers behind all of the above. `--naive` shows the
obvious analysis failing.
