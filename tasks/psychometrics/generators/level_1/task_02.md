# Task 02 — how many traits does the Dirty Dozen measure?

## The idea

The 12 Dirty Dozen answers come from one broad trait shared by every item,
plus a narrower trait for each of the three subscales. So each item reflects
two things at once.

## What goes wrong

- The three subscales the questionnaire was published with, and that nearly
  every study fits, pass every conventional cutoff. Nothing in that model's own
  output suggests anything is missing. Only comparing it against the true model
  shows the gap.
- Writing the three subscales as sharing a single higher-level trait produces
  the *same model in different words* when there are exactly three subscales.
  No fit measure can ever separate the two, so both get the same verdict.
- This task inverts Task 01. There the best-fitting model was the wrong
  answer; here the error is a model that is too simple. An agent that
  generalises a rule from one will fail the other.
- Analysing all countries together shifts the reported loadings by up to .11.

## Scoring

Stage 2 does the work here: the published structure is a legitimate model, so
it clears stage 1, and the comparison against the true model is what rejects it.
Stage 3 checks the reported loadings and which items group together.

The submission is the model and its loadings. Anything else the scorer needs
it works out by re-fitting the submitted model, so the agent is not asked for
it.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t02_dd_structure.py [--verify|--naive]
```

`--verify` prints the numbers behind everything above and checks the intended
answer wins. `--naive` checks that the obvious analysis fails.
