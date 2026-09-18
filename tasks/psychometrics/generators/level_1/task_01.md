# Task 01 — how many traits does the HSNS measure?

## The idea

The 10 HSNS answers come from two related traits: self-absorption and
sensitivity to others. The traits correlate .35, so the right model keeps them
apart but linked. The file also holds the Dark Triad items and respondents from
other countries, neither of which belongs in this analysis.

## What goes wrong

- Two rival models fit **better** than the true one on every fit measure, and
  both are unusable. One splits the items three ways but leaves two of the
  three so alike (correlated .90) that they cannot be told apart. The other
  finds a trait that almost no item actually tracks. No fit index reports
  either problem, so ranking models by fit picks a broken one.
- The formal test of whether a model matches the data rejects the true model
  too. With 27,000 respondents any imperfection is detectable, so a model can
  be the best available and still fail that test.
- One item tracks its trait weakly (.30), which invites deleting it or
  inventing a third trait for it.
- Analysing all countries together, rather than the US alone, shifts the
  reported correlation between the traits from .37 to .45.

## Scoring

Stage 1 rejects both rival models before any comparison: one for having two
traits that cannot be told apart, the other for a trait no item tracks. Stage 2
allows anything that fits at least as well as the true model, so adding the
small extras the data really contain scores full marks. Stage 3 checks the
reported loadings and trait correlation against the values the data were built
from.

The submission is the model, its loadings, and the correlation between the two
traits. Anything else the scorer needs it works out by re-fitting the
submitted model, so the agent is not asked for it.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t01_hsns_structure.py [--verify|--naive]
```

`--verify` prints the numbers behind everything above and checks the intended
answer wins. `--naive` checks that the obvious analysis fails.
