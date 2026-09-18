# Task 07 — which questionnaire travels to other countries?

## The idea

Both questionnaires are calibrated on the US sample and then tested in six
other countries. Each one in each country is sorted into one of four classes,
from holding exactly to not holding at all.

## What goes wrong

- The questionnaire that measures better at home is the one that does not
  transfer, and it does not announce itself. The US model keeps fitting those
  countries perfectly well; it has simply stopped being the best model there.
  Judging by fit alone gets most of the grid wrong.
- One country's answers are strong but its two traits have merged into one,
  which is a different kind of failure from measuring badly.
- Another country's items have moved between subscales, so the questionnaire
  measures something, just not what it does elsewhere.
- Sample sizes vary from 510 to 6,200, so the same real difference is
  statistically obvious in one country and invisible in another.

## Scoring

Stage 3 checks the whole grid of twelve verdicts at once. Every cell has to be
right, so there is no credit for getting the easy countries.

The submission is the US model and a class for each questionnaire in each
country. Anything else the scorer needs it works out by re-fitting the
submitted model, so the agent is not asked for it.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t07_cross_country_replication.py [--verify|--naive]
```

`--verify` prints the numbers behind everything above and checks the intended
answer wins. `--naive` checks that the obvious analysis fails.
