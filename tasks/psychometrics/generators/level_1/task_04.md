# Task 04 — which questionnaire supports a comparison between men and women?

## The idea

Both questionnaires were given to everyone, and the question is which one can
be used to compare the sexes. The HSNS cannot: four of its items are answered
differently by men and women who have the same amount of the trait. The Dirty
Dozen can, and shows women slightly higher on its broad trait.

## What goes wrong

- The Dirty Dozen only looks usable once it is modelled correctly. Fitted with
  its published subscale structure it appears unfair on three items that are
  perfectly fair. So an agent that takes each questionnaire at face value
  concludes neither can be used.
- Testing one item at a time finds bias almost everywhere, because the
  comparison baseline is itself contaminated by the bias being looked for. All
  the items have to be freed at once, after which the truly biased ones
  separate by the direction of their effect.
- Outside the US there is no item bias and the sex difference runs the other
  way, so analysing everyone together moves the answer well outside tolerance
  rather than merely diluting it.

## Scoring

Stage 3 checks the size of the sex difference, the items reported as biased,
and that the submitted model does not claim bias in the fair questionnaire.

The submission is the model, the size of the difference, and the biased items.
Anything else the scorer needs it works out by re-fitting the submitted model,
so the agent is not asked for it.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t04_invariant_combination.py [--verify|--naive]
```

`--verify` prints the numbers behind everything above and checks the intended
answer wins. `--naive` checks that the obvious analysis fails.
