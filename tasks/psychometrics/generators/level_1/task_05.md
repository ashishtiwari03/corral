# Task 05 — which comparisons between men and women can be defended?

## The idea

The items relate to the trait identically in both groups. That makes several
comparisons sound: the structure itself, how much the trait varies, and how it
relates to other traits. But six of the ten items are answered differently by
men and women at the same trait level, in both directions.

## What goes wrong

- Because the shifts run both ways and no item is known to be clean, there is
  no fixed point to anchor a comparison of averages. The group means are the
  one comparison everybody wants and the one that cannot be made.
- Comparing the raw totals looks fine and is not: the totals differ because of
  how the items behave, not because of the trait.
- The mixed directions roughly cancel in the totals, so the bias does not
  announce itself as a large obvious gap.

## Scoring

Stage 3 checks a yes-or-no verdict on each kind of comparison, plus the
reported correlation between the two traits in each group, which is one of the
comparisons that does hold.

The submission is the model, a verdict per comparison, and the trait
correlation in each group. Anything else the scorer needs it works out by re-
fitting the submitted model, so the agent is not asked for it.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t05_defensible_comparisons.py [--verify|--naive]
```

`--verify` prints the numbers behind everything above and checks the intended
answer wins. `--naive` checks that the obvious analysis fails.
