# Task 05 — which comparisons between men and women can be defended?

## The task

Select the US respondents who reported male or female, then judge six specific comparisons of the HSNS between the two groups as defensible or not. Also report how the two traits relate within each group.

## The data

The 10 HSNS items were drawn from one trait, loadings .48–.76, identical in
both sexes. A second trait, from four Dirty Dozen items, was generated
alongside it.

Put in deliberately:

- six of the ten items shifted between the sexes, in **both** directions:
  +.40, −.45, +.35, −.50, +.45, −.40
- a real trait difference of .30 between the sexes, which those shifts make
  impossible to recover
- the two traits correlated .50 in men and .30 in women, a difference that
  **is** recoverable

## The answer

The items relate to the trait identically in both groups, so comparing the
structure, the spread of the trait, and how it relates to other traits are all
sound. Comparing the group averages is not, and neither is comparing the raw
totals.

## Traps

- **Six of ten items are answered differently.** Men and women at the same trait
  level answer them differently, and in both directions. Since no item is
  known to be clean, there is no fixed point to anchor a comparison of
  averages.
- **The comparison everyone wants is the one that fails.** Group averages are
  the usual headline result and are exactly what these data cannot support.
- **The bias does not announce itself.** Because the shifts run both ways they
  roughly cancel in the totals, so there is no large obvious gap to notice.

## Scoring

Stage 3 checks a yes-or-no verdict on each of the six comparisons, plus the
reported correlation between the two traits in each group, which is one of the
comparisons that does hold.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t05_defensible_comparisons.py [--verify|--naive]
```

`--verify` prints the numbers behind all of the above. `--naive` shows the
obvious analysis failing.
