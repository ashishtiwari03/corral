# Task 03 — does the HSNS measure one trait or two?

## The idea

Every one of the 10 items reflects a single trait. But two pairs are worded
so alike that people answer them alike for reasons the trait does not explain.
That extra agreement is enough to make the usual check report two traits where
there is one.

## What goes wrong

- The standard way of counting traits says two. The answer is one trait plus
  two pairs of near-duplicate items.
- A third pair reads more alike than either real pair and has no extra
  agreement at all. Sorting the raw correlations puts it near the top, because
  both its items simply track the trait strongly. It is a decoy.
- Deleting the duplicated items also removes the apparent second trait, so it
  looks like a fix while throwing away good items.

## Scoring

Stage 3 checks the loadings and which pairs of items agree beyond the trait.
The pairs are read from the submitted model rather than asked for, so a
submission that gets the structure right gets them right for free.

The submission is the model, its loadings, and a factor correlation, which
is reported as not applicable when the answer is a single trait. Anything else the scorer needs
it works out by re-fitting the submitted model, so the agent is not asked for
it.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t03_local_dependence.py [--verify|--naive]
```

`--verify` prints the numbers behind everything above and checks the intended
answer wins. `--naive` checks that the obvious analysis fails.
