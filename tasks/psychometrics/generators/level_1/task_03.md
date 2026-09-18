# Task 03 — does the HSNS measure one trait or two?

## The task

Task is to identify how many traits lie behind HSNS. Select the US respondents, find the 10 HSNS items, and decide how many traits lie behind them. Submit that model and its loadings.

## The data

The 10 HSNS items were drawn from a single trait, loadings .49 to .82.

Put in deliberately:

- two pairs that agree beyond the trait: items 2 and 7 (.38) and items 5 and 10 (.33), both near-paraphrases in wording
- a decoy pair, items 1 and 8, given the two highest loadings (.82 and .80) and no extra agreement at all
- non-US respondents measured worse (loadings ×0.75) and with no such pairs

## The answer

A single trait. Two pairs of items are worded so alike that people answer them alike for reasons the trait does not explain, and that extra agreement is enough to make the usual check report two traits.

## Traps

- **The standard method gives the wrong count.** Counting traits the usual way says two. The answer is one trait plus two pairs of near-duplicate items.
- **A third pair is a decoy.** It reads more alike than either real pair and has no extra agreement at all. Sorting the raw correlations puts it near the top, because both its items simply track the trait strongly.
- **Deleting items looks like a fix.** Dropping one item from each duplicated pair also removes the apparent second trait, so it seems to work while throwing away good items.

## Scoring

Stage 3 checks the loadings and which pairs of items agree beyond the trait. Those pairs are read out of the submitted model rather than asked for, so a submission that gets the structure right gets them right for free.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t03_local_dependence.py [--verify|--naive]
```

`--verify` prints the numbers behind all of the above. `--naive` shows the
obvious analysis failing.
