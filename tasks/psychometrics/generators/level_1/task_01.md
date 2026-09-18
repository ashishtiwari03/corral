# Task 01 — how many traits does the HSNS measure?

## The task

The task is to Identify the theoretically plausible psychometric models for the Hypersensitive Narcissism Scale (HSNS). Select the respondents from the United States, find the 10 HSNS items, and work out how many traits lie behind them and which item belongs to which.
Submit that model, its loadings, and the correlation between the traits. A loading is how strongly an item tracks its trait, from 0 to 1.

## The data

The 10 HSNS items were drawn from two traits correlated **.35**:

| trait | items | loadings |
|---|---|---|
| self-absorption | 1, 4, 5, 6, 8, 10 | .55 **.30** .70 .48 .71 .66 |
| sensitivity to others | 2, 3, 7, 9 | .76 .58 .69 .52 |

Put in deliberately: a .15 second loading of item 9 on self-absorption and .10 of shared agreement between items 5 and 10; item 4's weak .30 loading.
The non-US respondents drawn from a weaker population (loadings ×0.72, traits correlated .68).

## The answer

Two related traits, self-absorption and sensitivity to others, correlated .35.
The data hold two small extra effects on top, so the true model is a close approximation.
The Dark Triad items and the respondents from other countries belong to neither trait and should be left out.

## Traps 

- **Better-fitting models are broken.** Two rival models beat the true one on every fit measure. One leaves two traits correlated .90, too alike to be separate things; the other finds a trait almost no item tracks. We need to compare more than fit metrics to compare models.
- **A weak item can be misleading.** One item has a loading of only .30 on its trait. This may tempt an analyst to delete the item or assign it to a third trait.
- **The wrong sample shifts the answer.** Using every country rather than the US alone moves the reported trait correlation from .37 to .45.

## Scoring

Stage 1 throws out both rivals before any comparison, one for traits too alike and one for the empty trait. 
Stage 2 accepts anything that fits at least as well as the true model, so a submission that also captures the small extra effects scores full marks. 
Stage 3 checks the reported loadings and trait correlation against the values the data were built from.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t01_hsns_structure.py [--verify|--naive]
```

`--verify` prints the numbers behind all of the above. `--naive` shows the
obvious analysis failing.
