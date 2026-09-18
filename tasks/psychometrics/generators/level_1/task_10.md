# Task 10 — which items are bad, and which is the data?

## The idea

Four of the ten HSNS items misbehave. Three are good items whose recorded
answers were damaged somewhere between the respondent and the file; only one is
genuinely a poor item. The task is to say which is which.

## What goes wrong

- One item was stored with its scale flipped. It is the only fault the model
  output shows, and the tempting response, dropping it, is wrong: it is a good
  item and recoding restores it.
- For another item, people who declined to answer were recorded as having
  picked the middle option. There is now no way to tell a real neutral from a
  refusal.
- A third item's top answer was never recorded, so its scale runs 1 to 4 while
  every other item runs 1 to 5. It still looks healthy in the model, better
  than several genuinely sound items. Only its range gives it away.
- The two items that need opposite treatment differ by a hundredth. The
  damaged item and the genuinely weak one have all but identical loadings, so
  no fit measure, residual or model comparison can separate them. What
  separates them is that the damaged item is answered with the middle option
  far more often than any real trait would produce.
- Analysing all countries together makes a perfectly sound item look like the
  weakest of the ten.

## Scoring

Stage 3 checks all ten verdicts at once, so every item has to be right. The
scorer undoes the flipped item before fitting; otherwise the reversed loading
would trip a stage 1 constraint and reject every submission, including the
correct one.

The submission is the model and a verdict per item. Anything else the scorer
needs it works out by re-fitting the submitted model, so the agent is not
asked for it.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t10_item_integrity.py [--verify|--naive]
```

`--verify` prints the numbers behind everything above and checks the intended
answer wins. `--naive` checks that the obvious analysis fails.
