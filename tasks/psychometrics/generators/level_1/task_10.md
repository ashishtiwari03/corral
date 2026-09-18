# Task 10 — which items are bad, and which is the data?

## The task

Select the US respondents and work through the 10 HSNS items one at a time,
classifying each as sound, damaged in one of three specific ways, or genuinely
a poor item.

## The data

The 10 HSNS items were drawn from one trait, loadings .48–.70.

Put in deliberately, after the answers were generated:

| item | what was done | result |
|---|---|---|
| HSNS4 | stored with its scale flipped | loading −.581 |
| HSNS7 | 45% of answers rewritten to the middle option | loading .444 |
| HSNS2 | top answer never recorded, scale capped at 4 | loading .630 |
| HSNS6 | nothing — generated weak at .48 | loading .430 |

Outside the US every item is measured worse and HSNS9 runs backwards (−.45),
as a mistranslated item would.

## The answer

Four items misbehave. Three are good items whose recorded answers were damaged
between the respondent and the file; only one is genuinely poor.

## Traps

- **Only one fault shows up in the model.** One item was stored with its scale
  flipped, which is obvious. The tempting response, dropping it, is wrong: it
  is a good item and recoding restores it.
- **One fault is invisible.** Another item's top answer was never recorded, so
  its scale runs 1 to 4 while the rest run 1 to 5. It still looks healthier
  than several genuinely sound items. Only its range gives it away.
- **Two items differ by a hundredth and need opposite treatment.** For one item,
  people who declined to answer were recorded as picking the middle option. It
  and the genuinely weak item have all but identical loadings, so no fit
  measure or model comparison can separate them. What does is that the damaged
  item is answered with the middle option far more often than any real trait
  would produce.
- **The wrong sample frames an innocent item.** Analysing every country makes a
  perfectly sound item look like the weakest of the ten.

## Scoring

Stage 3 checks all ten verdicts at once, so every item has to be right. The
scorer undoes the flipped item before fitting; otherwise its reversed loading
would trip a stage 1 constraint and reject every submission, including the
correct one.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t10_item_integrity.py [--verify|--naive]
```

`--verify` prints the numbers behind all of the above. `--naive` shows the
obvious analysis failing.
