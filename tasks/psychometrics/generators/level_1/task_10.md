# Level 1 / Task 10 — Which items are bad, and which is the data?

**Ask.** Select US participants and classify every HSNS item as `sound`,
`mis_keyed`, `missing_as_neutral`, `truncated_scale` or `weak_item`.

**Generator.** `gen_l1_t10_item_integrity.py`

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t10_item_integrity.py [--verify|--naive]
```

## Generating model

One trait behind ten ordinal items, loadings .48–.70. Three recording faults
from the survey platform, and one item that is simply poor:

| item | fault | generated | as stored | after recode | max | excess P(3) |
|---|---|---|---|---|---|---|
| HSNS4 | stored with its scale reversed | .62 | **−.581** | .581 | 5 | +.003 |
| HSNS7 | 45% of non-responses written as the midpoint | .66 | .444 | .444 | 5 | **+.383** |
| HSNS2 | the 5 was never recorded | .70 | .630 | .630 | **4** | +.001 |
| HSNS6 | none — the item measures poorly | .48 | .430 | .430 | 5 | +.001 |
| others | — | .59–.68 | .558–.651 | | 5 | ±.005 |

Outside the United States every item is measured worse and HSNS9 runs backwards
(λ = −.45), as a mistranslated item would.

## Why it is not a one-liner

- **Only one of the four faults is visible in the model.** The mis-keyed item
  announces itself with a loading of −.581. The other three do not.
- **The two items that matter are a hundredth apart.** HSNS6 loads .430 and
  HSNS7 loads .444 — no fit statistic, residual or modification can separate
  them, and they need opposite treatment. One is a bad item; the other is a
  good item whose non-responses were written as 3. The only thing that
  distinguishes them is that HSNS7 is answered 3 thirty-eight percentage points
  more often than a normal trait allows, while every other item is within ±.005.
- **The truncated item looks healthy.** It loads .630, inside the sound range
  of .558–.651. Nothing but its maximum value gives it away.
- **Reading the loadings misclassifies two of ten items** — enough to fail,
  since every item must be right.
- **Skipping the US filter makes a sound item the weakest.** Pooled, HSNS9
  drops to .249, well below the genuinely weak item.

## Scored

Three stages, all must pass; see `psychometrics/score.py`. Stage 3 is the
ten-item classification.

The scorer undoes the reversal on HSNS4 before fitting, so every submission is
judged on the same responses whether or not it noticed. Without that, a loading
of −.581 would trip the sign-reversal constraint and stage 1 would reject
everyone, including the correct answer.

Submission: `model_syntax`, `item_quality`.
