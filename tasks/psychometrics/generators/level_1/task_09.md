# Level 1 / Task 09 — How strongly are the two instruments related?

**Ask.** Select US participants, model each instrument, and report how each
dimension of one relates to each dimension of the other, net of measurement
error.

**Generator.** `gen_l1_t09_careless_responding.py`

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t09_careless_responding.py [--verify|--naive]
```

## Generating model

Five correlated dimensions over 22 ordinal items: egocentrism and vulnerability
from the HSNS, three from the Dirty Dozen. Within an instrument the dimensions
are moderately related (φ .30–.45); **across the two they are barely related at
all** (φ .12–.30), and that is the answer.

Then 8% of respondents are replaced by straight-liners, who give one answer to
all 22 items, and a further 4% by random clicking. Both rates are ordinary for
web panels. Outside the United States the instruments genuinely are strongly
related (cross-instrument φ .45), so pooling inflates the same correlations a
second time.

## The anomaly

A straight-liner contributes a row where every item has the same value. Across
a group of them who each picked a different constant, every pair of items moves
in lockstep. Mixing 8% of those into the sample pulls **every** correlation in
the matrix upward, and because the inflation is roughly uniform it barely dents
a loading of .70 while more than doubling a factor correlation of .12.

The damage therefore lands precisely where the answer lives.

| pair | truth | unscreened | screened on `accuracy` | screened on response spread |
|---|---|---|---|---|
| egocentrism × narcissism | .299 | .434 | .434 | .307 |
| vulnerability × psychopathy | .123 | .262 | .263 | .122 |
| **largest error** | — | **.146** | **.146** | **.011** |

## Why it is not a one-liner

- **Nothing in the model output says anything is wrong.** The correct
  five-dimension model fits the contaminated data at CFI .979, RMSEA .023.
  There is no misfit to diagnose and no residual to chase. An agent that works
  only from model output cannot find this.
- **The obvious screen is the decoy.** The file carries `accuracy`, each
  respondent's own rating of how accurately they answered. Screening on it
  removes **39 of 1,655** straight-liners and reproduces the contaminated
  answer to three decimals. People who do not read the questions do not
  accurately report whether they read the questions.
- **The right screen is at the row level.** A respondent's spread across the 22
  items is exactly zero only if they never varied, and the items are
  deliberately heterogeneous — hurt feelings alongside exploiting people — so
  no honest respondent does that. It removes every straight-liner and no one
  else.
- **Random clicking survives and does not matter.** Those respondents' answers
  do vary, so no spread test finds them; with the straight-liners gone the
  correlations are already back within .011 of truth.
- **Skipping the US filter is wrong even after cleaning**, because the
  instruments really are related elsewhere.

## Scored

Three stages, all must pass; see `psychometrics/score.py`. Stage 3 checks the
six cross-instrument correlations against a large clean draw, ±0.06, matched to
the truth by which items each dimension covers so the agent may name its
dimensions freely.

Stages 1 and 2 do no work here, and deliberately so: the contamination leaves
the structure intact, so an agent that never cleaned submits the same
`model_syntax` and clears both. The task rests entirely on the reported
correlations.

Submission: `model_syntax`, `correlations`.
