# Task 09 — how strongly are the two questionnaires related?

## The task

Select the US respondents, build a model for each questionnaire, and report
how strongly each trait behind one relates to each trait behind the other,
once the noise in the measurements is accounted for.

## The data

Five traits behind the two questionnaires, loadings .58–.72. Within a
questionnaire they are correlated .30 to .45; **across** the two, only .12 to
.30.

Put in deliberately:

- 8% of respondents replaced by straight-liners, who give one answer to all 22
  items, the answer weighted towards the middle of the scale
- a further 4% replaced by random clicking
- self-rated accuracy drawn identically for everyone, so it carries no signal
  about who was careless
- non-US respondents whose questionnaires really are strongly related (.45),
  so pooling inflates the same correlations a second time

## The answer

Within a questionnaire the traits are moderately related; across the two they
are barely related at all, and that is the finding. But a twelfth of the
respondents were not reading the questions: 8% gave the same answer to all 22
items and 4% answered at random.

## Traps

- **Non-readers inflate every correlation.** Someone who answers 3 to everything
  makes every pair of items move together perfectly. The rise is roughly
  uniform, so it barely dents a strong relationship while more than doubling a
  weak one. The damage lands exactly where the answer is.
- **The model gives no warning.** The correct structure still fits the
  contaminated data well, so there is no misfit to diagnose and no residual to
  chase.
- **The obvious screen is the decoy.** The file records how accurately each
  person says they answered. Screening on it removes 39 of 1,655 non-readers
  while discarding hundreds of good respondents: people who do not read the
  questions do not report that they did not read them.
- **The fix is in the raw answers.** Someone whose 22 answers never vary is not
  answering the questions. That catches every straight-liner and nobody else.
  The random responders survive any such screen and turn out not to matter.

## Scoring

Stages 1 and 2 do no work here, deliberately. The contamination leaves the
structure intact, so an agent that never cleaned the data submits the same
model and clears both. The task rests entirely on stage 3, the reported
correlations.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t09_careless_responding.py [--verify|--naive]
```

`--verify` prints the numbers behind all of the above. `--naive` shows the
obvious analysis failing.
