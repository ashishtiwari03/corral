# Task 08 — what may each questionnaire's scores be used for?

## The task

Select the US respondents, build a model for each questionnaire, and decide
what each one entitles you to add up: one overall total, separate subscale
scores, both, or neither.

## The data

- **HSNS**: one trait, even loadings .40–.52, plus a little shared agreement
  between items 2 and 7. Gives a sound total score.
- **Dirty Dozen**: a weak broad trait (loadings .22–.46) under three strong
  narrow ones (.62–.80). Gives sound subscale scores and a meaningless total.

The balance between broad and narrow varies from item to item within a
subscale. Without that variation the model would be three related traits
written a different way, and the task would have no answer.

Put in deliberately: outside the US the HSNS splits in two and the Dirty Dozen
collapses onto one strong trait, which breaks both verdicts if the countries
are pooled.

## The answer

The HSNS supports a total. The Dirty Dozen does not, and supports subscale
scores instead.

## Traps

- **The usual reliability figure ranks them backwards.** The Dirty Dozen scores
  higher on the number almost every paper reports, and its total is the one
  that means nothing: only about a third of it reflects anything its twelve
  items share, the rest blending three distinguishable traits.
- **The usual correction is not enough either.** By that measure the Dirty Dozen
  total looks excellent. Only the figure that asks how much of a total comes
  from a single common trait separates the two questionnaires.
- **The compromise answer is wrong.** Keeping both the total and the subscales
  is natural, since the subscales are genuinely good and the total looks fine.
- **The wrong sample breaks both verdicts at once.** Analysing every country
  rather than the US alone changes the answer for each questionnaire.

## Scoring

Stage 2 rejects a model of three merely related traits, which would reach the
right verdict by a route the data do not support. Stage 3 checks the verdict
for each questionnaire and which items group together. Neither reliability
figure is asked for: knowing which one answers the question is the task.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t08_score_justification.py [--verify|--naive]
```

`--verify` prints the numbers behind all of the above. `--naive` shows the
obvious analysis failing.
