# Task 08 — what may each questionnaire's scores be used for?

## The idea

For each questionnaire, decide what may legitimately be added up: one total,
separate subscale scores, both, or neither. The HSNS supports a total. The Dirty
Dozen does not, and supports subscale scores instead.

## What goes wrong

- The reliability figure almost every paper reports ranks the two backwards.
  The Dirty Dozen scores higher on it, and its total is the one that means
  nothing: only about a third of that total reflects anything its twelve items
  share, and the rest blends three distinguishable traits.
- Reaching for the usual correction is not enough either. By that measure the
  Dirty Dozen total looks excellent. Only the figure that asks specifically how
  much of the total comes from a *single common* trait separates them.
- Keeping both the total and the subscales is the natural compromise, since
  the subscales are genuinely good and the total looks fine. It is wrong.
- Analysing all countries together breaks both verdicts at once.

## Scoring

Stage 2 rejects a model of three merely related traits, which would reach the
right verdict by a route the data do not support. Stage 3 checks the verdict for
each questionnaire and which items group together. Neither reliability figure is
asked for: knowing which one answers the question is the task.

The submission is the model and a verdict per questionnaire. Anything else the
scorer needs it works out by re-fitting the submitted model, so the agent is
not asked for it.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t08_score_justification.py [--verify|--naive]
```

`--verify` prints the numbers behind everything above and checks the intended
answer wins. `--naive` checks that the obvious analysis fails.
