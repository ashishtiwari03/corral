# Task 09 — how strongly are the two questionnaires related?

## The idea

Five traits sit behind the two questionnaires. Within a questionnaire they
are moderately related; across the two they are barely related at all, and that
is the finding. But a twelfth of the respondents were not reading the questions:
8% gave the same answer to all 22 items and 4% answered at random.

## What goes wrong

- A respondent who answers 3 to everything makes every pair of items move
  together perfectly. Mixing those people in pulls **every** correlation
  upward, and since the rise is roughly uniform it barely dents a strong
  relationship while more than doubling a weak one. The damage lands exactly
  where the answer is.
- Nothing in the model output says so. The correct structure still fits the
  contaminated data well, so there is no misfit to notice and no residual to
  chase.
- The obvious screen is the decoy. The file records how accurately each person
  says they answered, and screening on it removes 39 of 1,655 non-readers while
  discarding hundreds of good respondents. People who do not read the questions
  do not report that they did not read them.
- The fix is to look at the raw answers: a person whose 22 answers never vary
  is not answering the questions. That catches every one of them and no one
  else.
- The random responders survive any such screen and turn out not to matter.

## Scoring

Stages 1 and 2 do no work here, deliberately: the contamination leaves the
structure intact, so an agent that never cleaned the data submits the same model
and clears both. The task rests entirely on stage 3, the reported correlations.

The submission is the model and the correlations across the two
questionnaires. Anything else the scorer needs it works out by re-fitting the
submitted model, so the agent is not asked for it.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t09_careless_responding.py [--verify|--naive]
```

`--verify` prints the numbers behind everything above and checks the intended
answer wins. `--naive` checks that the obvious analysis fails.
