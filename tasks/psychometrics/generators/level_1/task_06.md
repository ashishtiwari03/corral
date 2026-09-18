# Task 06 — how do the traits behind the two questionnaires relate?

## The idea

Five traits sit behind the two questionnaires, and the task is to report how
each one relates to the others once the noise in the measurements is accounted
for. One pair is so closely related, at .86, that the two are not worth treating
as separate traits.

## What goes wrong

- Adding up item scores and correlating the totals shrinks every relationship,
  and not evenly: the worst-measured traits shrink furthest. The result is a
  flat, mild picture with nothing standing out.
- The near-duplicate pair is exactly the one that shrinks most, because one of
  its traits is measured poorly. The finding that matters is the one the
  shortcut hides best.
- The threshold for calling two traits indistinguishable is given in the
  prompt, so the judgement has one right answer rather than being a matter of
  taste.

## Scoring

Stage 3 checks the reported correlations and which pair is called
indistinguishable. Factors are matched by the items they cover, so the agent can
name them anything.

The submission is the model, the correlations, and any pair that cannot be
told apart. Anything else the scorer needs it works out by re-fitting the
submitted model, so the agent is not asked for it.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t06_latent_relationships.py [--verify|--naive]
```

`--verify` prints the numbers behind everything above and checks the intended
answer wins. `--naive` checks that the obvious analysis fails.
