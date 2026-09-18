# Task 06 — how do the traits behind the two questionnaires relate?

## The task

Select the US respondents and report how each trait behind one questionnaire
relates to each trait behind the other, correcting for the fact that no trait
is measured perfectly. Also name any pair too alike to tell apart.

## The data

Five traits behind the two questionnaires: two from the HSNS, three from the
Dirty Dozen. Loadings .40–.81.

Put in deliberately:

- one HSNS trait measured poorly throughout (loadings .40–.48), so
  relationships involving it shrink furthest when the shortcut is used
- that same trait correlated **.86** with Dirty Dozen narcissism, close enough
  that the two are not separate things
- everything else moderate, so the near-duplicate pair is the only thing that
  stands out

## The answer

Five traits sit behind the two questionnaires. One cross-questionnaire pair
correlates .86, close enough that the two are not worth treating as separate
things. The rest are moderate.

## Traps

- **Adding up item scores distorts everything.** Correlating the totals shrinks
  every relationship, and not evenly: the worst-measured traits shrink
  furthest. The result is a flat, mild picture with nothing standing out.
- **The shortcut hides the one finding that matters.** The near-duplicate pair
  shrinks most of all, because one of its traits is measured poorly.
- **The threshold is given, not guessed.** The prompt states the correlation
  above which two traits count as indistinguishable, so the judgement has one
  right answer rather than being a matter of taste.

## Scoring

Stage 3 checks the reported correlations and which pair is called
indistinguishable. Traits are matched by the items they cover, so an agent can
name them anything it likes.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t06_latent_relationships.py [--verify|--naive]
```

`--verify` prints the numbers behind all of the above. `--naive` shows the
obvious analysis failing.
