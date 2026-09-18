# Task 07 — which questionnaire travels to other countries?

## The task

Build a model for each questionnaire on the US respondents, then test how far
each carries over to the six other countries in the file. Sort each
questionnaire in each country into one of four classes, from holding exactly
to not holding at all.

## The data

39,000 respondents in seven countries, from 510 to 24,000 each, rather than
the usual ten-country mix. Both questionnaires for everyone.
The US model is the HSNS as two traits correlated .35, and the Dirty Dozen as
one broad trait plus three narrow ones.

Put in deliberately, country by country:

- Great Britain identical to the US
- Canada and Australia with the broad Dirty Dozen trait weakened to half and
  a third of its strength
- India and Brazil with no broad trait at all; Brazil's two HSNS traits also
  merged (correlated .95)
- Germany with three Dirty Dozen items moved to the wrong subscale

## The answer

A twelve-cell grid. The HSNS mostly carries over; the Dirty Dozen mostly does
not, because the broad trait that holds it together in the US is absent
elsewhere.

## Traps

- **The questionnaire that fails is the one that measures better at home.** And
  it does not announce itself: the US model keeps fitting those countries
  perfectly well, it has simply stopped being the best model there. Judging by
  fit alone gets most of the grid wrong.
- **Two countries fail in different ways.** In one the answers are strong but
  its two traits have merged into one. In another the items have moved between
  subscales, so the questionnaire measures something, just not what it
  measures elsewhere.
- **Sample sizes differ tenfold.** From 510 to 6,200, so the same real
  difference is obvious in one country and invisible in another.

## Scoring

Stage 3 checks all twelve verdicts at once. Every cell has to be right, so
there is no credit for getting the easy countries.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t07_cross_country_replication.py [--verify|--naive]
```

`--verify` prints the numbers behind all of the above. `--naive` shows the
obvious analysis failing.
