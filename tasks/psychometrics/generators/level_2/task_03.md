# Task 03 — which population did each case sample come from?

## The task

Six anonymised reference populations answered the Dirty Dozen. Five case
samples of 200 respondents each came from one of them. Say which populations
each case could have come from.

The compatibility rule is given to the agent: a population is compatible when
the case's responses are no more than **0.06 units of log-likelihood per
respondent** less likely under it than under the best-fitting population. Some
cases have one compatible population, some have two.

## The data

| population | structure |
|---|---|
| a | three correlated factors, loadings .68, r .25 |
| b | one factor, loadings .62 |
| c | bifactor, general .62, specific .38 |
| d | two factors (P+N against M), loadings .70, r .30 |
| e | three correlated factors, loadings .66, r .29 |
| f | population a plus a residual covariance of .40 on DDP1–DDP2 |

3,000 respondents per reference file, all complete. Every population uses the
same item thresholds, so they differ only in how the items relate to one
another — not in what people answer on average.

**a and e are deliberately close.** 200 respondents cannot separate loadings of
.68/r .25 from .66/r .29, so any case from either is compatible with both.
Population f is the true residual dependence: two items share variance the
factors do not explain.

## The answer

Computed by applying the rule, not assigned:

| case | drawn from | compatible |
|---|---|---|
| case_1 | a | `a`, `e` |
| case_2 | c | `c` |
| case_3 | d | `d` |
| case_4 | e | `a`, `e` |
| case_5 | f | `f` |

Cases 1 and 4 come from different populations and correctly receive the same
answer. That is the point of the task, not a defect.

## Traps

**Forcing a unique answer.** Cases 1 and 4 have no unique answer. Picking the
single best-fitting population is wrong for both.

**Hedging everywhere.** Naming several populations for the identifiable cases
fails too. The margin between a's set and the nearest non-member is 9×, so the
singletons are not close calls.

**Comparing item means.** The populations share their thresholds, so their
marginal distributions are nearly identical. Matching on item means identifies
the source 20% of the time against a 17% chance baseline, and misclassifies all
five delivered cases.

**Classifying one respondent at a time.** The populations differ in covariance,
which is a property of a sample. A case has to be judged as a sample.

## Scoring

Exact set match on all five cases, all of which must pass. No model or working
is submitted — only the candidate populations per case.

`--verify` draws 25 fresh case samples from each of the six populations and
checks the rule gives the same set every time, and that each case's own
population is always in it. Both are 100%.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy \
  python generators/level_2/gen_l2_t03_ddm_population_classification.py [--verify|--naive]
```
