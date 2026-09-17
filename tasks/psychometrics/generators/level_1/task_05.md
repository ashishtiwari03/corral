# Level 1 / Task 05 — Which gender comparisons are defensible?

**Ask.** Work out how far the HSNS measures the same thing in men and women, then
decide which of six between-group comparisons that licenses — and carry out the
ones that are licensed.

**Generator.** `gen_l1_t05_defensible_comparisons.py`

## Generating model

Two correlated traits: the HSNS (10 items) and Dark Triad narcissism (4 items).

- **Loadings are identical in both groups.** The items relate to the trait the
  same way, so the construct means the same thing in each — comparisons of
  structure, spread and association are all sound.
- **Six of the ten HSNS items are answered differently at the same trait level**,
  three shifted one way and three the other. Four items are clean, but nothing in
  the data says which four.
- Women's trait level really is +0.30 higher. That is **not recoverable**.
- The two traits correlate **.50 in men** and **.30 in women** — a real
  difference, and a comparison that *is* licensed.

## The answer

| comparison | defensible | because |
|---|---|---|
| factor structure | yes | same items, same factor in both groups |
| loadings | yes | identical across groups |
| factor variances | yes | licensed once loadings are invariant |
| association with the other trait | yes | same |
| **latent means** | **no** | intercepts differ, with no trustworthy reference items |
| **observed score means** | **no** | strictly weaker than the latent comparison |

## Why "no" is the hard part

The mean difference is right there and significant. Refusing to report it is the
whole task — and it is not a rule of thumb here, it is a fact about the data:

```
latent mean difference, by which items are assumed unbiased:
  the truly clean four         +0.145
  four shifted one way         -0.054
  four shifted the other       +0.355
  all ten (assume no bias)     +0.169
  spread across anchor choices  0.409
```

The answer swings from negative to substantially positive depending on an
assumption the data cannot test. Meanwhile the association comparison is stable
and recovers cleanly — men +0.492 against a target of +0.494, women +0.304
against +0.295.

Both failure directions are scored: an agent that compares the means anyway
fails, and so does one that refuses every comparison.

## Scored

The reported correlations are what stop the judgement panel being answered from
textbook knowledge without touching the data — a submission with the right panel
and pooled correlations fails.

Submission: `model_syntax`, `comparisons` (six judgements),
`trait_correlation_men`, `trait_correlation_women`.
