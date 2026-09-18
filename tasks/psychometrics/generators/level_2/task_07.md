# Task 07 — does the HSNS predict behaviour beyond group membership?

## Task

Evaluate whether the Hypersensitive Narcissism Scale (HSNS) predicts an
independently measured behavioural outcome. A preliminary memo reports an
association in a labelled training sample. A separate holdout contains the same
questionnaire, group variable, and outcome.

The pooled association is not enough to answer the question. Determine whether
the latent association remains after adjusting for group membership, whether it
appears within both groups, whether it generalizes to the holdout, and whether
any item contributes an outcome association that fails to replicate.

## Files

- `data.csv`: training responses, group, and behavioural outcome.
- `holdout.csv`: separately collected responses and outcome.
- `codebook.md`: variable and item definitions.
- `preliminary_analysis.md`: the initial analysis and its limitations.

The two files are already joined within themselves by `participant_id`; no
record-linkage step is required.

## Decision rules

Use the following rules for the reported claims:

- `supported`: the group-adjusted latent coefficient has absolute value at least
  `.10` in both samples.
- `replicates`: the latent coefficient has absolute value at least `.10` in
  both groups, with the two within-group estimates no more than `.06` apart.
- `generalizes`: the group-adjusted latent coefficient changes by no more than
  `.06` between training and holdout.

For item stability, identify items with an absolute direct outcome coefficient
of at least `.10` in training and no more than `.05` in the holdout, after
accounting for the latent trait and group.

## Submission

Submit one JSON object:

```json
{
  "model_syntax": "F =~ HSNS1+...+HSNS10\nbehavior ~ F + gender",
  "association": {
    "group_adjusted": "supported|not_supported",
    "within_group": "replicates|does_not_replicate",
    "holdout": "generalizes|does_not_generalize"
  },
  "unstable_items": ["item_name"]
}
```

The model must cover every HSNS item and include a latent factor, `behavior`,
and `gender` in the behavioural analysis. Include item-level outcome paths when
they are needed to account for an item-specific contribution. The scorer
re-estimates the claims from the submitted measurement model; it does not
require a particular sequence of analyses.

## Why this requires investigation

A pooled questionnaire–behaviour association can combine a genuine within-group
relationship with a difference between groups. A training-only item effect can
also make a latent coefficient look stronger than it is. Neither issue is
resolved by a good global measurement fit or by statistical significance in the
training sample.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_2/gen_l2_t07_behavioral_validity.py [--verify|--naive]
```
