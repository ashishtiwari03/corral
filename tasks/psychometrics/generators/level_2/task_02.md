# Task 02 — does the HSNS model generalize out of sample?

Build a measurement model in a labelled training sample, then refit it —
unchanged — in three anonymised holdouts and classify what happens to it in each.
Use these decision rules:

| decision | when |
|---|---|
| `generalizes` | every loading and the factor correlation stay within .10 |
| `measurement_structure_holds_relations_differ` | loadings stay within .10, the factor correlation moves more |
| `measurement_structure_fails` | some loading moves more than .10 |

The training file is labelled and the holdout files are anonymised. Use the
codebook to identify the questionnaire variables; do not infer population
identities from file metadata.

## What to report

Submit the frozen model and one decision for each holdout. The decision must be
based on the model as refitted in that holdout, not on a modified holdout-specific
model.

## What makes this non-trivial

Global fit in the training data is not evidence that a model transports unchanged.
A holdout can preserve the item measurement structure while changing the
relationship between factors, or it can make the measurement structure itself
inadequate. The decisions must agree with what the frozen model produces in each
holdout.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_2/gen_l2_t02_out_of_sample_generalization.py [--verify|--naive]
```
