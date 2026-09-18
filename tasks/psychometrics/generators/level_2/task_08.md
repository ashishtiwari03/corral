# Task 08 — which model modifications survive replication?

## Task

You are reviewing a two-factor HSNS measurement model. A development sample
shows several item-level signs of misfit, and an independently collected
replication sample contains the same questionnaire and group variable.

Determine which apparent problems are reproducible and what mechanism each one
represents. Consider local dependence, cross-loadings, differential item
functioning, and poor item quality. A large development-sample modification
index is evidence to investigate, not evidence that a modification should be
retained.

## Files

- `data.csv`: development responses and group variable;
- `replication.csv`: independently collected replication responses;
- `codebook.md`: variable and item definitions;
- `preliminary_analysis.md`: the initial fit assessment.

## Required conclusions

Provide a final model and a compositional account of the findings. The finding
categories are not mutually exclusive: an item may participate in more than one
finding.

Use these categories:

- `local_dependence`: pairs of items requiring a residual covariance;
- `cross_loadings`: items requiring loading on a second factor;
- `dif_items`: items whose response differs by group after accounting for the
  factors;
- `poor_items`: items that should not be retained because their factor loading
  is too weak.

For each category, report whether the retained modification is supported in the
replication sample. Also report any development-supported modification that you
reject after examining the replication.

## Submission

Submit one JSON object:

```json
{
  "model_syntax": "F1 =~ HSNS1+...\nF2 =~ HSNS2+...\nF1 ~~ F2",
  "findings": {
    "local_dependence": [["item_a", "item_b"]],
    "cross_loadings": ["item_c"],
    "dif_items": ["item_d"],
    "poor_items": ["item_e"]
  },
  "replication": {
    "local_dependence": "replicates|does_not_replicate",
    "cross_loadings": "replicates|does_not_replicate",
    "dif_items": "replicates|does_not_replicate",
    "poor_items": "replicates|does_not_replicate",
    "rejected_development_modifications": [["item_f", "item_g"]]
  }
}
```

The final model may include residual covariances, cross-loadings, and paths from
`gender`. It must give `gender` a path to each factor. An item reported as poor
quality may be left out of the final measurement model; no other item may be
omitted.

The scorer refits the submitted syntax in both samples and derives the
replication verdicts by removing each retained modification from the submitted
model. It does not require a particular diagnostic workflow.

## Why this requires investigation

Several mechanisms can produce similar global misfit. Adding every path
suggested by one sample can improve fit while making the model less portable.
A modification is more defensible when its interpretation is clear and its
evidence is present again in the independent sample.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_2/gen_l2_t08_misfit_replication.py [--verify|--naive]
```
