# Task 01 — can this questionnaire compare the two groups?

## Task

Assess a reported HSNS difference between two groups. The preliminary memo bases
its recommendation on observed scores, but the response file and a separate
file of behavioural indicators provide evidence about the measurement model and
the underlying traits.

Decide whether the group comparison is defensible. Return a reproducible model,
one recommendation, and the items that materially affect that recommendation.

## Files

- `data.csv`: HSNS responses and group information;
- `behavior.csv`: behavioural indicators for the same participants;
- `codebook.md` and `behavior_codebook.md`: variable definitions;
- `preliminary_analysis.md`: the initial analysis.

## Output

Submit the model, a recommendation, and the affected-item set in the JSON
format given in the task definition. The model must support the comparison that
the recommendation rests on.

## Why this requires investigation

A difference in observed responses does not by itself establish a difference in
the underlying traits. Item response behaviour, alternative measurement models,
and the external indicators can support or weaken the memo's conclusion.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_2/gen_l2_t01_group_comparability.py [--verify|--naive]
```
