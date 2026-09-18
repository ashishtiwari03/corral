# Task 06 — is the group difference real or an export artifact?

## Task

Investigate a reported difference between two groups on the Hypersensitive
Narcissism Scale (HSNS). Decide which items are usable for comparison, repair
any problems that can be repaired, fit a model that allows genuine group
differences in item behaviour, and decide whether the group comparison should
be reported.

The task is deliberately not resolved by the export log or by one descriptive
statistic. Use the response patterns, the preliminary memo, and the audit as
competing pieces of evidence. In particular, distinguish an item whose recorded
values carry no information about respondents from an item that genuinely has a
different response function in the two groups.

## Files

- `data.csv`: item responses, group indicator, and country field.
- `codebook.md`: response coding and item descriptions.
- `preliminary_analysis.md`: an initial, unresolved analysis.
- `export_audit.csv`: transmission records for each item.
- `export_audit_codebook.md`: definitions of the audit fields.

## Required conclusions

Give one diagnosis for every HSNS item. Use exactly one of these labels for
each item: `sound`, `mis_keyed`, `inserted_neutral`, `gender_dif`, or
`weak_item`. A label may apply to no items, one item, or several items.

Then provide a model syntax and a comparison decision. Your model should make
the group comparison explicit and should represent item-level group dependence
where the evidence supports it. If an item is diagnosed as `mis_keyed`, repair
its direction before interpreting the latent group difference.

## Why this requires investigation

Several explanations can produce similar group-level symptoms. The audit is
about transmission rather than measurement, and a high frequency of one
response category is not by itself proof of an export error. A useful check is
to compare respondents who gave a suspect response with their responses on the
remaining items, while accounting for missing data and item quality.

## Submission

Submit one JSON object:

```json
{
  "model_syntax": "F =~ HSNS1+...+HSNS10\nF ~ gender\nHSNS2 ~ gender",
  "item_diagnoses": {"HSNS1": "sound", "...": "..."},
  "comparison": "reportable_after_repair|not_reportable"
}
```

The scorer checks the complete item diagnosis, model coverage, the explicit
group path, the items allowed to differ between groups, and the repaired latent
group effect. It does not require a particular workflow.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_2/gen_l2_t06_gender_item_integrity.py [--verify|--naive]
```
