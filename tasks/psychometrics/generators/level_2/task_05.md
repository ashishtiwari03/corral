# Task 05 — what explains the apparent relationship?

## Task

Investigate an unexpectedly strong relationship between two HSNS dimensions in
a panel-delivered survey. The response file, preliminary memo, and delivery
audit contain partially overlapping evidence.

Decide whether the relationship reflects independent respondents, repeated
deliveries, chance response collisions, or more than one mechanism. Identify a
record key, collapse repeated records using that key, and refit the measurement
model.

## Files

- `data.csv` and `codebook.md`: delivered responses and variables;
- `preliminary_analysis.md`: the initial analysis and panel notes;
- `delivery_audit.csv` and `delivery_audit_codebook.md`: delivery evidence joined
  by `session_id`.

The audit describes deliveries, not respondents. Different people may share its
values, and a repeated delivery may receive a new participant identifier.

## Output

Return one JSON object:

```json
{
  "model_syntax": "F1 =~ ...\nF2 =~ ...\nF1 ~~ F2",
  "diagnosis": "repeated_delivery_only|chance_collisions_only|both_mechanisms",
  "duplicate_key": ["<column>", "..."]
}
```

The key must use columns from `data.csv`. The scorer applies it to collapse
records, refits the submitted model, and reads the resulting factor
correlation.

## Why this requires investigation

Exact response matches are expected in a long survey. Removing every match can
delete genuine respondents; retaining every row can count one respondent more
than once. The audit helps test the explanations but is not a respondent ID.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy \
  python generators/level_2/gen_l2_t05_duplicate_records.py [--verify|--naive]
```
