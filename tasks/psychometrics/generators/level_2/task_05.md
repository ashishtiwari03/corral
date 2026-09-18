# Task 05 — what explains the apparent relationship?

You are investigating an unexpectedly strong relationship between two dimensions of the Hypersensitive Narcissism Scale in a panel-delivered survey. The response file, a preliminary memo, and a delivery audit provide partially overlapping evidence.

Decide whether the relationship is supported by independent respondents, distorted by repeated deliveries, affected by chance response collisions, or shaped by more than one mechanism. Establish an operational record key, collapse records using that key, and fit the measurement model again.

## Available files

- `data.csv` and `codebook.md`: delivered HSNS responses and variables;
- `preliminary_analysis.md`: the initial analysis and panel notes;
- `delivery_audit.csv` and `delivery_audit_codebook.md`: delivery-level evidence joined by `session_id`.

The audit contains delivery characteristics rather than a respondent identifier. Different people may share those characteristics, and the panel can issue fresh identifiers when a record is transmitted again. Treat the memo as a hypothesis to test, not as an answer.

## What to return

Return one JSON object:

```json
{
  "model_syntax": "F1 =~ ...\nF2 =~ ...\nF1 ~~ F2",
  "diagnosis": "repeated_delivery_only|chance_collisions_only|both_mechanisms",
  "duplicate_key": ["<column>", "..."]
}
```

The key must contain columns from `data.csv`. The scorer applies it to collapse records, refits the submitted model, and reads the factor correlation from that refit.

## Why this is not a lookup

Exact response matches are expected in a long survey with five-point items, but repeated deliveries can create the same pattern for a different reason. Removing every match can delete genuine respondents, while retaining every row can count one respondent more than once. The audit is supporting evidence, not a direct identity field.

## Rebuild

```bash
uv run python generators/level_2/gen_l2_t05_duplicate_records.py [--verify|--naive]
```
