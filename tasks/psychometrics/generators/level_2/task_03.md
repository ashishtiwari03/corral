# Task 03 — which population did each case sample come from?

## Task

Six anonymised reference populations answered the Dirty Dozen. Five case samples
of 200 respondents each came from one of those populations. Determine which
populations are compatible with each case.

A population is compatible when its case log-likelihood is no more than `.06`
units per respondent below the best-fitting population. Some cases have one
compatible population; some have more than one.

## Files

- six labelled reference datasets;
- `cases.csv`: the five anonymised case samples;
- `codebook.md`: item and response definitions.

## Output

Return one JSON object containing every case exactly once. Give each case the
complete set of compatible population identifiers. Do not force a unique answer
when the likelihood rule leaves more than one population compatible.

```json
{
  "classifications": {
    "case_1": ["population_a"],
    "case_2": ["population_b", "population_c"]
  }
}
```

## Why this requires investigation

The populations differ mainly in how items relate to one another, not in their
marginal response distributions. Those differences can be studied from a case
sample, but not reliably from one respondent or from item means alone.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy \
  python generators/level_2/gen_l2_t03_ddm_population_classification.py [--verify|--naive]
```
