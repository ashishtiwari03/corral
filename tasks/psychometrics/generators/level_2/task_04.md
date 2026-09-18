# Task 04 — which HSNS population fits each case sample?

## Task

Six anonymised reference populations answered the HSNS. Five case samples,
each containing 300 respondents, came from one of them. Determine which
population or populations are compatible with each case.

A population is compatible when its case log-likelihood is no more than `.04`
units per respondent below the best-fitting population. Some cases have one
compatible population; some have more than one.

## Files

- six labelled reference datasets;
- `cases.csv`: the five anonymised case samples;
- `codebook.md`: item and response definitions.

## Output

Return one JSON object containing every case exactly once. Order does not matter.

```json
{
  "classifications": {
    "case_1": ["population_a"],
    "case_2": ["population_b", "population_c"]
  }
}
```

## Why this requires investigation

The populations differ mainly in the relationships among items, not in their
marginal response distributions. A case should retain compatible alternatives
when its sample does not separate the populations.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy \
  python generators/level_2/gen_l2_t04_hsns_population_classification.py [--verify|--naive]
```
