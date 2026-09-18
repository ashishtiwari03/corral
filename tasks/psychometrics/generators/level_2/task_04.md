# Task 04 — which HSNS population fits each case sample?

Six anonymised reference populations answered the Hypersensitive Narcissism Scale. Five case samples, each containing 300 respondents, came from one of them. Determine which population or populations are compatible with each case.

The population identifiers do not describe the populations. Infer their measurement structures from the reference samples. A population is compatible when the case's responses are no more than 0.04 units of log-likelihood per respondent less likely under it than under the best-fitting population. Some cases have one compatible population and some have more than one.

## Available files

- six labelled reference datasets;
- `cases.csv`, containing the five anonymised case samples;
- `codebook.md`, describing the HSNS items.

## What to return

Return one JSON object containing every case exactly once. Each value is a list of compatible population identifiers. Order does not matter.

```json
{
  "classifications": {
    "case_1": ["population_a"],
    "case_2": ["population_b", "population_c"]
  }
}
```

## Why this is not a lookup

The populations differ mainly in how the items relate to one another, not in their marginal response distributions. Those differences can be studied from a sample but not reliably from one respondent. A unique answer should be reported only when the case sample separates the populations; otherwise retain the compatible alternatives.

## Rebuild

```bash
uv run python generators/level_2/gen_l2_t04_hsns_population_classification.py [--verify|--naive]
```
