# Task 03 — classify respondents against DDM populations

You are given large response samples from six anonymised reference populations and five anonymised respondent profiles. The questionnaire is the Dirty Dozen. Determine which reference population or populations are compatible with each person.

The population identifiers do not tell you how the populations differ. Use the reference samples to investigate plausible measurement structures and parameter patterns before comparing the five profiles. A respondent can have more than one compatible population when the available responses do not distinguish them. Do not force a unique assignment merely because one answer is required.

## What to return

Return one JSON object containing every person exactly once. Each value is a list of compatible population identifiers, such as `['population_b', 'population_c']`. The list may contain one or several populations.

## Why this is not a lookup

The reference populations include several competing structures and some deliberately similar cases. Overall fit alone is not enough to decide whether a short response vector is compatible with a population, and a close match to one model does not prove that the alternatives are impossible. The task is about calibrated compatibility, not about guessing a real-world country from a label.

## Rebuild

```bash
uv run python generators/level_2/gen_l2_t03_ddm_population_classification.py
```
