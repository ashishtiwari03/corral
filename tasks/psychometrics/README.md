# Psychometrics Task Environment

Agents recover the latent measurement model behind questionnaire data and decide
what it supports. Every dataset is simulated from a known model, so the true
structure is exact; each task is built so the default analysis produces a
specific wrong answer.

```
generators/common.py                 shared simulation, fitting and artifact code
generators/level_*/gen_*.py          one script per task: its generating model
generators/level_*/task_*.md         what that task generates and tests
artifacts/level_*/task_*/            data.csv, codebook.md (agent-visible)
                                     truth.json (hidden)
environments/level_*/tasks_json/     Corral task definitions
psychometrics/score.py               scorer
tests/test_scoring.py                every task scores as intended
```

Generators are seed-deterministic and read nothing external. Run one with
`--verify` to check the intended answer wins and each rival fails for its own
reason, or `--naive` to check the default analysis fails; both run in CI.

Submissions carry a lavaan model specification plus the estimates it produced;
anything else the scorer needs it derives by re-fitting. Scoring runs in three
stages — constraints, then Pareto dominance over the generating model as a
floor, then the reported estimates against the generative parameters — and
criteria are never combined into a weighted sum. Reported: `score_binary`,
`score_partial` (diagnostic), `checks_vector`.

The task description never says which columns belong to which instrument, or
how they group into subscales. Working that out from the item text is part of
the task.

| task | question | generating model |
|---|---|---|
| [L1-T1](generators/level_1/task_01.md) | HSNS factor structure | two correlated factors |
| [L1-T2](generators/level_1/task_02.md) | Dirty Dozen factor structure | general factor + specifics |
| [L1-T3](generators/level_1/task_03.md) | HSNS: one trait or two? | one factor + two duplicate item pairs |
| [L1-T4](generators/level_1/task_04.md) | Which instrument supports a gender comparison? | HSNS biased on four items; Dirty Dozen invariant, but only under its real structure |
| [L1-T5](generators/level_1/task_05.md) | Which gender comparisons are defensible? | loadings invariant, six items shifted, means not identified |
| [L1-T6](generators/level_1/task_06.md) | How do the two instruments' dimensions relate? | five correlated dimensions, one pair near-redundant |
| [L1-T7](generators/level_1/task_07.md) | Which instrument travels across countries? | HSNS holds up; the Dirty Dozen's general factor is US-specific |
| [L1-T8](generators/level_1/task_08.md) | What may each instrument's scores be used for? | HSNS one factor; Dirty Dozen's total is reliable but not interpretable |
| [L1-T9](generators/level_1/task_09.md) | How strongly are the two instruments related? | five dimensions, barely related across instruments; 8% of respondents straight-lined |
| [L1-T10](generators/level_1/task_10.md) | Which items are bad, and which is the data? | one trait, three recording faults and one genuinely poor item |

Level 2 is not yet defined; the boundary between the levels is being reworked.
