# Psychometrics Task Environment

Every dataset here is simulated from a known model, so the ground truth is known. 
Each task is built so that the obvious analysis produces a specific wrong answer.

## Layout

```
build.py                          rebuild and check the whole environment
generators/common.py              simulation, fitting and artifact code shared by all tasks
generators/level_*/gen_*.py       one script per task: the model its data come from
generators/level_*/task_*.md      what that task is about and how it is scored
artifacts/level_*/task_*/         data.csv and codebook.md, which the agent sees;
                                  truth.json, which it does not
environments/level_*/tasks_json/  the task definitions
psychometrics/score.py            the scorer
tests/test_scoring.py             checks every task scores as intended
```

## Building

Generators build everything under `artifacts/` and `environments/` , we can rebuild it with the following comments

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  --with factor_analyzer python build.py --clean   # rebuild from nothing
uv run ... python build.py --check                 # build, then check everything
uv run ... python build.py --tasks 3 7             # only these
```

`--check` runs three things for every task: that the intended answer wins (`--verify`), that the obvious analysis fails (`--naive`), and that the scorer gives the intended verdicts. 

## Scoring

A submission gives a model plus the numbers that model produced.
Anything else the scorer needs, it works out by re-fitting that model.

Scoring runs in three stages, all of which must pass. 

1. **Constraints.** Checks the usability of the model. A negative variance, or two factors too alike to tell apart, make a model invalid.
2. **Comparison.** The model the data came from sets a floor. A submission must be at least as good on fit, on how many parameters it spends, and on how close the correlations it implies come to the truth. 
3. **Claims.** The numbers reported, checked against the values the data were built from.

Reported: `score_binary`, `score_partial` (for diagnosis only) and `checks_vector`.

## Level 1

Links to tasks in level 1

| task | question |
|---|---|
| [1](generators/level_1/task_01.md) | How many traits does the HSNS measure? |
| [2](generators/level_1/task_02.md) | How many traits does the Dirty Dozen measure? |
| [3](generators/level_1/task_03.md) | Does the HSNS measure one trait or two? |
| [4](generators/level_1/task_04.md) | Which questionnaire supports a comparison between men and women? |
| [5](generators/level_1/task_05.md) | Which comparisons between men and women can be defended? |
| [6](generators/level_1/task_06.md) | How do the traits behind the two questionnaires relate? |
| [7](generators/level_1/task_07.md) | Which questionnaire is portable to other countries? |
| [8](generators/level_1/task_08.md) | What may each questionnaire's scores be used for? |
| [9](generators/level_1/task_09.md) | How strongly are the two questionnaires related? |
| [10](generators/level_1/task_10.md) | Which items are bad, and which is the data? |

## Level 2

:TODO

Basic idea - Two or more analyses, both defensible, give opposite answers, and doing either one better does not help. The agent has to notice the conflict, work out what would settle it, and run a test.
