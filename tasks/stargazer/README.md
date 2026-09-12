# Stargazer Task Environment

This task adapts the Stargazer radial-velocity exoplanet benchmark to the
standard Corral lifecycle. An agent receives public observations and tools,
may request a bounded number of diagnostic candidate evaluations, and is
scored only on its final JSON answer.

## Task splits and candidate budgets

The two official levels contain 10 fixed, reference-valid synthetic tasks
each. Their explicit IDs are committed in `environments/level_*/tasks_json`,
so membership never changes at runtime.

| Split | Upstream difficulty | Synthetic tasks | Diagnostic evaluations | Final answer | Total candidates |
| --- | --- | ---: | ---: | ---: | ---: |
| Level 1 | 5–7 | 10 | 4 | 1 | 5 |
| Level 2 | 8–10 | 10 | 9 | 1 | 10 |

Selection prioritizes `corral_exoplanet_rv.zip`, deduplicating by the original
Stargazer task ID. The ZIP contains 15 synthetic tasks, all with IDs already in
the local bank: three fall in difficulty 5–7 and five in difficulty 8–10. All
eight are selected using their ZIP datasets, including three repaired systems.
The remaining seven Level 1 tasks and five Level 2 tasks come from the previous
official environment banks. Each level is balanced across its difficulties
(4/3/3), with additional tasks chosen by task ID from passing references.
`data/selection_manifest.json` records every selected ID, its source, ZIP
aliases, and the archive checksum. Level 3 is no longer available.

The 20 archival tasks are available separately as the `real` challenge split
and are not included in official Levels 1–2. Their published systems do not
currently pass the unchanged observation model and thresholds, so real-split
results should be reported separately until that split is calibrated.

## Setup and execution

Create and synchronize the task-local Python 3.12 environment:

```bash
cd tasks/stargazer
uv venv --python 3.12
uv sync --locked
```

Inspect an official level or the separate real-data split:

```bash
uv run python -m stargazer.env --level 1
uv run python -m stargazer.env --level real
```

Level `2` is also available. These commands build and list environment
definitions; Corral no longer uses a separate task HTTP server.

Run one task through the current local runtime:

```bash
uv run corral run --agent tool-calling --environment stargazer \
  --task seed15_diff5 --model openai/gpt-4o
```

For scored trials, use `corral bench` with Docker as shown below. Select a split
with `--env-kwargs '{"level": 2}'` (or `{"level": "real"}`), and use
`task_config` or `selector_path` in the same object for a custom selector.
`CORRAL_WORK_DIR` controls the workspace root.
Use `--sandbox local` only for local debugging.

For Docker execution, build the task image from the repository root (build the
base image first if `corral-benchmark:latest` is unavailable):

```bash
docker build -f docker/benchmark.Dockerfile -t corral-benchmark:latest .
docker build -f docker/stargazer.Dockerfile -t corral-stargazer:latest .
tasks/stargazer/.venv/bin/python -m corral.cli bench \
  --agent tool-calling --environment stargazer --task seed15_diff5 \
  --model openai/gpt-5.6-terra --sandbox docker \
  --sandbox-image corral-stargazer:latest --trials 1 \
  --agent-kwargs '{"reasoning_effort": "medium", "additional_drop_params": ["temperature"]}'
```

Docker runs model-written analysis in Corral's unprivileged worker filesystem.
Only public observations and an opaque analysis checkpoint cross that boundary.
The checkpoint is decoded after privilege dropping; the controller never
unpickles it. Source under `/opt/corral`, task truth, private checkpoints,
other workspaces, and the host filesystem are not mounted into the worker.
The writable worker filesystem is limited to its trial workspace; installed
OS and scientific dependencies are available read-only. The REPL worker also
clears its environment before decoding state or running analysis code.

All task images built from the repository root use the shared `.dockerignore`
to exclude credentials, virtual environments, caches, and local workspaces.
The REBOUND compatibility fix lives in
[`scripts/install_rebound.py`](scripts/install_rebound.py), which installs the
locked source after verifying its checksum and corrects REBOUND 5.1.1's x86
architecture detection on ARM. The Dockerfile calls this task setup script;
it can also be run with a local environment's Python interpreter when a C
compiler is available.

## Execution state

Like Wetlab, `StargazerEnvironment` restores disposable sessions from
`ExecutionState.environment` and returns their changes for Corral to commit.
Diagnostic history, remaining budget, and the success lock persist across
resumption and branch forks. The Python namespace is checkpointed inside its
public-data-only worker, preserving arrays, functions, and numerical random
state without replaying earlier code. Checkpoints are decoded only inside the
worker and should be resumed with the same Python and scientific environment.
Task truth stays in the task definition, outside the analysis checkpoint.
Values that cannot be checkpointed, such as live generators, cause the call to
fail without committing its namespace changes.

Run the task's regression suite with:

```bash
uv run pytest
```

## Interaction and submission contract

Every trial exposes:

- `python_repl`, a persistent per-trial numerical namespace containing NumPy,
  SciPy, the observation arrays, instrument labels, stellar mass, and the
  reference epoch;
- `planet_from_fit`, which converts fitted semi-amplitude and phase values to
  Stargazer's native planet fields;
- `evaluate_candidate`, which evaluates canonical candidate arguments and
  returns redacted four-gate diagnostics without ending or scoring the trial.

Valid diagnostic evaluations consume the split allowance. Invalid JSON or
schema-invalid candidates do not. Once a diagnostic candidate passes all four
gates, further diagnostic calls are locked and the agent is instructed to
return that candidate as its final answer. Diagnostic history is never used as
a scored fallback.

The same canonical JSON works unchanged as `evaluate_candidate` arguments and
as the final answer:

```json
{
  "planets": [
    {
      "P_days": 23.5,
      "m_sin_i_mjup": 0.12,
      "e": 0.1,
      "omega_rad": 1.2,
      "l_rad": 3.4
    }
  ],
  "noise_jitter_ms": 0.2
}
```

`P_days` is in days, `m_sin_i_mjup` is in Jupiter masses, and angles are in
radians. `l_rad` is mean longitude at the first observation time. The optional
compatibility fields `inc_rad` and `Omega_rad` are accepted but are unnecessary
for the radial-velocity model. The evaluator fits one constant velocity offset
per instrument.

For migration only, the final scorer continues to accept the previously
supported field aliases and nested `noise.sigma_jitter_ms`. New candidates
should use the flat canonical schema above; aliases are not exposed by the
diagnostic tool or task prompt.

## Evaluation

Both `evaluate_candidate` and the final scorer call the same
`evaluate_submission()` implementation. A final answer scores `1.0` only when
all four gates pass:

1. BIC improvement over a per-instrument constant model is greater than zero
   per observation;
2. residual RMS is at most 1.5 times the median measurement uncertainty;
3. aggregate Hungarian-assigned physical match score is at least 0.8;
4. recovered planet count equals the hidden reference count.

Diagnostic feedback reports BIC and BIC per observation, residual RMS and MAE
with the RMS threshold, aggregate match score with its threshold, count
pass/fail, and remaining evaluations. It does not reveal hidden parameters,
truth indices, assignments, signed errors, or the true planet count.

## Task-bank audit

`python -m stargazer.audit` deterministically evaluates all 120 published
reference systems through the final scorer, validates official membership, and
regenerates `data/reference_audit.json`. CI-style verification uses:

```bash
uv run python -m stargazer.audit --check
```

The committed audit records invalid reference parameters and failures of the
BIC, RMS, physical-match, and count gates. At the current pinned revision,
79/100 synthetic references and 0/20 real references pass after importing the
eight selected ZIP records. The official levels select 20 passing synthetic
references in the requested difficulty ranges; thresholds are not weakened
per task. Validation checks membership counts, uniqueness, source, difficulty,
and reference scores.

## Provenance and deliberate interface differences

The original task bank comes from Stargazer revision
`3f617667472061e253288c7b26f0e70f186f2dff`. Eight selected synthetic records are
replaced by the preferred exports in `corral_exoplanet_rv.zip`, whose provenance
identifies `Stargazer_synthetic_task_repaired_v1`. Their observations, planetary
truth, and stellar masses are preserved exactly in Stargazer's native record
schema. They are marked as RV-only to avoid a second compatibility conversion.
Other synthetic records are converted in memory from their original REBOUND
signal to the current RV-only Keplerian semantics while retaining their noise
realization.

Compared with the upstream interaction loop, Corral owns the final submission:
the iterative submission action is named `evaluate_candidate`, the final
answer is the last candidate opportunity, and only that final answer affects
the benchmark score. See `THIRD_PARTY_NOTICES.md` for attribution and licenses.
