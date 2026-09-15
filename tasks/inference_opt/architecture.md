# Inference-opt architecture

This document describes the current first-iteration design of the inference-time optimization environment.

The environment asks a teacher agent to write a Python policy that improves a frozen student model. The teacher edits `policy/policy.py`, runs experiments on the training split, and submits the policy for evaluation on the held-out test split.

The implementation currently follows the stateful pattern used by the wetlab environment.
Inference tools are trusted and return updated session state through Corral's normal environment-state transition mechanism.

## System overview

```text
                         Corral host

  teacher agent
       │
       │ actions: edit files, call tools, submit answer
       ▼
  AgentSession
       │
       ▼
  InferenceOptEnvironment
       │
       ├── workspace tools ───────────────► task workspace
       │                                    policy/policy.py
       │                                    guide/, notes.md, TODO.md
       │
       ├── trusted inference tools
       │       ├── get_baseline
       │       ├── reveal_train_questions
       │       ├── query_student
       │       ├── dry_run_policy
       │       ├── evaluate_candidate
       │       ├── inspect_failures
       │       ├── compare_runs
       │       ├── get_budget
       │       └── submit_policy
       │
       └── committed Corral environment state
               └── inference_state
                       budget counters
                       revealed question IDs
                       experiment history
                       best run ID

  evaluate_candidate / final scorer
       │
       ▼
  inspect-ai runner
       │
       ├── loads policy.py
       ├── calls the student model through StudentClient
       ├── grades with hidden targets
       └── returns accuracy and run artifacts
```

## Main abstractions

### `TaskDefinition`

Defined by Corral in `src/corral/core/task.py`.

Inference-opt creates one task definition per `(benchmark, student-model)` pair.
The task contains:

- the benchmark description shown to the teacher
- the list of teacher-facing tools
- immutable benchmark/model configuration in `initial_input`
- the prompt function
- the workspace setup function
- the final scoring function

`TaskDefinition` is configuration. It does not contain mutable experiment history.

### `InferenceOptEnvironment`

Defined in `inference_opt/env.py`.

This is the task-specific Corral environment subclass. It is responsible for:

- exposing the task's tools
- preparing the workspace
- initializing the inference session state
- executing trusted inference tools
- returning updated state through `ToolExecutionResult`

It does not retain the current execution state as an object attribute. Corral gives
it the current `ExecutionState` for every operation.

### `ExecutionState`

Defined by Corral in `src/corral/core/state.py`.

This is the immutable materialized projection of the committed event stream. It
contains the normal Corral execution fields:

- runtime status
- submitted answer
- actions and tool invocations
- workspace state
- environment state
- dependency outputs

Inference-opt stores its mutable session data inside:

```text
ExecutionState.environment.values
└── hidden_arguments
    ├── work_dir
    └── inference_state
```

The environment state is serialized JSON-shaped data. Corral commits the updated
namespace after every successful tool transition, so it survives replay, restart,
branching, and Docker task execution.

### `inference_state`

The current session state has this shape:

```json
{
  "experiments": 0,
  "debug_runs": 0,
  "student_calls": 0,
  "reveals": 0,
  "probe_calls": 0,
  "revealed_ids": [],
  "runs": [],
  "best_run_id": null,
  "limits": {
    "max_experiments": 20,
    "max_debug_runs": 10,
    "max_student_calls": 250,
    "max_reveals": 4,
    "max_probe_calls": 30,
    "reveal_batch": 5
  }
}
```

The state is initialized by `_prepare_workspace()` and then updated by trusted
tools.

### `StateLedger`

Defined in `inference_opt/budget.py`.

`StateLedger` is a small convenience wrapper over the mutable JSON mapping passed to a tool. It provides operations such as:

- `reserve()`
- `remaining()`
- `record_run()`
- `mark_revealed()`
- `best_run()`
- `snapshot()`

The source of truth is the mapping inside the Corral environment projection. `StateLedger` only gives tool implementations a convenient API for reading and updating that mapping.

## Tool execution flow

For a normal teacher action:

```text
1. Agent proposes an Action.
2. Corral records the action.
3. Corral materializes the current ExecutionState.
4. Corral injects hidden arguments from environment state.
5. InferenceOptEnvironment.execute_tool() receives the state.
6. The tool mutates a copied inference_state mapping.
7. The environment returns ToolExecutionResult:

       content = tool response
       environment = updated environment namespace

8. Corral commits the tool result and updated environment state.
9. The agent receives the tool response.
```

The environment makes a deep copy before tool execution. This prevents a failed tool from accidentally mutating the `ExecutionState` object supplied by Corral.

## Trusted tools

All inference-opt domain tools are currently marked with:

```python
@tool(hidden_args=["work_dir", "inference_state"], trusted=True)
```

This includes tools that execute the teacher-written policy. That is an explicit first-iteration trust assumption: teacher policy code runs as trusted task code. Docker can still provide the outer deployment boundary when needed.

## Workspace state versus Corral state

The two kinds of state have different roles.

### Corral environment state

Used for small, structured session state that must survive tool calls:

- budget counters
- revealed IDs
- run summaries
- best run ID

This state is committed through Corral events.

### Workspace files

Used for files that the teacher edits or needs to inspect:

```text
policy/
  policy.py
guide/
  policy_api.md
notes.md
TODO.md
revealed/
  train_revealed.jsonl
runs/
  <run-id>/
    questions.jsonl
    predictions.jsonl
    summary.json
    log/
submission.json
```

The policy source and detailed run artifacts are intentionally files. They are large, human-readable artifacts rather than compact execution-state fields.

The revealed training records are also written to a workspace JSONL file because the inspect runner consumes them during `Policy.setup()`.

## Evaluation layers

There are two different evaluations.

### Teacher-side candidate evaluation

`evaluate_candidate()`:

1. resolves `policy/policy.py`
2. creates a trusted question file for the train split
3. runs the inspect-ai adapter
4. reads the resulting accuracy/log artifacts
5. computes the candidate's improvement over the stored baseline
6. records a compact run summary in `inference_state`
7. writes detailed results below `runs/`

The inspect adapter remains in `inference_opt/runner/`. It handles conversion from the policy API to inspect samples, student-client calls, answer normalization, and inspect scoring.

### Final Corral scoring

The teacher eventually submits the string `submission.json`. Corral resolves that answer against the task workspace and invokes the task scoring function.

`policy_score()` then:

1. resolves the staged policy directory
2. loads the held-out test questions and hidden targets
3. runs the policy through the inspect runner
4. obtains policy accuracy
5. subtracts the stored scalar baseline accuracy
6. returns the final score

The policy-run labels are required internally by inspect to calculate policy accuracy. They are not exposed to the teacher as an answer-selection mechanism.
Item-level baseline pairing is not required for the scalar score.

## Why inspect-ai is retained

Inspect-ai provides the actual model-evaluation and grading machinery:

- model calls
- task/sample execution
- multiple-choice grading
- numeric grading
- logs
- accuracy metrics

Inference-opt supplies a policy solver and a metered student client so the policy controls prompting while inspect remains responsible for evaluation.

The current inspect adapter is therefore an evaluation abstraction, not another teacher-agent environment abstraction.

## Docker execution

When Corral runs the task in Docker, the task container receives a task-scoped workspace and Corral state volume. The same environment-state protocol applies:

```text
Docker container
├── /workspace       policy and run artifacts
└── /corral-state   committed Corral state and request/result files
```

Docker improves isolation and makes it possible to run the trusted first-iteration design in a contained task environment. It does not change the Corral state model.

The first iteration deliberately does not add another policy-execution boundary.

## Current deliberate simplifications

The first iteration intentionally does not use:

- a persistent Python REPL
- background evaluation jobs
- a file-backed budget ledger for active session state
- a separate restricted policy worker
- per-item baseline pairing for the primary score

The following abstractions remain because they still provide direct value:

- Corral `TaskDefinition` and `Environment`
- Corral committed `ExecutionState`
- Corral `ToolExecutionResult`
- `StateLedger` as a budget API
- the inspect-ai policy runner
- the policy API and answer normalization layer
- workspace files for source and detailed artifacts

## Main source map

| Responsibility | Source |
|---|---|
| Environment factory | `inference_opt/env.py` |
| Corral-native environment state | `inference_opt/env.py` |
| Teacher tools | `inference_opt/tools.py` |
| State budget wrapper | `inference_opt/budget.py` |
| Policy contract | `inference_opt/api.py` |
| Policy loading | `inference_opt/policy.py` |
| Inspect runner interface | `inference_opt/runner/spec.py` |
| Inspect execution | `inference_opt/runner/__main__.py` |
| Policy solver | `inference_opt/runner/solver.py` |
| Student client and per-run metering | `inference_opt/runner/runtime.py` |
| Answer normalization and grading specs | `inference_opt/scoring_specs.py` |
| Final scoring | `inference_opt/score.py` |
| Corral environment loading | `src/corral/runtime/environment_loader.py` |
| Corral action/tool transitions | `src/corral/core/transition.py` |
| Corral environment base class | `src/corral/core/environment.py` |
