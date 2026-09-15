# Inference-time optimization environment

This environment asks a teacher agent to write a Python policy that improves a
frozen student model at test time. The policy controls prompts, sampling,
aggregation, memory, and answer extraction; it does not change model weights.

There are six benchmarks (`gsm8k`, `mmlu_pro`, `gpqa_diamond`, `bbh`, `chembench`,
`arc_challenge`). Level 1 evaluates one student per task; level 2 evaluates one
policy against two students and scores the smaller improvement. The final score is
the accuracy improvement over the measured zero-shot baseline on held-out test
questions.

## Run

```bash
cd tasks/inference_opt
uv sync
uv run python -m inference_opt.env --level 1
uv run pytest tests -q
```

Set `CORRAL_VLLM_URL` (or a model-specific URL) when using a real student server.
The committed tasks use placeholder baseline values until
`scripts/measure_baselines.py` has been run.

## Agent tools

The domain tools are trusted Corral tools and use Corral's committed environment
state. Workspace file tools create and edit the submitted policy.

| tool | purpose |
| --- | --- |
| `get_baseline` | Show the measured train baseline and topic breakdown. |
| `reveal_train_questions` | Spend a reveal to unlock labelled train examples. |
| `query_student` | Probe the student directly under the probe budget. |
| `dry_run_policy` | Run the policy on a small train sample. |
| `evaluate_candidate` | Evaluate and record a full train experiment. |
| `inspect_failures` | Inspect predictions and logs from an earlier run. |
| `compare_runs` | Show recorded experiments and the current best. |
| `get_budget` | Show remaining session budget. |
| `submit_policy` | Stage `submission.json` for final scoring. |

## State and execution

Corral owns the session state. Each trusted tool receives a JSON-shaped
`inference_state` namespace and returns its updated state in `ToolExecutionResult`.
The environment commits that state after every call. No active budget or run ledger
is stored in the workspace filesystem.

The policy evaluation itself runs through Inspect AI in a dedicated eval-host
process. That process receives only the questions materialised for its run, uses a
metered `StudentClient`, writes predictions and Inspect logs, and returns a summary.
The teacher-facing tools remain synchronous; there is no background job or
persistent policy REPL.

This first iteration trusts teacher-written policy code. The client is the only
supported model interface, but policy execution is not an AST sandbox or a
restricted Corral worker. Docker can provide the outer deployment boundary when
needed.

## Source map

- `inference_opt/env.py`: Corral environment and state handoff.
- `inference_opt/tools.py`: trusted teacher-facing tools.
- `inference_opt/budget.py`: local meters and `StateLedger`.
- `inference_opt/policy.py` and `api.py`: policy loading and policy/client contract.
- `inference_opt/runner/`: Inspect AI eval host, solver, runtime, and summaries.
- `inference_opt/outcomes.py`: Inspect-log outcome parsing.
- `inference_opt/score.py`: held-out scoring and baseline delta.
- `inference_opt/datasets.py`: packaged questions and targets.
- `architecture.md`: detailed design and call flow.
