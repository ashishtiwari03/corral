# Inference-time optimization environment

A teacher agent is given a **frozen** student LLM (7–9B instruct, served over an
OpenAI-compatible vLLM endpoint) and a subset of benchmark questions. It cannot
touch the weights. It can change everything that happens *around* the model at test
time — prompts, how many samples are drawn and how they are combined, whether one
call checks another, what is remembered between questions, how the final answer is
extracted — and it writes that as a Python **policy**.

It iterates under hard budgets against a training split, then submits. The score is
the **improvement over the student's measured zero-shot baseline** on a held-out
test split.

The question the environment asks is whether an agent can find and implement
test-time strategies that *generalise*, rather than fitting the questions it can see.

## Setup

```bash
cd tasks/inference_opt
uv sync
uv run python -m inference_opt.env --level 1     # list the tasks
uv run python -m pytest tests -q                 # no GPU needed
```

Point the environment at a running student:

```bash
export CORRAL_VLLM_URL=http://127.0.0.1:8000     # or CORRAL_VLLM_URL_<MODEL>
```

Run one task:

```bash
uv run corral bench --agent tool-calling --environment inference_opt \
    --task gsm8k_a --model openai/gpt-5.6
```

## Tasks

Six benchmarks — `gsm8k`, `mmlu_pro`, `gpqa_diamond`, `bbh`, `chembench`,
`arc_challenge` — each with 30 train and 30 test questions.

* **Level 1** (12 tasks): one student per task. Score = improvement on that student.
* **Level 2** (12 tasks): two students share one policy. Score = the **smaller** of
  the two improvements, so a policy must help both rather than trading one off
  against the other.

## The agent's tools

| tool | costs | what it is for |
| --- | --- | --- |
| `get_baseline` | — | the zero-shot accuracy to beat, broken down by topic |
| `reveal_train_questions` | 1 reveal | unlock 5 labelled train questions plus what the student answered zero-shot |
| `query_student` | calls | probe the student directly, without writing a policy |
| `dry_run_policy` | 1 dry run | **does my strategy actually run?** Identical pipeline, two questions, full traces |
| `evaluate_candidate` | 1 experiment | scored run on the train split, with paired statistics |
| `inspect_failures` | — | per-question prompts, answers and logs from an earlier run |
| `compare_runs` | — | the run ledger, best first, with an honest noise warning |
| `get_budget` | — | what is left and what it buys |
| `submit_policy` | — | validate and stage the final answer |

Corral's workspace file tools are also present: the agent writes `policy.py` with
them, and keeps `notes.md` and `TODO.md` as a scratchpad. A runnable starter policy
and `guide/policy_api.md` are seeded into the workspace, so the first dry run works
before anything is changed.

## The one hard rule

**A policy may use only the student client it is handed.** No other model, no
network, no filesystem outside its own directory, no reflection to reach any of
those. An AST validator enforces it at three points — `dry_run_policy` (advisory,
so the agent learns the rule while it still has budget), `submit_policy` (refuses to
stage), and scoring (refuses to run).

### Threat model, and what is *not* covered

The validator denies every non-stdlib import, `open`, `eval`/`exec`/`__import__`,
`getattr`/`setattr`, and dunder or foreign private attribute access. The eval host
additionally runs in a separate process with provider credentials scrubbed and no
path to the frozen dataset, and the controller cross-checks the reported call count
against the model events inspect recorded independently. A policy that beats the
baseline while making zero student calls scores zero.

What this is **not** is a sandbox. Policy code executes in a normal Python process,
and grading happens in that same process, so the targets for the items in a run are
reachable in principle — the import allowlist is what stands in the way, and that is
a checked guarantee rather than a structural one. It defends against accidental
leakage and casual shortcuts, not a determined exfiltrator. A real guarantee needs a
container boundary. The environment assumes a non-adversarial teacher.

## Baselines

Every score is relative to the baseline, so the baseline *is* the benchmark. It is
pinned: one call per question, `temperature=0.0`, `seed=0`, **no system message**,
zero-shot, and an unparseable response counts as wrong. It is measured through the
same runner, solver and grading a submitted policy goes through.

```bash
uv run python scripts/measure_baselines.py \
    --model model_a=Qwen/Qwen2.5-7B-Instruct \
    --base-url http://127.0.0.1:8000/v1 --write-tasks
```

This prints a **headroom table** and refuses any (benchmark, model) pair whose
baseline sits at the ceiling or the floor — such a pair cannot be moved by any
strategy, so scoring agents on it would be scoring noise. Run this before running
agents; the task JSONs ship with placeholder zeros until it has.

The agent is shown the *train* baseline; the scorer subtracts the *test* baseline.
Showing the test number would leak how hard the held-out split is.

## What a score can and cannot tell you

**With 30 paired test items, this environment can distinguish a policy that adds
about 20 accuracy points from one that adds nothing. It cannot distinguish +5 from
+10, and it cannot rank two policies whose true improvements differ by less than
roughly 15 points.**

Pooling all six benchmarks (180 paired items per student) brings the detectable
effect down to about 7 points, so **the pooled improvement is the headline number**
and per-benchmark scores are a breakdown, not six independent verdicts.

Every comparison reports `n_changed` — how many questions actually changed outcome.
A delta of +0.10 with `n_changed = 3` got lucky three times; the same delta with
`n_changed = 15` is a policy that is genuinely doing something *and* causing some
harm. The score alone cannot tell those apart, and they are entirely different
results.

## Architecture

Two processes, because Corral's permission model forces it. A non-`trusted` tool is
cloudpickled into a fresh privilege-dropped subprocess on every call and only its
return value comes back — so any counter held in a `create_tools()` closure is
silently discarded. That isolation is only active under Docker, so an in-memory
design passes every local test and loses state in the real benchmark.

| | controller (tools, scoring) | eval host (`python -m inference_opt.runner`) |
| --- | --- | --- |
| serves labels to the agent | yes, via trusted tools | no |
| durable budget and run ledger | `state/ledger.json` | in-memory enforcement only |
| imports policy code | never | yes |
| network | — | the student endpoint only |

All durable state is workspace files:

```
policy/                    the agent's policy.py and its artifacts
state/ledger.json          budget, run history, best run
submission.json            staged submission
revealed/                  labelled train questions the agent unlocked
runs/<id>/                 summary.json, predictions.jsonl, inspect logs
guide/, notes.md, TODO.md
```

Evaluation uses `inspect_ai`. The policy owns all prompting, so the solver never
calls `generate` itself; it hands the policy a metered client and rewrites whatever
the policy returns into the canonical `ANSWER: <value>` form that inspect's own
scorers (`choice`, `match(numeric=True)`, plus two small custom ones for chembench)
expect. Grading is therefore inspect's, not hand-rolled.

## Budgets

Per task: 20 experiments, 10 dry runs, a total student-call allowance, and 4 reveals
of 5 questions. Each question in a run gets a guaranteed reserve plus access to a
shared pool, so spending heavily early cannot starve later questions.

When the budget runs out the client raises `BudgetExhausted`, **and the remaining
questions still run** — every call raises immediately, so a policy that degrades
gracefully still scores. Unattempted questions count as wrong; the denominator is
always the whole split.

## Scoring outcomes

`score.py` distinguishes why a run scored what it did, because the previous
implementation caught every exception and returned `0.0`, which reported a config
typo, a missing file and a refused connection identically to "the agent did badly".

| outcome | score |
| --- | --- |
| `ok` | improvement over baseline, clamped to `[0, 1]` |
| `policy_invalid` | 0.0 — the policy failed the stated contract |
| `policy_crashed` | per-item containment; 0.0 only above a 50% crash rate |
| `budget_exhausted` / `timeout` | partial credit, constant denominator |
| `suspected_cheating` | 0.0, with the evidence recorded |
| a harness fault | **raises** — never recorded as an agent score |

A diagnostics sidecar is written to `state/scoring_diagnostics.json` with per-model
accuracy, paired McNemar counts, call usage, and per-item outcomes.

## Notes

* The question set is committed and fixed. Corral's shared contract suite builds
  every task in every level with no network and no GPU, so nothing may be fetched
  or computed at import time.
* `inference_opt` is already registered in `src/corral/runtime/environment_loader.py`.
* `CORRAL_INFERENCE_DATA_DIR` overrides the packaged dataset location;
  `CORRAL_INFERENCE_LABELS_PATH` overrides where labels are read from.

## How the questions were chosen

The question set is fixed and committed; there is no selection step at run time.

Each benchmark contributes 60 questions, split 30 train / 30 test. Items were kept
only where a **reference cohort of 7–8B models scored between 20% and 80%**. Outside
that band a question is answered correctly, or not, almost regardless of how it is
asked, so no test-time strategy could move it and including it would only add noise.
The cohort is drawn from published per-question results for models of that size —
88 of them for BBH and ARC, 84 for MMLU-Pro, ~3000 for GSM8K, 5 for GPQA-Diamond.
ChemBench has no small-model results available, so its four weakest available
respondents (45–55% accuracy) stand in.

Within that band, questions are sampled uniformly across each benchmark's
categories, preferring those the cohort is most evenly split on, and the train/test
halves are matched on both category and difficulty. Across the six benchmarks the
cohort averages 42–61% on the selected questions, which is the range where a
7–8B student has room to move in either direction.

Two source-data details worth knowing: BBH questions ship as pre-rendered 3-shot
prompts, so only the final question is kept — otherwise a "zero-shot" baseline would
not be zero-shot; and ChemBench mixes lettered, multi-select and free-numeric
answers, which are scored separately.

`manifest.json` records the protocol, the per-benchmark counts and cohort sizes, and
a `content_fingerprint` pinning exactly which questions are in which split.
