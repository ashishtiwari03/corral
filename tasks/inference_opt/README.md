# Inference-time LLM Optimization Environment

This environment evaluates a teacher agent that improves frozen student LLMs
using inference-time policies. A policy may contain Python code, prompts,
demonstrations, memory, routing, verification, and sampling logic, but it may
not modify or inspect student-model weights.

The student models are expected to be exposed through OpenAI-compatible vLLM
servers. Dataset files use JSONL records with the following fields:

```json
{"id": "q1", "question": "...", "answer": "...", "choices": ["..."], "category": "math"}
```

The evaluator deliberately separates development/validation/test data. Only
development results may include per-question diagnostics; final scoring runs
the submitted policy independently on the test split.

The initial benchmark set is GSM8K, GPQA, MMLU-Pro, ChemBench, and AIME. The
dataset adapters are intentionally small and generic; benchmark-specific
parsing should be added in `inference_opt/adapters.py` as the data sources are
selected.

## Policy contract

The submitted directory must contain `policy.py` with:

```python
def solve(question: str, model_client, context: dict) -> str:
    ...
```

`model_client.generate(prompt, ...)` is the only permitted student-model
interface. `submit_policy` validates this contract before accepting the final
artifact.
