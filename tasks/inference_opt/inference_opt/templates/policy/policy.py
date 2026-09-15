"""Starter policy: a single zero-shot call, which is exactly the baseline.

It runs end to end, so you can `dry_run_policy('policy')` right now and watch the
whole loop before changing anything. As written it will score ~0.0 improvement,
because it is the baseline. Everything interesting is what you do instead.

See `guide/policy_api.md` for the full contract.
"""

MANIFEST = {
    "name": "starter-zero-shot",
    "max_calls_per_question": 1,
}


class Policy:
    """Answers each question with one direct call to the student."""

    def solve(self, question, ctx):
        # If anything below fails, this is what gets scored. Always set it: a
        # cheap guess beats no answer when the budget runs out mid-run.
        ctx.scratch["fallback"] = (question.choice_labels or ("A",))[0]

        prompt = question.text
        if question.choices:
            prompt += "\n\n" + question.rendered_choices()
            prompt += "\n\nAnswer with the letter of the correct option."

        answer = ctx.student.generate(prompt, temperature=0.0, max_tokens=512)
        ctx.log(f"one call, {len(answer)} chars back")
        return answer
