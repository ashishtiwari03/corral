"""Build the review page for the proposed two-stage Corral MD tasks.

The page explains the proposal in plain language. The linked task JSON files
remain the source for the current, exact Level 2 instructions. Building this
page does not change any active task or scorer.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TASK_ROOT = ROOT.parent
DESIGN = ROOT / "paired_workflows.json"
OUTPUT = ROOT / "paired_workflows.html"


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def bullet_list(items: list[str]) -> str:
    return "<ol class=checks>" + "".join(f"<li>{escape(item)}</li>" for item in items) + "</ol>"


def load_current_level_2(number: int) -> dict:
    path = TASK_ROOT / f"environments/level_2/tasks_json/task_{number}.json"
    tasks = json.loads(path.read_text(encoding="utf-8"))
    if len(tasks) != 1 or tasks[0]["id"] != f"level_2_task_{number}":
        raise ValueError(f"Unexpected task shape or ID: {path}")
    return tasks[0]


def render_pair(proposal: dict) -> str:
    number = proposal["id"]
    load_current_level_2(number)
    l2_source = f"../environments/level_2/tasks_json/task_{number}.json"
    checks_l1 = bullet_list(proposal["level_1_checks"])
    checks_l2 = bullet_list(proposal["level_2_checks"])

    return f"""
    <details class="pair" id="pair-{number}" data-category="{escape(proposal['category'])}"{" open" if number == 1 else ""}>
      <summary>
        <span class="pair-number">{number:02d}</span>
        <span class="pair-heading"><strong>{escape(proposal['title'])}</strong><small>Level 1 stops here: {escape(proposal['cutoff'])}</small></span>
        <span class="category">{escape(proposal['category'])}</span>
      </summary>
      <div class="pair-body">
        <div class="definition-grid">
          <section class="level level-one" aria-labelledby="level-one-{number}">
            <div class="section-kicker">First part</div>
            <h3 id="level-one-{number}">Level 1 · Task {number}</h3>
            <h4>What to do</h4>
            <p>{escape(proposal['level_1_description'])}</p>
            <h4>What to turn in</h4>
            <p>{escape(proposal['level_1_submission'])}</p>
            <div class="score-panel">
              <h4>How this part will be checked</h4>
              <p class="score-equation">The checker opens the files you turn in and checks the information inside them. All {len(proposal['level_1_checks'])} checks must pass.</p>
              {checks_l1}
            </div>
          </section>
          <section class="level level-two" aria-labelledby="level-two-{number}">
            <div class="section-kicker">Full task · start from the original inputs</div>
            <h3 id="level-two-{number}">Level 2 · Task {number}</h3>
            <h4>What to do</h4>
            <p>{escape(proposal['level_2_description'])}</p>
            <h4>What to turn in</h4>
            <p>{escape(proposal['level_2_submission'])}</p>
            <h4>Extra files for this plan</h4>
            <p>{escape(proposal['level_2_submission_addendum'])}</p>
            <div class="score-panel">
              <h4>How the full task will be checked</h4>
              <p class="score-equation">The checker opens all submitted files. The {len(proposal['level_1_checks'])} Level 1 checks on the left and these {len(proposal['level_2_checks'])} extra checks must all pass.</p>
              {checks_l2}
            </div>
          </section>
        </div>
        <div class="source-row">
          <a href="{escape(l2_source)}">Read the current Level 2 task file ↗</a>
        </div>
      </div>
    </details>"""


STYLE = """
:root{color-scheme:light;--ink:#172a30;--muted:#56696e;--line:#d8e0df;--paper:#f5f7f4;--white:#fff;--teal:#0c726f;--teal-soft:#e9f5f2;--amber:#92581d;--amber-soft:#fbf1e4;--shadow:0 12px 36px rgba(18,49,48,.07)}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font:16px/1.58 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
a{color:var(--teal);text-underline-offset:3px}a:hover{color:#084b49}button,input,select{font:inherit}button{cursor:pointer}
.wrap{max-width:1240px;margin:0 auto;padding:0 26px}.hero{background:linear-gradient(125deg,#0a3839 0%,#125657 65%,#216b67 100%);color:white;padding:54px 0 56px}
.eyebrow{text-transform:uppercase;letter-spacing:.17em;font-size:.76rem;font-weight:800;color:#acded5}.hero h1{font-size:clamp(2.3rem,5vw,4.3rem);line-height:1.04;letter-spacing:-.045em;max-width:840px;margin:15px 0 19px}
.hero p{max-width:820px;color:#e4f3ee;font-size:1.12rem;margin:0}.hero .stats{display:flex;flex-wrap:wrap;gap:10px;margin-top:30px}.stat{padding:10px 16px;border:1px solid #ffffff56;border-radius:999px;background:#ffffff14;font-weight:700;font-size:.91rem}
main{padding:27px 0 75px}.section-title{margin:0 0 14px}.section-title h2{font-size:1.8rem;letter-spacing:-.02em;margin:0 0 4px}.section-title p{margin:0;color:var(--muted)}
.controls{position:sticky;top:0;z-index:3;background:rgba(245,247,244,.96);backdrop-filter:blur(12px);display:flex;align-items:center;gap:10px;flex-wrap:wrap;border-block:1px solid var(--line);padding:12px 0;margin-bottom:17px}
.controls label{font-weight:700;font-size:.9rem}.controls input{min-width:240px;flex:1;border:1px solid #aebfbd;border-radius:9px;padding:9px 12px;background:white;color:var(--ink)}.controls select,.controls button{border:1px solid #aebfbd;border-radius:9px;padding:9px 12px;background:white;color:var(--ink)}.controls button:hover{background:var(--teal-soft)}#shown{font-size:.83rem;color:var(--muted);margin-left:auto}
.pair{background:white;border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);margin-bottom:14px;overflow:hidden;scroll-margin-top:88px}.pair[hidden]{display:none}.pair>summary{display:flex;align-items:center;gap:14px;list-style:none;cursor:pointer;padding:18px 22px}.pair>summary::-webkit-details-marker{display:none}.pair>summary::after{content:"+";font-size:1.35rem;color:var(--teal);font-weight:500;margin-left:12px}.pair[open]>summary::after{content:"−"}
.pair-number{display:grid;place-items:center;flex:none;width:42px;height:42px;border-radius:12px;background:#e0efec;color:#0b6763;font-weight:800;font-size:.87rem}.pair-heading{display:flex;flex-direction:column;flex:1;min-width:0}.pair-heading strong{font-size:1.13rem}.pair-heading small{color:var(--muted);font-size:.88rem;margin-top:2px}.category{border:1px solid #d6e8e4;border-radius:999px;background:#f2f9f7;color:#246d68;padding:4px 10px;font-size:.75rem;font-weight:700;white-space:nowrap}
.pair-body{border-top:1px solid var(--line)}.definition-grid{display:grid;grid-template-columns:1fr 1fr}.level{padding:24px 27px 28px;min-width:0}.level-one{background:#f4faf8}.level-two{border-left:1px solid var(--line)}.section-kicker{text-transform:uppercase;letter-spacing:.12em;font-size:.68rem;font-weight:800;color:var(--teal)}.level-two .section-kicker{color:var(--amber)}
.level h3{font-size:1.35rem;line-height:1.25;margin:6px 0 20px}.level h4{font-size:.84rem;text-transform:uppercase;letter-spacing:.08em;margin:21px 0 5px;color:#456065}.level p{margin:0;color:#263e43}.checks{padding-left:22px;margin:6px 0 0}.checks li{padding-left:4px;margin:0 0 8px;color:#263e43}.checks li::marker{color:var(--teal);font-weight:800}
.score-panel{background:#fff;border:1px solid #b9d9d4;border-left:5px solid var(--teal);border-radius:11px;padding:17px 19px 12px;margin:0 0 23px}.level-two .score-panel{background:#fffaf4;border-color:#e4cfb4;border-left-color:#ad722a}.score-panel h4{margin:0 0 7px;font-size:.95rem;color:#1c4b49}.level-two .score-panel h4{color:#724613}.score-panel .score-equation{font-weight:500;line-height:1.48;color:#263e43}.score-panel .checks{margin-top:12px}.score-panel .checks li{margin-bottom:10px}.level-two .checks li::marker{color:#a16622}
code{font-size:.86em}.source-row{border-top:1px solid var(--line);padding:13px 27px 16px;font-size:.88rem}
.footer{border-top:1px solid var(--line);padding:28px 0 50px;color:var(--muted);font-size:.9rem}.footer p{max-width:860px}
@media(max-width:850px){.definition-grid{grid-template-columns:1fr}.level-two{border-left:0;border-top:1px solid var(--line)}.category{display:none}.controls{top:0}.pair>summary{padding:15px}.pair-heading strong{font-size:1rem}}
@media(max-width:560px){.wrap{padding:0 15px}.hero{padding:37px 0}.hero p{font-size:1rem}.level{padding:19px}.controls input{min-width:100%;order:2}.controls select{flex:1}.controls label{display:none}#shown{display:none}}
@media print{body{background:white}.hero{color:#172a30;background:white;padding:10px 0}.hero p,.eyebrow{color:#172a30}.controls{display:none}.pair{box-shadow:none;break-inside:avoid;display:block}.pair>summary::after{display:none}.definition-grid{grid-template-columns:1fr 1fr}.level-two{border-left:1px solid var(--line);border-top:0}a{color:inherit}}
"""


SCRIPT = """
const search = document.querySelector('#search');
const category = document.querySelector('#category');
const cards = Array.from(document.querySelectorAll('.pair'));
const shown = document.querySelector('#shown');
function filterCards() {
  const query = search.value.trim().toLowerCase();
  const selected = category.value;
  let visible = 0;
  for (const card of cards) {
    const searchable = card.querySelector('.definition-grid').textContent.toLowerCase();
    const matches = (!query || searchable.includes(query)) &&
      (selected === 'all' || card.dataset.category === selected);
    card.hidden = !matches;
    if (matches) visible += 1;
  }
  shown.textContent = `${visible} of ${cards.length} tasks shown`;
}
search.addEventListener('input', filterCards);
category.addEventListener('change', filterCards);
document.querySelector('#expand').addEventListener('click', () => {
  for (const card of cards) if (!card.hidden) card.open = true;
});
document.querySelector('#collapse').addEventListener('click', () => {
  for (const card of cards) card.open = false;
});
if (location.hash) {
  const target = document.getElementById(location.hash.slice(1));
  if (target?.classList.contains('pair')) target.open = true;
}
"""


def build() -> str:
    data = json.loads(DESIGN.read_text(encoding="utf-8"))
    pairs = data["pairs"]
    if sorted(pair["id"] for pair in pairs) != list(range(1, 11)):
        raise ValueError("Expected exactly one proposal for each task ID 1–10")
    categories = sorted({pair["category"] for pair in pairs})
    category_options = "".join(
        f'<option value="{escape(item)}">{escape(item)}</option>' for item in categories
    )
    cards = "\n".join(render_pair(pair) for pair in sorted(pairs, key=lambda p: p["id"]))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A plain-language plan for two-part Corral MD tasks and file-based checks.">
  <title>Corral MD · Two-part task plan</title>
  <style>{STYLE}</style>
</head>
<body>
  <header class="hero"><div class="wrap">
    <div class="eyebrow">Corral MD · task plan</div>
    <h1>Ten tasks, each in two parts.</h1>
    <p>Level 1 covers the first part of a task. Level 2 starts from the original inputs, does that first part, and then finishes the task. This page explains what to do, what to turn in, and how the submitted files would be checked.</p>
    <div class="stats"><span class="stat">10 tasks</span><span class="stat">2 parts each</span><span class="stat">All checks must pass</span></div>
  </div></header>
  <main class="wrap">
    <div class="section-title"><h2>Choose a task</h2><p>Open a task to see both parts. Level 2 must also pass the checks shown for Level 1.</p></div>
    <div class="controls" role="search">
      <label for="search">Find a task</label><input id="search" type="search" placeholder="Search for a material or topic…" autocomplete="off">
      <select id="category" aria-label="Choose a topic"><option value="all">All topics</option>{category_options}</select>
      <button type="button" id="expand">Open all</button><button type="button" id="collapse">Close all</button>
      <span id="shown" aria-live="polite">10 of 10 tasks shown</span>
    </div>
    {cards}
  </main>
  <footer class="footer"><div class="wrap"><p>This is a plan. The new checks will apply only after the tasks and scoring code are updated. The original Level 2 task files are linked below each task.</p><p>Page data: <a href="paired_workflows.json">paired_workflows.json</a>. To update this page after editing that file, run <code>python build_paired_workflows.py</code>.</p></div></footer>
  <script>{SCRIPT}</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if HTML is out of date")
    args = parser.parse_args()
    rendered = build()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text(encoding="utf-8") != rendered:
            raise SystemExit(f"Out of date: {OUTPUT}")
        print(f"Up to date: {OUTPUT}")
    else:
        OUTPUT.write_text(rendered, encoding="utf-8")
        print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
