# Two-part Corral MD task plan

Open [`paired_workflows.html`](paired_workflows.html) in a browser. On a Mac,
you can run this from the repository root:

```bash
open tasks/corral_md/new_tasks/paired_workflows.html
```

The page works without installing an app.

To use the links to the current task files, start a local server from the
`corral_md` directory:

```bash
cd tasks/corral_md
python -m http.server 8000
```

Then open <http://localhost:8000/new_tasks/paired_workflows.html>.

The plain-language task text and planned checks are in
[`paired_workflows.json`](paired_workflows.json). The planned checker must open
the submitted files, read the data inside them, compare related files, and
recalculate selected results. A file name or a claimed setting alone cannot
earn a passing score.

After editing the plan, rebuild the page with:

```bash
cd tasks/corral_md/new_tasks
python build_paired_workflows.py
```

Use `python build_paired_workflows.py --check` to confirm the page matches the
plan. This is a plan for future tasks and checks; the current tasks and scorers
do not use it yet.
