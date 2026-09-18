# Task 05 — duplicate records or chance collisions?

## The task

A survey panel delivered 12,965 records. 2,258 of them share an answer pattern
with at least one other row. Decide what is producing that, and report the
correlation between the two traits the data actually support.

The agent states which columns together identify a record as the same
respondent as another. One row is kept per distinct combination of those
columns, the submitted model is refitted on what remains, and the correlation is
read from that refit.

## The data

Ten HSNS items drawn from two correlated traits, true r = .35:

| trait | items | loadings |
|---|---|---|
| F1 | HSNS1 4 5 6 8 10 | .55 .30 .70 .48 .71 .66 |
| F2 | HSNS2 3 7 9 | .76 .58 .69 .52 |

12,000 distinct respondents, no missing responses.

Put in deliberately: 965 respondents were delivered a second time, with a fresh
`participant_id`, `session_id` and `recorded_on`, but the same answers, age and
accuracy. Re-delivery probability rises with the sum of both traits, so the
repeated block is the high-scoring end of the sample and inflates the
correlation.

Nothing else was done. The other repeated patterns are two different people who
happened to answer identically — ten five-point items cannot tell 12,000 people
apart, and 328 rows collide by chance.

## The answer

`both_present`, key = the ten items with `age` and `accuracy`, r = .342.

| what gets removed | rows left | r |
|---|---|---|
| nothing | 12,965 | +.398 |
| every row in a repeated pattern | 10,707 | +.226 |
| one row kept per answer pattern | 11,730 | +.318 |
| one row kept per answers + age + accuracy | **12,000** | **+.342** |

## Traps

**Cleaning the anomaly away.** Dropping every row in a repeated pattern lands at
+.226 against a truth of +.350 — *further* from the answer than doing nothing at
all. The cleaning reflex is worse than no cleaning here.

**Deduplicating on the answers.** The obvious key gives +.318 and silently
deletes 270 real respondents. It is close enough to look right, and the retained
row count is what gives it away.

**Keying on everything.** `session_id` and `recorded_on` were reissued, so a key
including them finds no duplicates at all and leaves the file as delivered. The
agent has to reason about which columns a re-delivery would preserve.

**Assuming one mechanism.** Both are present. 1,930 of the 2,258 shared rows are
re-delivery, 328 are chance. Either single-mechanism answer is wrong.

## Scoring

Four things must all hold.

| check | what it catches |
|---|---|
| `diagnosis` | assuming one mechanism instead of two |
| retained respondents, derived | a key that merges real people or none at all |
| repeats remaining, derived | a key that leaves duplicates in |
| corrected correlation, derived | the wrong answer from the wrong sample |

Only the label is taken from the submission. The other three are computed by
applying the submitted key to the data and refitting the submitted model, so a
key that does not do what the agent claims is caught regardless of what the
agent reports.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_2/gen_l2_t05_duplicate_records.py [--verify|--naive]
```
