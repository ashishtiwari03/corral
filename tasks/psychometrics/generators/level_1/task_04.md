# Level 1 / Task 04 — Which instrument supports a gender comparison?

**Ask.** Two instruments were administered. Determine which one measures the same
thing in both groups, report the gender difference on it, and list the items that
function differently across gender in the other.

**Generator.** `gen_l1_t04_invariant_combination.py`

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t04_invariant_combination.py [--verify|--naive]
```

## Generating model

| instrument | structure | gender | biased items |
|---|---|---|---|
| HSNS | one factor | women **+0.45** on the trait | **4** — HSNS1, 5, 8, 10, thresholds raised by .55 against women |
| Dirty Dozen | general factor + 3 specifics | women **+0.30** on the general factor | **none** — fully invariant |

Outside the US there is no bias, and the Dirty Dozen gender difference runs the
other way (−0.25), so pooling the countries moves the answer out of tolerance.

## Why it is not a one-liner

Taking each instrument at face value rejects **both**:

```
HSNS (one factor)          -> biased items found
Dirty Dozen (3 subscales)  -> biased items found  [DDN4]
conclusion: neither instrument supports the comparison
truth: the Dirty Dozen does, and the difference is +0.147
```

- The HSNS bias is real, and it **hides** the trait difference: the observed
  gap looks negligible, because bias against women cancels a genuine +0.45.
- The Dirty Dozen has no bias at all, but under its **nominal subscale
  structure** an item still looks biased. When the groups differ on a general
  factor and the model has no general factor, that difference has to be absorbed
  item by item — and because the general loadings vary, the absorption is uneven
  and reads as bias.
- So the invariance verdict depends on the measurement model. Getting L1-T2's
  answer right for the Dirty Dozen is a precondition for getting this one right.

| route | latent difference | |
|---|---|---|
| HSNS, assumes no bias | +0.028 | wrong instrument, wrong answer |
| HSNS, bias freed | +0.195 | right analysis, wrong instrument |
| DD, nominal three-factor | +0.107 | right instrument, wrong model |
| **DD, bifactor** | **+0.144** | correct (target +0.147) |

**Finding the biased items** needs freeing every gender path at once and reading
the pattern: the four biased items come out negative (−0.109 to −0.131) and the
six clean ones positive (+0.092 to +0.148). Scanning items one at a time does
not work — the no-bias baseline is itself contaminated by the bias, so every
item looks significant.

## Scored

The same three stages and the same scorer.

- **Instrument choice** is derived from which variables the model uses, so a
  submission analysing the HSNS fails before anything is fitted.
- **Tier 2** rejects the nominal three-factor model on fit and accuracy.
- **Tier 3** checks the reported difference (±.06 of the calibrated target), that
  the selected model claims **no** item bias (derived from its gender paths —
  freeing spurious ones fails), and the biased-item set in the rejected
  instrument.

Submission: `model_syntax`, `latent_difference`, `biased_items`.
