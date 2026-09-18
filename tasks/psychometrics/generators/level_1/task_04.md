# Task 04 — which questionnaire supports a comparison between men and women?

## The task

Both questionnaires were given to everyone. Select the US respondents and decide which questionnaire measures the same thing in both sexes, then report the size of the difference on it and which items of the other are unfair.

## The data

Both questionnaires were generated for every respondent.

- **HSNS**: one trait, loadings .48–.76. Women really do hold .45 more of it, but four items (1, 5, 8, 10) are shifted .55 against them, which hides it.
- **Dirty Dozen**: one broad trait plus three narrow ones, identical in both sexes. Women hold .30 more of the broad trait. Nothing is unfair.

Put in deliberately: outside the US no item is unfair and the Dirty Dozen difference runs the other way (−0.25), so pooling the countries moves the answer rather than merely blurring it.

## The answer

The Dirty Dozen is the usable one, and shows women slightly higher on its broad trait. The HSNS is not: four of its items are answered differently by men and women who hold the same amount of the trait.

## Traps

- **The usable questionnaire looks unusable.** Fitted with its published subscale structure, the Dirty Dozen appears unfair on three items that are perfectly fair. An agent that takes each questionnaire at face value concludes neither can be used.
- **Testing one item at a time fails.** It finds unfairness almost everywhere,  because the baseline it compares against is itself contaminated by the unfairness being looked for. 
  All items have to be freed at once, after which the truly unfair ones separate by the direction of their effect.
- **The wrong sample moves the answer, not just blurs it.** Outside the US no item is unfair and the sex difference runs the other way, so pooling the countries pushes the result well outside tolerance.

## Scoring

Stage 3 checks the size of the difference, the items reported as unfair, and that the submitted model does not claim unfairness in the questionnaire that has none.

## Rebuild

```bash
uv run --with numpy --with pandas --with scipy --with semopy \
  python generators/level_1/gen_l1_t04_invariant_combination.py [--verify|--naive]
```

`--verify` prints the numbers behind all of the above. `--naive` shows the
obvious analysis failing.
