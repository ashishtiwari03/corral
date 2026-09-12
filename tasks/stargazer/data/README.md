# Stargazer task bank

`synthetic/` contains 100 synthetic task IDs (ten at each difficulty from 1
through 10). `real/` contains the 20 released archival RV tasks. The original
records were copied from Stargazer revision
`3f617667472061e253288c7b26f0e70f186f2dff`, licensed CC BY 4.0.

Eight synthetic records now use the preferred versions from the supplied
`corral_exoplanet_rv.zip`: `seed15_diff5`, `seed64_diff6`, `seed43_diff7`,
`seed1_diff8`, `seed17_diff8`, `seed101_diff9`, `seed93_diff9`, and `seed82_diff10`.
Each keeps its original Stargazer ID, preventing duplicates under the ZIP's
`exo_rv_*` aliases. Observations, reference planets, and stellar masses are
copied exactly from the ZIP into the native JSON schema. `meta.rv_semantics`
marks them as already converted to RV-only coordinates. Unknown generation
parameters are omitted rather than inherited from older records.

The ZIP versions of `seed43_diff7`, `seed93_diff9`, and `seed82_diff10` contain
repaired systems (including different observations and orbits); their original
local versions failed the reference RMS gate. The other five selected exports
match the local normalized systems within numerical precision.

`selection_manifest.json` records inspection counts, all ZIP aliases, the
archive SHA-256, selection policy, and membership/provenance for the two
official levels. Level 1 selects 10 synthetic tasks at difficulties 5–7
(3 ZIP, 7 existing); Level 2 selects 10 at difficulties 8–10 (5 ZIP, 5 existing).
Unused source records remain available for custom selectors and auditing.

`reference_audit.json` is a deterministic Corral-generated report of every
published reference submission under the unchanged final scoring contract.
Regenerate or verify it with `python -m stargazer.audit` or
`python -m stargazer.audit --check`, respectively.

See `../THIRD_PARTY_NOTICES.md` for full attribution.
