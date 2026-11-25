## Empirical depth profile (Edward, measured)

- blocks: 21, conditions: 100
- depth distribution: {0:81, 1:13, 2:3, 3:1, 4:1, 5:1}
- conditions with non-empty parent chain: 19 / 100

## Consequence for C2 / C4

- For 81% of conditions, C2 (parent context) is empty -> C2 == C0 by construction.
- Therefore:
  - C0 vs C2 primary comparison MUST be reported on the `has_parents` subset (n=19 conditions).
  - Full-set C0 vs C2 is reported too, but explicitly as diluted / null-by-construction for most cases.
  - C0 vs C4 reported on full set (character description applies everywhere)
    AND on has_parents subset (where both description and parents are present).
  - C1 vs C4 primary comparison on has_parents subset.
- This is a pre-registered analysis choice; not changed after seeing results.

## Target character decision (revised v2)

- Touhou/Koishi: NOT in Fandom Benchmark. Koishi.code.json = parser fixture only.
- Fandom_Benchmark questions are behavioral ("What does X do?"), NOT True/False/Unknown.
  -> Cannot be used as condition-checker inputs directly.
- Extension dataset = (condition questions from Fandom_Codified_Profiles)
                       x (scenes from Fandom_Benchmark for the same character).
  Labels from Ollama teacher (no existing labels).
- Candidate: Edward (21 codified segments, 178 benchmark scenes).
  Fallbacks: Polnareff (18,195), Katara (16,202).
- Selection gated by:
  - code must contain nested check_scene calls (depth > 1),
  - scenes must fit 512 tokens with question + minimal context.
- C0-C4 built on the cross-product; same (scene, question) pairs across variants.

## Final target character: Edward

- Chosen after comparing Edward / Polnareff / Katara.
- All three below the pre-registered threshold (with_parents >= 30); none switched.
- Rationale: highest condition count (100), highest scene count (178), richest depth (max 5).
- Alternative candidates (for sensitivity check, if time permits): Polnareff, Katara.

## Pre-registered analysis (do not change after seeing results)

- Full set: C0 vs C4, C1 vs C4 reported on full set.
- Subset `has_parents` (n=19 conditions): C0 vs C2, C1 vs C4, C2 vs C3.
- All subset analyses labeled as "subset (n=19, parent-bearing conditions)".