# Split decision (Check_Scenes)

## Question
Can the original 90/10 split be recovered from released artifacts?

## Evidence
- Official repo commit: 9bd6d82f6ed4cf7dde5d771b75877f00c2c8c6e0
- Official repo contains no training script and no split files.
- `codes/` contains only Koishi.code.json (10 parse_by_scene blocks).
- Check_Scenes: 20,759 rows, columns = [prompt, answer]. No `split`, no `character`, no `scene_id`.
- Check_Scenes class distribution: True 8290 / False 7084 / None 5385.
- Fandom_Benchmark: 5,141 rows, columns = [artifact, genre, character, scene, question, reference]. Character is explicit.
- Fandom_Codified_Profiles: 1,027 rows, columns = [character, segment, code]. Per-segment code.
- Consequence: C1/C2/C4 cannot be built on Check_Scenes (no character link). Extension dataset will be built from Fandom_Benchmark (Koishi subset).

## Decision
- [ ] Exact split recovered → используем его, фиксируем индексы.
- [ ] Not recovered → replacement split:
  - random 90/10, seed=43
  - scene-grouped 90/10 (если есть scene id)
  - индексы сохраняются в data/splits/check_scenes_{random,grouped}.json

## Deviation note
Явно перечисляем, чем наш сплит отличается от оригинального.