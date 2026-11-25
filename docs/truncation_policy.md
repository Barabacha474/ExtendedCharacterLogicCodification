# Truncation policy

## Fixed budget
max_length = 512 (DeBERTa-v3 max_position_embeddings)

| slot | tokens |
|---|---|
| instruction | 16 |
| question | 80 |
| character description | 64 |
| parent conditions | 64 |
| scene | 288 |

## Rules
- Scene budget = 288 tokens for ALL variants C0-C4 (matched input).
- Scene > 288 tokens -> head 200 + tail 88 + "..." marker.
- Scene <= 288 tokens -> kept as is.
- Unused slots are NOT reallocated to scene (prevents confound).
- Primary set: scenes with len_chars <= 1150 (~288 tokens).
- Secondary set: 1150 < len_chars <= 2400, head+tail truncation, flagged as partial-context.

## Never
- Never truncate the question.
- Never truncate the target condition's own text.
- Never reallocate scene budget based on variant.