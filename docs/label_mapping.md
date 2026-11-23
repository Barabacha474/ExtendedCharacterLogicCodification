# Label mapping

## DeBERTa-v3 check-scene (all released checkers)
Source: model README, https://huggingface.co/KomeijiForce/deberta-v3-base-check-scene
Decoding: `answer = [False, None, True][argmax(logits)]`

| logit idx | label | config id2label |
|---|---|---|
| 0 | False | LABEL_0 |
| 1 | None (Unknown) | LABEL_1 |
| 2 | True | LABEL_2 |

## Rule
All training, evaluation, and prediction serialization MUST use this mapping.
Never infer label order from data or from sorted class names.