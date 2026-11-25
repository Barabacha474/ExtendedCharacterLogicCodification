## Teacher (locked after Stage 0 diagnosis)

- Model: `gemma4:e4b` via Ollama
- Endpoint: `/api/chat`
- System prompt: strict condition-checker + 3 few-shot examples (True/False/Unknown)
- Options: temperature=0, seed=43, num_ctx=4096
- Agreement with GPT-4.1 on 80 Check_Scenes cases: **66.2%** (best across 6 models tried)
- Label distribution (80 cases): None 46 / True 19 / False 15 (GPT: 30/28/22)
- Known bias: None-leaning; catches None 87%, True 64%, False 46%.
- Rationale: we measure **context sensitivity**, not GPT-4.1 fidelity. Systematic bias is acceptable; C0–C4 share the same teacher.

## Ollama options (locked)
- num_ctx: 4096 (8192 caused context spillover to CPU and changed label behavior)
- num_predict: 1024 (16 and 252 caused >50% parse errors; gemma4:e4b requires generation room despite "one word" instruction)
- temperature: 0, seed: 42

### Models tried and rejected
| Model | Best variant | Agreement | Why rejected |
|---|---|---|---|
| qwen2.5:7b-instruct | chat_fewshot | 56.2% | 92% None bias |
| cogito:8b-v1-preview | gen_fewshot | 56.2% | False-biased |
| Tohur/natsumura-storytelling-rp-llama-3.1 | chat_fewshot | 45.0% | no Unknown concept |
| qwen3:14b | — | — | CPU offload, ~1h/5 cases |
| tobestyledintro/qwen3.8-9b-distill | — | — | timeouts |
| mistral-small:22b | — | — | not pulled (404) |