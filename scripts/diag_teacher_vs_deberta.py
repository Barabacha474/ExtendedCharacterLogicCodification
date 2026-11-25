"""Compare qwen vs released DeBERTa on the pilot cases (input_c0)."""
from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.utils.paths import DATA_PROCESSED, MODELS

pilot = pd.read_parquet(DATA_PROCESSED / "pilot_v2.parquet")
model = AutoModelForSequenceClassification.from_pretrained(MODELS / "deberta-v3-base-check-scene")
tok = AutoTokenizer.from_pretrained(MODELS / "deberta-v3-base-check-scene")
model.eval()
id2lab = {0: "False", 1: "None", 2: "True"}

rows = []
for _, row in pilot.iterrows():
    inputs = tok(row["input_c0"], return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        pred = model(**inputs).logits.argmax(-1).item()
    rows.append({"deberta": id2lab[pred], "qwen": row["label_input_c0"]})

res = pd.DataFrame(rows)
print("DeBERTa distribution:", res["deberta"].value_counts().to_dict())
print("qwen distribution:", res["qwen"].value_counts().to_dict())
print("agreement:", f"{(res['deberta'] == res['qwen']).mean():.1%}")
print("\ncross-tab (rows=DeBERTa, cols=qwen):")
print(pd.crosstab(res["deberta"], res["qwen"]))