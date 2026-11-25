"""Compare qwen teacher vs GPT-4.1 labels on original Check_Scenes."""
from __future__ import annotations
import sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from src.data.label_teacher import call_ollama, parse_label
from src.utils.paths import DATA_RAW

df = pd.read_parquet(DATA_RAW / "Check_Scenes" / "data" / "train-00000-of-00001.parquet")
sample = df.sample(100, random_state=42).reset_index(drop=True)

rows = []
for i, row in sample.iterrows():
    lab = parse_label(call_ollama(row["prompt"]))
    rows.append({"gpt": row["answer"], "qwen": lab})
    if i % 20 == 0:
        print(f"{i}/100: gpt={row['answer']} qwen={lab}")

res = pd.DataFrame(rows)
agree = (res["gpt"] == res["qwen"]).mean()
print(f"\nagreement qwen vs GPT-4.1: {agree:.1%}")
print("\nqwen distribution:", res["qwen"].value_counts().to_dict())
print("GPT distribution:", res["gpt"].value_counts().to_dict())
print("\ncross-tab (rows=GPT, cols=qwen):")
print(pd.crosstab(res["gpt"], res["qwen"]))