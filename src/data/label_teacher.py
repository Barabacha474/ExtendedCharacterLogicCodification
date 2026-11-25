"""Label extension cases with Ollama teacher (gemma4:e4b, chat endpoint)."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import re
import pandas as pd
import requests

from src.utils.paths import DATA_PROCESSED, RESULTS

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"
MODEL = "gemma4:e4b"
OPTIONS = {"temperature": 0, "seed": 43, "num_ctx": 4096, "num_predict": 1024}
CACHE_DIR = RESULTS / "teacher_cache"

SYSTEM = (
    "You are a strict condition checker for role-playing scenes. "
    "Given a scene and a yes/no question about the scene, decide whether the condition holds. "
    "Rules:\n"
    "- Answer True if the scene provides clear evidence that the condition holds.\n"
    "- Answer False if the scene provides clear evidence that it does not hold.\n"
    "- Answer Unknown only if the scene contains no information relevant to the question.\n"
    "- Do NOT default to Unknown out of caution. If the scene gives any evidence, commit to True or False.\n"
    "Respond with exactly one word: True, False, or Unknown. No explanation, no punctuation.\n\n"
    "Examples:\n"
    "Scene: Haruhi grabs Kyon's tie and drags him toward the clubroom, ignoring his protests.\n"
    "Question: Is Haruhi forcing Kyon to do something he does not want to do?\n"
    "Answer: True\n\n"
    "Scene: Kyon sits quietly in the clubroom reading a book while the others argue about the summer plans.\n"
    "Question: Is Haruhi forcing Kyon to do something he does not want to do?\n"
    "Answer: False\n\n"
    "Scene: The SOS Brigade members discuss what to do next, but no decisions are reached.\n"
    "Question: Is Haruhi forcing Kyon to do something he does not want to do?\n"
    "Answer: Unknown"
)


def cache_key(prompt: str) -> str:
    h = hashlib.sha1()
    for part in (MODEL, SYSTEM, prompt):
        h.update(part.encode("utf-8"))
        h.update(b"||")
    h.update(json.dumps(OPTIONS, sort_keys=True).encode())
    return h.hexdigest()


def call_ollama(prompt: str, retries: int = 3) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = cache_key(prompt)
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())["response"]

    body = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "options": OPTIONS,
    }
    last_err = None
    for attempt in range(retries):
        try:
            r = requests.post(OLLAMA_URL, json=body, timeout=180)
            r.raise_for_status()
            resp = r.json()["message"]["content"].strip()
            cache_file.write_text(json.dumps({"prompt": prompt, "system": SYSTEM, "model": MODEL, "response": resp}))
            return resp
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Ollama failed after {retries}: {last_err}")


_LABEL_PATTERNS = [
    (re.compile(r"\btrue\b"),  "True"),
    (re.compile(r"\byes\b"),   "True"),
    (re.compile(r"\bfalse\b"), "False"),
    (re.compile(r"\bno\b"),    "False"),
    (re.compile(r"\bunknown\b"), "None"),
    (re.compile(r"\bnone\b"),    "None"),
]

def parse_label(raw: str) -> str:
    """Last-occurrence label parser (regex, word-boundary).

    Reasoning models emit explanations; the final decision is the LAST
    label keyword in the output. Word boundaries prevent matching 'no' in
    'not', 'none' in 'nonetheless', 'true' in 'untrue', etc.
    """
    s = raw.lower()
    hits = []
    for pat, lab in _LABEL_PATTERNS:
        for m in pat.finditer(s):
            hits.append((m.start(), lab))
    if not hits:
        return "PARSE_ERROR"
    hits.sort(key=lambda x: x[0])
    return hits[-1][1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--variant", default="input_c0")
    ap.add_argument("--out", default=str(DATA_PROCESSED / "extension_labeled.parquet"))
    args = ap.parse_args()

    df = pd.read_parquet(DATA_PROCESSED / "extension_cases_filtered.parquet")

    if args.pilot:
        sub = df[df["block_id"] == df["block_id"].iloc[0]]
        sample = sub.sort_values("scene_chars").head(3)
        print(f"PILOT: {len(sample)} cases")
    else:
        sample = df

    labels = []
    for i, row in sample.iterrows():
        raw = call_ollama(row[args.variant])
        lab = parse_label(raw)
        labels.append({"case_id": row["case_id"], "raw": raw, "label": lab})
        if i % 50 == 0:
            print(f"{i}/{len(sample)}: {row['case_id']} -> {lab} (raw={raw[:40]!r})")

    lab_df = pd.DataFrame(labels)
    print("label distribution:", lab_df["label"].value_counts().to_dict())
    n_err = (lab_df["label"] == "PARSE_ERROR").sum()
    print(f"parse errors: {n_err}/{len(lab_df)} ({100*n_err/len(lab_df):.1f}%)")

    if args.pilot:
        lab_df.to_parquet(DATA_PROCESSED / "pilot_labels.parquet", index=False)
        print("pilot labels saved")
    else:
        merged = df.merge(lab_df[["case_id", "label"]], on="case_id", how="left")
        merged.to_parquet(args.out, index=False)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()