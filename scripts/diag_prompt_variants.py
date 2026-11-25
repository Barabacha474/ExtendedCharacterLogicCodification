"""Diagnose prompt variants and (optionally) model choice for the teacher.

Compares agreement with GPT-4.1 on Check_Scenes across several prompt designs.
Reports which design has best agreement and a healthy label distribution.
"""
from __future__ import annotations
import argparse
import json
import hashlib
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import requests
from src.utils.paths import DATA_RAW, RESULTS

OLLAMA_URL = "http://localhost:11434"
CACHE_DIR = RESULTS / "teacher_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FEW_SHOT = [
    {
        "scene": "Haruhi grabs Kyon's tie and drags him toward the clubroom, ignoring his protests.",
        "q": "Is Haruhi forcing Kyon to do something he does not want to do?",
        "a": "True",
    },
    {
        "scene": "Kyon sits quietly in the clubroom reading a book while the others argue about the summer plans.",
        "q": "Is Haruhi forcing Kyon to do something he does not want to do?",
        "a": "False",
    },
    {
        "scene": "The SOS Brigade members discuss what to do next, but no decisions are reached.",
        "q": "Is Haruhi forcing Kyon to do something he does not want to do?",
        "a": "Unknown",
    },
]

SYSTEM_STRICT = (
    "You are a strict condition checker for role-playing scenes. "
    "Given a scene and a yes/no question about the scene, decide whether the condition holds. "
    "Rules:\n"
    "- Answer True if the scene provides clear evidence that the condition holds.\n"
    "- Answer False if the scene provides clear evidence that it does not hold.\n"
    "- Answer Unknown only if the scene contains no information relevant to the question.\n"
    "- Do NOT default to Unknown out of caution. If the scene gives any evidence, commit to True or False.\n"
    "Respond with exactly one word: True, False, or Unknown. No explanation, no punctuation."
)


def cache_get(key: str):
    p = CACHE_DIR / f"{key}.json"
    return json.loads(p.read_text())["response"] if p.exists() else None


def cache_put(key: str, payload: dict):
    (CACHE_DIR / f"{key}.json").write_text(json.dumps(payload))


def key_of(model: str, prompt: str, system: str | None) -> str:
    h = hashlib.sha1()
    for part in (model, prompt, system or ""):
        h.update(part.encode())
        h.update(b"||")
    return h.hexdigest()


def call_generate(model: str, prompt: str) -> str:
    k = key_of(model, prompt, None)
    v = cache_get(k)
    if v is not None:
        return v
    r = requests.post(f"{OLLAMA_URL}/api/generate",
                      json={"model": model, "prompt": prompt, "stream": False,
                            "options": {"temperature": 0, "seed": 42, "num_ctx": 8192}},
                      timeout=180)
    r.raise_for_status()
    out = r.json()["response"].strip()
    cache_put(k, {"response": out, "model": model, "prompt": prompt, "system": None})
    return out


def call_chat(model: str, system: str, user: str) -> str:
    k = key_of(model, user, system)
    v = cache_get(k)
    if v is not None:
        return v
    r = requests.post(f"{OLLAMA_URL}/api/chat",
                      json={"model": model, "stream": False,
                            "messages": [{"role": "system", "content": system},
                                         {"role": "user", "content": user}],
                            "options": {"temperature": 0, "seed": 42, "num_ctx": 8192}},
                      timeout=180)
    r.raise_for_status()
    out = r.json()["message"]["content"].strip()
    cache_put(k, {"response": out, "model": model, "prompt": user, "system": system})
    return out


def parse_label(raw: str) -> str:
    s = raw.strip().lower().strip(" .,\"'")
    if s in {"true", "yes"}:
        return "True"
    if s in {"false", "no"}:
        return "False"
    if s in {"unknown", "none", "unsure"}:
        return "None"
    for kw, lab in [("true", "True"), ("yes", "True"),
                    ("false", "False"), ("no", "False"),
                    ("unknown", "None"), ("none", "None")]:
        if kw in s:
            return lab
    return "PARSE_ERROR"


def build_prompt_original(scene: str, q: str) -> str:
    return f"Scene: {scene}\n\nQuestion: {q}\n\nDirectly answer only yes/no/unknown."


def build_prompt_strict(scene: str, q: str) -> str:
    return (f"{SYSTEM_STRICT}\n\n"
            f"Scene: {scene}\n\nQuestion: {q}\n\nAnswer:")


def build_prompt_fewshot(scene: str, q: str) -> str:
    shots = []
    for ex in FEW_SHOT:
        shots.append(f"Scene: {ex['scene']}\nQuestion: {ex['q']}\nAnswer: {ex['a']}")
    return (f"{SYSTEM_STRICT}\n\n"
            + "\n\n".join(shots)
            + f"\n\nScene: {scene}\nQuestion: {q}\nAnswer:")


VARIANTS = {
    "gen_original":      lambda m, s, q: call_generate(m, build_prompt_original(s, q)),
    "gen_strict":        lambda m, s, q: call_generate(m, build_prompt_strict(s, q)),
    "gen_fewshot":       lambda m, s, q: call_generate(m, build_prompt_fewshot(s, q)),
    "chat_strict":       lambda m, s, q: call_chat(m, SYSTEM_STRICT, f"Scene: {s}\n\nQuestion: {q}"),
    "chat_fewshot":      lambda m, s, q: call_chat(m, SYSTEM_STRICT + "\n\nExamples:\n" +
                                                    "\n".join(f"Scene: {e['scene']}\nQ: {e['q']}\nA: {e['a']}" for e in FEW_SHOT),
                                                    f"Scene: {s}\n\nQuestion: {q}"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Tohur/natsumura-storytelling-rp-llama-3.1")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--variants", default=",".join(VARIANTS.keys()))
    args = ap.parse_args()

    df = pd.read_parquet(DATA_RAW / "Check_Scenes" / "data" / "train-00000-of-00001.parquet")
    sample = df.sample(args.n, random_state=args.seed).reset_index(drop=True)
    print(f"model={args.model}  n={len(sample)}  GPT dist={df['answer'].value_counts().to_dict()}")

    wanted = args.variants.split(",")
    results = {}
    for name in wanted:
        fn = VARIANTS[name]
        preds = []
        for i, row in sample.iterrows():
            # extract scene/question from the canonical prompt
            p = row["prompt"]
            try:
                scene = p.split("Scene:", 1)[1].split("\n\nQuestion:", 1)[0].strip()
                q = p.split("Question:", 1)[1].split("\n\nDirectly", 1)[0].strip()
            except Exception:
                preds.append("PARSE_ERROR")
                continue
            try:
                raw = fn(args.model, scene, q)
            except Exception as e:
                print(f"[{name}] {i}: request error {e}")
                preds.append("PARSE_ERROR")
                continue
            preds.append(parse_label(raw))
            if i % 10 == 0:
                print(f"[{name}] {i}/{len(sample)} gpt={row['answer']} pred={preds[-1]}")
        results[name] = preds

    print("\n=== summary ===")
    gpt = sample["answer"].tolist()
    for name, preds in results.items():
        res = pd.DataFrame({"gpt": gpt, "pred": preds})
        agree = (res["gpt"] == res["pred"]).mean()
        errors = (res["pred"] == "PARSE_ERROR").sum()
        print(f"\n{name}: agreement={agree:.1%}  parse_errors={errors}/{len(res)}")
        print(f"  pred dist: {res['pred'].value_counts().to_dict()}")
        print(f"  gpt  dist: {res['gpt'].value_counts().to_dict()}")
        print(f"  crosstab:\n{pd.crosstab(res['gpt'], res['pred']).to_string()}")


if __name__ == "__main__":
    main()