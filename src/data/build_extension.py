"""Build extension dataset for the Edward experiment.

Step 1 of 2: construct inputs C0-C4 (no labels).
Step 2 (label_teacher.py): annotate with Ollama teacher.

Inputs:
  data/raw/Fandom_Codified_Profiles  -> conditions (via parse_hierarchy)
  data/raw/Fandom_Benchmark          -> Edward scenes
  vendor/codes/Koishi.code.json      -> not used (Edward only)

Outputs:
  data/processed/extension_cases.parquet
    columns: case_id, block_id, cond_id, question, scene, depth,
             parents_json, char_description, input_c0..input_c4,
             scene_chars, label  (label=None here)
"""
from __future__ import annotations
import json
import re
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.data.parse_hierarchy import parse_profiles
from src.utils.paths import DATA_RAW, DATA_PROCESSED

TARGET_CHAR = "Edward"
SCENE_CHAR_LIMIT = 1150  # ~288 tokens

INSTRUCTION = "Directly answer only True/False/Unknown."

# Character description for Edward (stable properties only, no scene-specific facts).
# Source: Fandom_Codified_Profiles segments + FMA canon, verified manually.
CHAR_DESC = (
    "Edward Elric is a young State Alchemist known as the Fullmetal Alchemist. "
    "He is fiercely protective of his younger brother Alphonse, whose soul is bound to a suit of armor. "
    "His central goal is to restore both of their bodies using the Philosopher's Stone. "
    "He is short-tempered, proud, and strongly dislikes being called short or treated as a child. "
    "He is highly skilled in alchemy, especially transmutation without circles after seeing the Gate of Truth. "
    "He has automail limbs (right arm, left leg) and feels their weight and pain. "
    "He is loyal, determined, and willing to sacrifice himself for Alphonse."
)


def build_inputs(scene: str, question: str, parents: list[dict], cond_by_id: dict) -> dict:
    """Return 5 input strings C0-C4 with fixed slot budgets."""
    parent_questions = []
    for p in parents:
        c = cond_by_id.get(p["cond_id"])
        if c is None:
            continue
        marker = "" if p.get("required", True) else " (NOT)"
        parent_questions.append(f"- {c.question}{marker}")
    parents_text = "\n".join(parent_questions)

    def fmt(desc: str, par: str) -> str:
        blocks = []
        if desc:
            blocks.append(f"Character: {desc}")
        if par:
            blocks.append(f"Active conditions:\n{par}")
        blocks.append(f"Scene: {scene}")
        blocks.append(f"Question: {question}")
        blocks.append(INSTRUCTION)
        return "\n\n".join(blocks)

    return {
        "input_c0": fmt("", ""),
        "input_c1": fmt(CHAR_DESC, ""),
        "input_c2": fmt("", parents_text),
        "input_c3": fmt("", parents_text),  # placeholder; C3 handled separately
        "input_c4": fmt(CHAR_DESC, parents_text),
    }


def main():
    prof = pd.read_parquet(DATA_RAW / "Fandom_Codified_Profiles" / "data" / "train-00000-of-00001.parquet")
    prof = prof[prof["character"] == TARGET_CHAR].reset_index(drop=True)
    conds = parse_profiles(prof)
    cond_by_id = {c.cond_id: c for c in conds}
    print(f"conditions: {len(conds)}")

    bench = pd.read_parquet(DATA_RAW / "Fandom_Benchmark" / "data" / "train-00000-of-00001.parquet")
    scenes = (
        bench[bench["character"] == TARGET_CHAR]["scene"]
        .dropna()
        .drop_duplicates()
        .reset_index(drop=True)
    )
    print(f"unique scenes: {len(scenes)}")

    short_scenes = scenes[scenes.str.len() <= SCENE_CHAR_LIMIT].tolist()
    print(f"short scenes (<= {SCENE_CHAR_LIMIT} chars): {len(short_scenes)}")

    rows = []
    cid = 0
    for si, scene in enumerate(short_scenes):
        for cond in conds:
            inputs = build_inputs(scene, cond.question, cond.parents, cond_by_id)
            rows.append({
                "case_id": f"s{si}_c{cond.cond_id}",
                "block_id": cond.block_id,
                "cond_id": cond.cond_id,
                "question": cond.question,
                "scene": scene,
                "depth": cond.depth,
                "parents_json": json.dumps(cond.parents),
                "has_parents": bool(cond.parents),
                "char_description": CHAR_DESC,
                "scene_chars": len(scene),
                "label": None,
                **inputs,
            })
            cid += 1
    df = pd.DataFrame(rows)
    print(f"cases: {len(df)}")
    print(f"cases with parents: {df['has_parents'].sum()}")

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_parquet(DATA_PROCESSED / "extension_cases.parquet", index=False)
    print("saved data/processed/extension_cases.parquet")


if __name__ == "__main__":
    main()