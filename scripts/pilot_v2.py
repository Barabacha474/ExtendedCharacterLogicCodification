"""Pilot v2: does the teacher change its answer when context is added?"""
from __future__ import annotations
import argparse
import sys
from collections import Counter
from pathlib import Path

# --- make project root importable when run as `python scripts\pilot_v2.py`
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.data.label_teacher import call_ollama, parse_label
from src.utils.paths import DATA_PROCESSED

VARIANTS = ["input_c0", "input_c1", "input_c2", "input_c4"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(DATA_PROCESSED / "extension_cases.parquet")

    with_p = df[df["has_parents"]].sample(
        n=min(args.n // 3, int(df["has_parents"].sum())), random_state=args.seed
    )
    without_p = df[~df["has_parents"]].sample(
        n=args.n - len(with_p), random_state=args.seed
    )
    sample = pd.concat([with_p, without_p]).sample(frac=1, random_state=args.seed).reset_index(drop=True)
    print(f"sample: {len(sample)}  (with_parents={len(with_p)}, without={len(without_p)})")
    print(f"blocks covered: {sample['block_id'].nunique()} / {df['block_id'].nunique()}")
    print(f"scene_chars: min={sample['scene_chars'].min()} "
          f"median={int(sample['scene_chars'].median())} max={sample['scene_chars'].max()}")

    results = {v: [] for v in VARIANTS}
    for i, row in sample.iterrows():
        for v in VARIANTS:
            results[v].append(parse_label(call_ollama(row[v])))
        if i % 10 == 0:
            print(f"{i}/{len(sample)}: {row['case_id']} -> "
                  f"{ {v: results[v][-1] for v in VARIANTS} }")

    for v in VARIANTS:
        sample[f"label_{v}"] = results[v]

    print("\n=== label distribution per variant ===")
    for v in VARIANTS:
        print(v, dict(Counter(sample[f"label_{v}"])))

    print("\n=== parse errors per variant ===")
    for v in VARIANTS:
        n_err = (sample[f"label_{v}"] == "PARSE_ERROR").sum()
        print(v, f"{n_err}/{len(sample)}")

    print("\n=== agreement with C0 ===")
    for v in VARIANTS[1:]:
        agree = (sample[f"label_{v}"] == sample["label_input_c0"]).sum()
        print(f"{v} vs C0: {agree}/{len(sample)} = {100*agree/len(sample):.1f}%")

    print("\n=== disagreement C0 vs C4 (the signal we need) ===")
    dis = sample[sample["label_input_c4"] != sample["label_input_c0"]]
    print(f"{len(dis)}/{len(sample)} cases changed answer")
    if len(dis):
        cols = ["case_id", "has_parents", "label_input_c0", "label_input_c1",
                "label_input_c2", "label_input_c4", "question"]
        print(dis[cols].head(15).to_string())

    print("\n=== disagreement on has_parents subset (C0 vs C2) ===")
    sub = sample[sample["has_parents"]]
    if len(sub):
        d2 = (sub["label_input_c2"] != sub["label_input_c0"]).sum()
        print(f"{d2}/{len(sub)} cases changed answer")

    sample.to_parquet(DATA_PROCESSED / "pilot_v2.parquet", index=False)
    print(f"\nsaved {DATA_PROCESSED / 'pilot_v2.parquet'}")


if __name__ == "__main__":
    main()