"""Filter cross-product (scene x condition) by semantic relevance.

Keeps pairs where cosine(emb(scene), emb(question)) >= threshold.
Threshold chosen so that ~20-30% of pairs survive.

This is a PRE-REGISTERED filter, applied before seeing labels.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from src.utils.paths import DATA_PROCESSED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.35)
    ap.add_argument("--min-keep-frac", type=float, default=0.15)
    ap.add_argument("--max-keep-frac", type=float, default=0.40)
    args = ap.parse_args()

    df = pd.read_parquet(DATA_PROCESSED / "extension_cases.parquet")
    print(f"input pairs: {len(df)}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    unique_scenes = df.drop_duplicates("scene")[["scene"]].reset_index(drop=True)
    unique_qs = df.drop_duplicates("question")[["question"]].reset_index(drop=True)
    print(f"unique scenes: {len(unique_scenes)}  unique questions: {len(unique_qs)}")

    scene_emb = model.encode(unique_scenes["scene"].tolist(), show_progress_bar=True, batch_size=32)
    q_emb = model.encode(unique_qs["question"].tolist(), show_progress_bar=True, batch_size=32)

    scene_to_idx = {s: i for i, s in enumerate(unique_scenes["scene"])}
    q_to_idx = {q: i for i, q in enumerate(unique_qs["question"])}

    sims = np.array([
        float(np.dot(scene_emb[scene_to_idx[s]], q_emb[q_to_idx[q]]))
        for s, q in zip(df["scene"], df["question"])
    ])
    df = df.assign(relevance=sims)

    print(f"relevance: min={sims.min():.3f} median={np.median(sims):.3f} max={sims.max():.3f}")
    print(f"quantiles: p25={np.percentile(sims,25):.3f} p50={np.percentile(sims,50):.3f} "
          f"p75={np.percentile(sims,75):.3f} p90={np.percentile(sims,90):.3f}")

    # auto-threshold: choose so that kept fraction is in [min, max]
    for thr in np.linspace(0.10, 0.55, 46):
        kept = (sims >= thr).mean()
        if kept <= args.max_keep_frac:
            break

    thr = float(thr)
    kept_df = df[df["relevance"] >= thr].copy()
    print(f"\nchosen threshold: {thr:.3f}  kept: {len(kept_df)}/{len(df)} ({100*len(kept_df)/len(df):.1f}%)")
    print(f"kept with parents: {kept_df['has_parents'].sum()}")
    print(f"kept per block:\n{kept_df['block_id'].value_counts().sort_index().to_string()}")

    kept_df.to_parquet(DATA_PROCESSED / "extension_cases_filtered.parquet", index=False)
    print(f"\nsaved {DATA_PROCESSED / 'extension_cases_filtered.parquet'}")


if __name__ == "__main__":
    main()