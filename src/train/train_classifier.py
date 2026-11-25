"""Train DeBERTa-v3-base classifier for condition checking (C0-C4 variants).

Implements:
- Scene truncation: head(200) + tail(88) + "..." for scenes > 288 tokens
- Scene-grouped 90/10 split (seed=42)
- Class-weighted CrossEntropyLoss (dtype-matched to model)
- 3 seeds (42/43/44) with full logging
- Saves logits + predictions for downstream ECE and bootstrap CI
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)
import time
from torch.amp import GradScaler, autocast

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.paths import DATA_PROCESSED, DATA_SPLITS, MODELS, RESULTS_PREDICTIONS, RESULTS_METRICS

# Fixed hyperparameters (documented for reproducibility)
MODEL_NAME = "microsoft/deberta-v3-base"
MAX_LENGTH = 512
MAX_SCENE_TOKENS = 288
SCENE_HEAD_TOKENS = 200
SCENE_TAIL_TOKENS = 88
LABEL_MAP = {"False": 0, "None": 1, "True": 2}
NUM_LABELS = 3
DEFAULT_SEEDS = [42, 43, 44]
DEFAULT_EPOCHS = 8
DEFAULT_LR = 3e-5
DEFAULT_BATCH_SIZE = 16
DEFAULT_WEIGHT_DECAY = 0.01
DEFAULT_WARMUP_RATIO = 0.1


class ConditionCheckDataset(Dataset):
    """Dataset for condition checking with pre-truncated scenes."""

    def __init__(self, df: pd.DataFrame, tokenizer, input_col: str, label_col: str = "label_id"):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.input_col = input_col
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_text = row[self.input_col]
        label = row[self.label_col]

        # Tokenize with truncation
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }


def truncate_scene_in_input(input_str: str, tokenizer, max_scene_tokens: int = MAX_SCENE_TOKENS) -> str:
    """Parse input_cX string, truncate scene if needed, rebuild."""
    blocks = input_str.split("\n\n")
    new_blocks = []

    for block in blocks:
        if block.startswith("Scene: "):
            scene_text = block[len("Scene: "):]
            # Tokenize scene
            tokens = tokenizer.encode(scene_text, add_special_tokens=False)
            if len(tokens) > max_scene_tokens:
                # Apply head + tail truncation
                head = tokenizer.decode(tokens[:SCENE_HEAD_TOKENS], skip_special_tokens=True)
                tail = tokenizer.decode(tokens[-SCENE_TAIL_TOKENS:], skip_special_tokens=True)
                truncated_scene = f"{head} ... {tail}"
                new_blocks.append(f"Scene: {truncated_scene}")
            else:
                new_blocks.append(block)
        else:
            new_blocks.append(block)

    return "\n\n".join(new_blocks)


def create_scene_grouped_split(df: pd.DataFrame, test_frac: float = 0.1, seed: int = 42) -> Dict[str, List[int]]:
    """Create scene-grouped 90/10 split."""
    df = df.copy()
    df["scene_idx"] = df["case_id"].str.extract(r"s(\d+)_").astype(int)

    unique_scenes = df["scene_idx"].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_scenes)

    n_test = int(len(unique_scenes) * test_frac)
    test_scenes = set(unique_scenes[:n_test])

    train_idx = df[~df["scene_idx"].isin(test_scenes)].index.tolist()
    test_idx = df[df["scene_idx"].isin(test_scenes)].index.tolist()

    return {"train": train_idx, "test": test_idx}


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse frequency class weights in float32."""
    class_counts = np.bincount(labels, minlength=NUM_LABELS)
    class_counts = np.maximum(class_counts, 1)
    weights = 1.0 / class_counts
    weights = weights / weights.sum() * NUM_LABELS  # Normalize
    return torch.tensor(weights, dtype=torch.float32)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_seed(
        train_df, test_df, tokenizer, seed, variant, input_col,
        epochs=DEFAULT_EPOCHS, lr=DEFAULT_LR, batch_size=DEFAULT_BATCH_SIZE,
        output_dir=None,
):
    print(f"\n{'=' * 60}")
    print(f"Seed {seed} | Variant {variant}")
    print(f"{'=' * 60}")

    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"  # AMP: включаем только на GPU
    print(f"Device: {device} | AMP: {use_amp}")

    # VRAM monitor
    if torch.cuda.is_available():
        print(f"VRAM before model: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    # Scene truncation
    print("Applying scene truncation...")
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["input_truncated"] = train_df[input_col].apply(
        lambda x: truncate_scene_in_input(x, tokenizer))
    test_df["input_truncated"] = test_df[input_col].apply(
        lambda x: truncate_scene_in_input(x, tokenizer))

    # AMP: num_workers + pin_memory для ускорения DataLoader
    train_dataset = ConditionCheckDataset(train_df, tokenizer, "input_truncated")
    test_dataset = ConditionCheckDataset(test_df, tokenizer, "input_truncated")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True)

    # Model (float32 master weights — AMP сам приведёт к fp16 на forward)
    print(f"Loading model {MODEL_NAME}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=NUM_LABELS, torch_dtype=torch.float32,
    ).to(device)

    if torch.cuda.is_available():
        print(f"VRAM after model: {torch.cuda.memory_allocated() / 1e9:.2f}GB")

    # Class weights (всегда float32)
    train_labels = train_df["label_id"].values
    class_weights = compute_class_weights(train_labels).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    # Optimizer
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)],
         "weight_decay": DEFAULT_WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

    total_steps = len(train_loader) * epochs
    warmup_steps = int(total_steps * DEFAULT_WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # AMP: GradScaler для стабильности fp16
    scaler = GradScaler("cuda", enabled=use_amp)

    print(f"Training for {epochs} epochs ({total_steps} steps)...")
    best_val_f1 = 0.0
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        epoch_start = time.time()  # AMP: замер скорости

        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # AMP: быстрее, чем zero_grad()

            # AMP: autocast для forward в fp16
            with autocast("cuda", enabled=use_amp):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            # AMP: scaler.scale + backward + step
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # AMP: защита от NaN
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - epoch_start

        # VRAM monitor в конце эпохи
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / 1e9
            print(f"  VRAM peak this epoch: {peak_vram:.2f}GB")

        # Evaluation
        model.eval()
        all_preds, all_labels, all_logits = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                labels = batch["labels"]

                with autocast("cuda", enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits.float().cpu()  # AMP: .float() перед переносом на CPU
                preds = logits.argmax(dim=-1)

                all_preds.extend(preds.numpy())
                all_labels.extend(labels.numpy())
                all_logits.extend(logits.numpy())

        all_preds, all_labels, all_logits = map(np.array, (all_preds, all_labels, all_logits))
        from sklearn.metrics import f1_score
        macro_f1 = f1_score(all_labels, all_preds, average="macro")

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"Val Macro-F1: {macro_f1:.4f} | Time: {epoch_time:.1f}s")

        if macro_f1 > best_val_f1:
            best_val_f1 = macro_f1
            best_model_state = model.state_dict().copy()

    # Save best model
    if output_dir and best_model_state is not None:
        model.load_state_dict(best_model_state)
        model_dir = output_dir / f"seed_{seed}"
        model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Saved best model to {model_dir}")

    # Save predictions and logits
    pred_dir = RESULTS_PREDICTIONS / variant
    pred_dir.mkdir(parents=True, exist_ok=True)

    pred_df = test_df[["case_id", "label_id"]].copy()
    pred_df["pred"] = all_preds
    pred_df["logits"] = all_logits.tolist()
    pred_df.to_parquet(pred_dir / f"predictions_seed_{seed}.parquet", index=False)
    print(f"Saved predictions to {pred_dir / f'predictions_seed_{seed}.parquet'}")

    # Save metrics
    metrics = {
        "seed": seed,
        "variant": variant,
        "macro_f1": float(macro_f1),
        "best_val_f1": float(best_val_f1),
    }
    metrics_dir = RESULTS_METRICS / variant
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"metrics_seed_{seed}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    return macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", type=str, required=True, choices=["c0", "c1", "c2", "c3", "c4"],
                    help="Input variant (c0/c1/c2/c3/c4)")
    ap.add_argument("--input", type=str, default=None,
                    help="Path to labeled parquet (default: data/processed/filtered_labeled_{variant}.parquet)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                    help=f"Random seeds (default: {DEFAULT_SEEDS})")
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = ap.parse_args()

    # Determine input file
    if args.input:
        input_file = Path(args.input)
    else:
        input_file = DATA_PROCESSED / f"filtered_labeled_{args.variant}.parquet"

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    print(f"Loading data from {input_file}...")
    df = pd.read_parquet(input_file)
    print(f"Loaded {len(df)} cases")

    # Drop PARSE_ERROR
    n_parse_errors = (df["label_y"] == "PARSE_ERROR").sum()
    if n_parse_errors > 0:
        print(f"Dropping {n_parse_errors} PARSE_ERROR cases")
        df = df[df["label_y"] != "PARSE_ERROR"].reset_index(drop=True)

    # Label mapping
    df["label_id"] = df["label_y"].map(LABEL_MAP)
    if df["label_id"].isna().any():
        raise ValueError(f"Unknown labels found: {df[df['label_id'].isna()]['label_y'].unique()}")

    print(f"Label distribution: {df['label_y'].value_counts().to_dict()}")

    # Create or load split
    split_file = DATA_SPLITS / "extension_scene_grouped_split.json"
    if split_file.exists():
        print(f"Loading existing split from {split_file}")
        with open(split_file) as f:
            split = json.load(f)
    else:
        print("Creating scene-grouped 90/10 split...")
        split = create_scene_grouped_split(df, test_frac=0.1, seed=42)
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w") as f:
            json.dump(split, f, indent=2)
        print(f"Saved split to {split_file}")

    train_df = df.iloc[split["train"]].reset_index(drop=True)
    test_df = df.iloc[split["test"]].reset_index(drop=True)
    print(f"Train: {len(train_df)} | Test: {len(test_df)}")

    # Load tokenizer
    print(f"Loading tokenizer {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Input column
    input_col = f"input_{args.variant}"
    if input_col not in df.columns:
        raise ValueError(f"Column {input_col} not found in dataframe")

    # Train for each seed
    output_dir = MODELS / f"deberta-v3-base-{args.variant}"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for seed in args.seeds:
        macro_f1 = train_one_seed(
            train_df=train_df,
            test_df=test_df,
            tokenizer=tokenizer,
            seed=seed,
            variant=args.variant,
            input_col=input_col,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            output_dir=output_dir,
        )
        results.append({"seed": seed, "macro_f1": macro_f1})

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary for variant {args.variant}")
    print(f"{'='*60}")
    for r in results:
        print(f"Seed {r['seed']}: Macro-F1 = {r['macro_f1']:.4f}")

    mean_f1 = np.mean([r["macro_f1"] for r in results])
    std_f1 = np.std([r["macro_f1"] for r in results])
    print(f"Mean ± Std: {mean_f1:.4f} ± {std_f1:.4f}")

    # Save summary
    summary = {
        "variant": args.variant,
        "seeds": args.seeds,
        "results": results,
        "mean_macro_f1": float(mean_f1),
        "std_macro_f1": float(std_f1),
    }
    summary_file = RESULTS_METRICS / args.variant / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_file}")


if __name__ == "__main__":
    main()