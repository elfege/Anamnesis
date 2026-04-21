"""
config.py — Configuration for Anamnesis-δ² (R2D2).

All hyperparameters and paths in one place.
Nothing clever here — just settings.
"""

import os
from dataclasses import dataclass, field


@dataclass
class TrainingConfig:
    """Everything that controls how training runs."""

    # ── Data ─────────────────────────────────────────────────────────────
    dataset: str = "wikitext"           # "wikitext", "openwebtext", or path to custom JSONL
    data_dir: str = "./d2/data"         # where prepared datasets live
    train_file: str = "train.bin"       # tokenized training data (binary)
    val_file: str = "val.bin"           # tokenized validation data (binary)

    # ── Model (imported from neural_network.TransformerConfig) ────────
    # These are duplicated here for convenience; they override TransformerConfig
    block_size: int = 256               # shorter than GPT-2's 1024 for faster iteration
    vocab_size: int = 50304             # GPT-2 tokenizer vocab
    n_layer: int = 6                    # fewer layers than GPT-2 (12) for faster training
    n_head: int = 6                     # fewer heads
    n_embd: int = 384                   # smaller embedding dim (GPT-2 uses 768)
    dropout: float = 0.1                # regularization during training
    bias: bool = False                  # no bias = slightly better + faster

    # ── Optimizer selection ──────────────────────────────────────────────
    # "adam"   = standard AdamW (control/baseline)
    # "delta2" = the δ² optimizer with bassin
    optimizer: str = "delta2"

    # ── Adam hyperparameters (used when optimizer="adam") ────────────────
    adam_lr: float = 6e-4               # learning rate
    adam_beta1: float = 0.9             # first moment decay
    adam_beta2: float = 0.95            # second moment decay
    adam_weight_decay: float = 0.1      # L2 regularization

    # ── δ² hyperparameters (used when optimizer="delta2") ────────────────
    d2_alpha1: float = 1e-5             # learning rate for logical friction δ₁
    d2_alpha2: float = 1e-4             # learning rate for empirical friction δ₂
    d2_gamma: float = 0.99              # bassin retention factor (0=no memory, 1=infinite memory)
    d2_eta: float = 1e-3                # bassin injection rate
    d2_bound_fn: str = "tanh"           # bounding function: "tanh", "clip", "sigmoid"
    d2_clip_value: float = 1.0          # clip range for bound_fn="clip"
    d2_w_bar_mode: str = "ema"          # reference state W̄: "init", "ema", "fisher"
    d2_w_bar_ema_decay: float = 0.999   # EMA decay for W̄ when mode="ema"
    d2_use_fisher: bool = False         # use Fisher Information weighting for δ₁

    # ── Training loop ────────────────────────────────────────────────────
    batch_size: int = 32                # sequences per batch
    grad_accum_steps: int = 4           # gradient accumulation (effective batch = 32 * 4 = 128)
    max_steps: int = 5000               # total training steps
    eval_interval: int = 250            # evaluate every N steps
    eval_steps: int = 50                # how many batches to average for eval loss
    log_interval: int = 10              # print metrics every N steps
    save_interval: int = 1000           # checkpoint every N steps

    # ── Learning rate schedule ───────────────────────────────────────────
    warmup_steps: int = 100             # linear warmup from 0 to lr
    lr_decay: bool = True               # cosine decay after warmup
    min_lr: float = 6e-5                # minimum learning rate (10% of max)

    # ── Bassin storage ───────────────────────────────────────────────────
    bassin_to_mongo: bool = False       # store bassin snapshots in MongoDB
    mongo_uri: str = "mongodb://localhost:5438"
    mongo_db: str = "anamnesis"
    bassin_collection: str = "bassin_tensors"
    bassin_snapshot_interval: int = 500 # save bassin to MongoDB every N steps

    # ── Output ───────────────────────────────────────────────────────────
    output_dir: str = "./d2/output"     # checkpoints, logs, metrics
    experiment_name: str = "default"    # subdirectory name

    # ── Hardware ─────────────────────────────────────────────────────────
    device: str = "auto"                # "auto", "cuda", "cpu"
    compile_model: bool = False         # torch.compile (PyTorch 2.0+, ~2x speedup)
    dtype: str = "float16"              # "float32", "float16", "bfloat16"
