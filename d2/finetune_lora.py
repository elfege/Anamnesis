"""
finetune_lora.py — Continual fine-tuning of an instruct LLM with δ² on LoRA adapters.

WHAT THIS DOES (for the dummies):
==================================
Takes a pretrained instruct model (e.g. Llama-3.2-3B-Instruct) and continually
fine-tunes it on YOUR Anamnesis episodes, in chronological order. The model
keeps its general language ability (because the base is frozen) AND learns
your style/recurring themes (because the LoRA adapters retain what's
fine-tuned in).

The δ² optimizer is applied to the LoRA adapter weights ONLY (the base model
stays frozen). The bassin (tension reservoir) accumulates structural memory
across the chronological tasks — that's the whole point: when you revisit
an old theme months later, the bassin should already encode the past tension.

PIPELINE:
=========
1. anamnesis_to_tokens.py prepares chronological .bin files (task_00.bin, task_01.bin, …)
2. This script:
   - Loads the HF base model + tokenizer
   - Wraps it in a LoRA adapter via peft
   - For each task in chronological order:
       - Trains for N steps using δ² (or AdamW for the control arm)
       - Evaluates on val.bin (held-out chronological tail)
       - Optionally evaluates on a public dataset (WikiText) to measure forgetting
   - Saves checkpoints at /workspace/checkpoints_personal/<experiment>/
   - Emits a metrics.jsonl with one line per step

USAGE:
======
    # δ² arm (the experiment)
    python -m d2.finetune_lora \\
        --base-model meta-llama/Llama-3.2-3B-Instruct \\
        --data-dir  /workspace/data_personal/anamnesis_chronological \\
        --output-dir /workspace/checkpoints_personal \\
        --experiment personal_delta2_v1 \\
        --optimizer delta2 \\
        --steps-per-task 200

    # AdamW arm (the control)
    python -m d2.finetune_lora \\
        ... (same flags) ... \\
        --experiment personal_adam_v1 \\
        --optimizer adam

PRIVACY:
========
Output goes to checkpoints_personal/ which is in .gitignore AND .dockerignore.
NEVER COPY this output into a docker image — the resulting weights encode
your personal patterns. See README_canonical_two_tracks.md.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

# Sibling-import helper (mirrors the pattern in d2/server.py and continual.py)
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("finetune_lora")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ────────────────────────────────────────────────────────────────────────────
# Lazy imports for heavyweight deps — fail with a clear message if missing
# ────────────────────────────────────────────────────────────────────────────
def _require_hf():
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model, TaskType
        return AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model, TaskType
    except ImportError as e:
        logger.error(
            "Missing HF dependencies. Install with:\n"
            "  pip install transformers peft accelerate bitsandbytes\n"
            f"Original error: {e}"
        )
        sys.exit(1)


# ────────────────────────────────────────────────────────────────────────────
# Data loading: read each task_NN.bin as a uint16 token stream
# ────────────────────────────────────────────────────────────────────────────
def load_task_streams(data_dir: Path) -> list[tuple[str, np.ndarray]]:
    """
    Reads manifest.json from data_dir, returns [(task_id, token_array), ...] in
    chronological order. Each token array is a flat uint16 sequence — same
    format as WikiText prep emits.
    """
    manifest_path = data_dir / "manifest.json"
    if not manifest_path.exists():
        logger.error(f"No manifest at {manifest_path}. Run scripts/anamnesis_to_tokens.py first.")
        sys.exit(2)
    manifest = json.loads(manifest_path.read_text())
    out = []
    for t in manifest["tasks"]:
        path = data_dir / t["filename"]
        if not path.exists():
            logger.warning(f"missing {path} — skipping")
            continue
        ids = np.fromfile(path, dtype=np.uint16)
        out.append((t["id"], ids))
        logger.info(f"loaded {t['id']}: {len(ids):,} tokens from {t['filename']}")
    return out


def load_val_stream(data_dir: Path) -> np.ndarray | None:
    p = data_dir / "val.bin"
    if not p.exists():
        return None
    return np.fromfile(p, dtype=np.uint16)


def get_batch(stream: np.ndarray, block_size: int, batch_size: int, device: torch.device):
    if len(stream) <= block_size + 1:
        raise ValueError(f"stream too short ({len(stream)}) for block_size={block_size}")
    idx = np.random.randint(0, len(stream) - block_size - 1, size=batch_size)
    x = np.stack([stream[i : i + block_size].astype(np.int64) for i in idx])
    y = np.stack([stream[i + 1 : i + 1 + block_size].astype(np.int64) for i in idx])
    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


# ────────────────────────────────────────────────────────────────────────────
# Optimizer: AdamW (control) or DeltaSquaredOptimizer (experiment, additive form)
# ────────────────────────────────────────────────────────────────────────────
def build_optimizer(model: torch.nn.Module, kind: str, lr: float, d2_eta: float):
    """
    Returns an optimizer that operates on the *trainable* parameters only
    (i.e. the LoRA adapters; the base model is frozen via requires_grad=False
    so it's automatically excluded from param_groups).
    """
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    logger.info(f"{n_train/1e6:.2f}M trainable parameters (LoRA adapters)")

    if kind == "adam":
        opt = torch.optim.AdamW(trainable, lr=lr)
        logger.info(f"optimizer: AdamW (control), lr={lr}")
    elif kind == "delta2":
        from optimizer import DeltaSquaredOptimizer
        opt = DeltaSquaredOptimizer(
            trainable,
            alpha1=1e-5,
            alpha2=1e-4,
            gamma=0.99,
            eta=d2_eta,
            bound_fn="tanh",
            w_bar_mode="ema",
            w_bar_ema_decay=0.999,
            additive_mode=True,   # path B (the only form that learns)
            base_lr=lr,
        )
        logger.info(f"optimizer: DeltaSquared (additive, path B), base_lr={lr}, eta={d2_eta}")
    else:
        raise ValueError(f"Unknown optimizer: {kind}")
    return opt


# ────────────────────────────────────────────────────────────────────────────
# Eval: cross-entropy loss on a held-out token stream
# ────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_stream(model, stream: np.ndarray, block_size: int, batch_size: int, n_batches: int, device: torch.device) -> float:
    if stream is None or len(stream) <= block_size + 1:
        return float("nan")
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(stream, block_size, batch_size, device)
        out = model(input_ids=x, labels=y)
        losses.append(out.loss.item())
    model.train()
    return float(np.mean(losses))


# ────────────────────────────────────────────────────────────────────────────
# Main loop
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct",
                    help="HF model id (default: Llama-3.2-3B-Instruct)")
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="Directory containing task_NN.bin + manifest.json (output of anamnesis_to_tokens.py)")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write checkpoints/metrics (must be under d2_checkpoints_personal/ for personal-corpus runs)")
    ap.add_argument("--experiment", required=True,
                    help="Subdir name under --output-dir (must start with 'personal_' for personal-corpus runs)")
    ap.add_argument("--optimizer", choices=["adam", "delta2"], default="delta2")
    ap.add_argument("--steps-per-task", type=int, default=200)
    ap.add_argument("--block-size", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d2-eta", type=float, default=1e-6)
    ap.add_argument("--lora-r", type=int, default=8)
    ap.add_argument("--lora-alpha", type=int, default=16)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--load-in-4bit", action="store_true", default=True,
                    help="QLoRA: load base in 4-bit to fit on smaller GPUs (default ON)")
    ap.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    ap.add_argument("--eval-interval", type=int, default=50)
    ap.add_argument("--eval-batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Safety: refuse to write personal-corpus outputs into a non-personal dir
    output_path = args.output_dir / args.experiment
    is_personal_run = args.experiment.startswith("personal_")
    is_personal_dir = "checkpoints_personal" in str(args.output_dir)
    if is_personal_run and not is_personal_dir:
        logger.error(
            f"Refusing to write a 'personal_*' experiment to a non-personal dir.\n"
            f"--experiment={args.experiment}\n--output-dir={args.output_dir}\n"
            f"Personal-corpus checkpoints MUST land under d2_checkpoints_personal/ "
            f"to prevent accidental publication. See README_canonical_two_tracks.md."
        )
        sys.exit(3)
    if is_personal_dir and not is_personal_run:
        logger.error(
            f"Personal checkpoint dir requires --experiment to start with 'personal_'.\n"
            f"Got: {args.experiment}"
        )
        sys.exit(3)

    output_path.mkdir(parents=True, exist_ok=True)
    metrics_file = open(output_path / "metrics.jsonl", "a")
    config_file = output_path / "config.json"
    config_file.write_text(json.dumps(vars(args), default=str, indent=2))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    # HF model + tokenizer
    AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_peft_model, TaskType = _require_hf()
    logger.info(f"loading base model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    quant_kwargs = {}
    if args.load_in_4bit and device.type == "cuda":
        try:
            from transformers import BitsAndBytesConfig
            quant_kwargs = dict(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            )
            logger.info("loading in 4-bit (QLoRA mode)")
        except ImportError:
            logger.warning("bitsandbytes not installed — falling back to fp16")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map="auto" if device.type == "cuda" else None,
        **quant_kwargs,
    )

    # LoRA wrap
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Optimizer (operates on requires_grad=True params only — i.e. LoRA adapters)
    optimizer = build_optimizer(model, args.optimizer, args.lr, args.d2_eta)

    # Load tasks
    tasks = load_task_streams(args.data_dir)
    val_stream = load_val_stream(args.data_dir)
    if not tasks:
        logger.error("No tasks loaded. Aborting.")
        sys.exit(2)

    # Training loop: one task at a time, in chronological order
    global_step = 0
    t0 = time.time()
    for task_idx, (task_id, stream) in enumerate(tasks):
        logger.info(f"━━━ TASK {task_idx} ({task_id}): {len(stream):,} tokens ━━━")
        for step in range(args.steps_per_task):
            x, y = get_batch(stream, args.block_size, args.batch_size, device)
            out = model(input_ids=x, labels=y)
            loss = out.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            if step % args.eval_interval == 0 or step == args.steps_per_task - 1:
                val_loss = eval_stream(model, val_stream, args.block_size, args.batch_size, args.eval_batches, device)
                # Optionally measure on previous tasks (BWT). Cheap to add but
                # potentially expensive on memory; for now just task + val.
                row = {
                    "global_step": global_step, "task_idx": task_idx, "task_id": task_id,
                    "step_in_task": step, "train_loss": float(loss.item()), "val_loss": val_loss,
                    "elapsed_s": time.time() - t0,
                }
                logger.info(f"  step {step:4d} | train {row['train_loss']:.4f} | val {val_loss:.4f}")
                metrics_file.write(json.dumps(row) + "\n")
                metrics_file.flush()

            global_step += 1

        # Save checkpoint after each task (can resume mid-curriculum)
        ckpt_path = output_path / f"after_task_{task_idx:02d}_{task_id}.pt"
        torch.save({
            "lora_state_dict": {k: v for k, v in model.state_dict().items() if "lora" in k.lower()},
            "task_idx": task_idx, "global_step": global_step, "config": vars(args),
        }, ckpt_path)
        logger.info(f"  saved {ckpt_path}")

    # Final save (full PEFT adapter, ready for reload)
    final_path = output_path / "lora_adapter_final"
    model.save_pretrained(str(final_path))
    logger.info(f"saved final PEFT adapter: {final_path}")
    metrics_file.close()
    logger.info("done.")


if __name__ == "__main__":
    main()
