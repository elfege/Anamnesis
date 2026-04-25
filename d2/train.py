"""
train.py — Training loop for Anamnesis-δ² (R2D2).

This file orchestrates the training process:
1. Load data
2. Build model
3. Create optimizer (Adam OR δ²)
4. Run the training loop
5. Log metrics, save checkpoints

It's designed to run TWO experiments with the same code:
    python d2/train.py --optimizer adam      # baseline (control)
    python d2/train.py --optimizer delta2    # the δ² experiment

Same model, same data, same hyperparameters. Only the optimizer changes.
That's what makes it a valid experiment.


THE TRAINING LOOP (for dummies):
=================================

Every training step does this:

    1. Grab a batch of text          (e.g., 32 sequences of 256 tokens)
    2. Feed it through the model     (forward pass → get predictions)
    3. Compute how wrong we are      (loss = cross-entropy between predictions and reality)
    4. Compute the gradient           (backward pass → which direction reduces the error)
    5. Update the weights             (optimizer.step() → Adam subtracts, δ² sign-squares and injects)
    6. Log metrics                    (loss, bassin stats, learning rate, etc.)
    7. Occasionally evaluate          (run on validation data to check real performance)
    8. Occasionally save              (checkpoint so we can resume if interrupted)

Steps 1-3 are identical for Adam and δ².
Step 4 is identical — both compute the same gradient.
Step 5 is WHERE THE DIFFERENCE IS — Adam subtracts, δ² accumulates and injects.
Steps 6-8 are identical, but δ² also logs bassin statistics.
"""

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path

import torch

from neural_network import Transformer, TransformerConfig
from optimizer import DeltaSquaredOptimizer
from controller import DialecticalController, build_controller_from_model
from bassin import BassinStore, classify_negation, NegationType
from config import TrainingConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("anamnesis.d2.train")


# ============================================================================
# DATA LOADING
# ============================================================================

def get_batch(
    split: str,
    config: TrainingConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: torch.device,
):
    """
    Get a random batch of training or validation data.

    The data is stored as a flat 1D tensor of token IDs. We randomly
    sample starting positions and extract sequences of length block_size.

    Args:
        split:      "train" or "val"
        config:     training configuration
        train_data: 1D tensor of all training token IDs
        val_data:   1D tensor of all validation token IDs
        device:     where to put the batch (CPU or GPU)

    Returns:
        x: (batch_size, block_size) — input token IDs
        y: (batch_size, block_size) — target token IDs (shifted by 1)
           y[b][t] = the token that SHOULD come after x[b][t]
    """
    data = train_data if split == 'train' else val_data

    # Random starting positions for each sequence in the batch
    # We need room for block_size + 1 tokens (input + one target at the end)
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))

    # Extract sequences: x is the input, y is the target (shifted by 1)
    x = torch.stack([data[i:i + config.block_size] for i in ix])
    y = torch.stack([data[i + 1:i + 1 + config.block_size] for i in ix])

    # Move to GPU if available
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(
    model: Transformer,
    config: TrainingConfig,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    device: torch.device,
) -> dict:
    """
    Estimate the average loss on train and val sets.

    We run eval_steps batches and average the loss. This gives a more
    stable estimate than looking at a single batch.

    Returns:
        {"train": float, "val": float}
    """
    model.eval()  # switch to evaluation mode (disables dropout)
    out = {}
    for split in ['train', 'val']:
        losses = []
        for _ in range(config.eval_steps):
            x, y = get_batch(split, config, train_data, val_data, device)
            _, loss = model(x, y)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()  # switch back to training mode
    return out


# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================

def get_lr(step: int, config: TrainingConfig) -> float:
    """
    Cosine learning rate schedule with linear warmup.

    1. Warmup phase (steps 0 to warmup_steps):
       LR increases linearly from 0 to max_lr.
       WHY: Starting with a large LR on random weights causes instability.

    2. Cosine decay phase (after warmup):
       LR decreases following a cosine curve from max_lr to min_lr.
       WHY: As training progresses, we want smaller adjustments.

    This schedule is used for BOTH Adam and δ². For δ², it controls
    the bassin injection rate (eta) rather than the gradient step size.
    """
    max_lr = config.adam_lr if config.optimizer == 'adam' else config.d2_eta

    # Linear warmup
    if step < config.warmup_steps:
        return max_lr * (step + 1) / config.warmup_steps

    # After max_steps, return minimum
    if step > config.max_steps:
        return config.min_lr

    # Cosine decay between warmup_steps and max_steps
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # goes from 1 to 0
    return config.min_lr + coeff * (max_lr - config.min_lr)


# ============================================================================
# OPTIMIZER CREATION
# ============================================================================

def create_optimizer(model: Transformer, config: TrainingConfig):
    """
    Create either Adam (baseline) or δ² (experiment) optimizer.

    For Adam:
        Standard AdamW with weight decay. This is what every LLM uses.
        It's the control group — the thing we're comparing δ² against.

    For δ²:
        The DeltaSquaredOptimizer from optimizer.py. This is the novel part.

    Both optimizers receive the same parameters from the same model.
    """
    # Separate parameters into two groups:
    # - 2D+ tensors (weight matrices): apply weight decay
    # - 1D tensors (biases, layernorms): no weight decay
    # WHY: Weight decay regularizes the big matrices but shouldn't
    # affect biases/norms (they're small and serve a different purpose).
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    num_decay = sum(p.numel() for p in decay_params)
    num_nodecay = sum(p.numel() for p in nodecay_params)
    logger.info(f"Parameters: {num_decay:,} with decay, {num_nodecay:,} without")

    if config.optimizer == 'adam':
        # ── Standard Adam (control) ─────────────────────────────────
        optim_groups = [
            {'params': decay_params, 'weight_decay': config.adam_weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=config.adam_lr,
            betas=(config.adam_beta1, config.adam_beta2),
        )
        logger.info("Using optimizer: AdamW (baseline)")

    elif config.optimizer == 'delta2':
        # ── δ² optimizer (experiment) ────────────────────────────────
        # No weight decay groups needed — δ² handles things differently.
        # All parameters get the same treatment.
        optimizer = DeltaSquaredOptimizer(
            model.parameters(),
            alpha1=config.d2_alpha1,
            alpha2=config.d2_alpha2,
            gamma=config.d2_gamma,
            eta=config.d2_eta,
            bound_fn=config.d2_bound_fn,
            clip_value=config.d2_clip_value,
            w_bar_mode=config.d2_w_bar_mode,
            w_bar_ema_decay=config.d2_w_bar_ema_decay,
        )
        logger.info("Using optimizer: DeltaSquared (δ²)")

    elif config.optimizer == 'controller':
        # ── Dialectical Controller (Adam ↔ δ² switching) ─────────────
        # Holds BOTH optimizers and picks one each step based on a
        # confidence signal (loss / grad_norm / entropy). See controller.py
        # for the full explanation. This is the "speculative moment" of
        # the framework — neither pure Adam nor pure δ², but their
        # preserved opposition.
        optimizer = build_controller_from_model(model, config)
        logger.info(
            f"Using optimizer: DialecticalController "
            f"(signal={config.controller_signal}, warmup={config.controller_warmup_steps})"
        )

    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    return optimizer


# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

def train(config: TrainingConfig):
    """
    The main training function.

    This is the entry point. It:
    1. Sets up device, data, model, optimizer
    2. Runs the training loop
    3. Logs metrics and saves checkpoints
    """

    # ── Device setup ─────────────────────────────────────────────────
    if config.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config.device)
    logger.info(f"Device: {device}")

    # ── Data type ────────────────────────────────────────────────────
    # float16 = faster, uses less memory, slightly less precise
    # float32 = slower, full precision
    # bfloat16 = like float16 but with more range (better for training)
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    ptdtype = dtype_map.get(config.dtype, torch.float16)
    ctx = torch.amp.autocast(device_type=device.type, dtype=ptdtype) \
        if device.type == 'cuda' else torch.nullcontext()

    # ── Load data ────────────────────────────────────────────────────
    data_dir = Path(config.data_dir) / config.dataset
    train_path = data_dir / config.train_file
    val_path = data_dir / config.val_file

    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Run the data preparation script first:")
        logger.error(f"  python d2/data/prepare_{config.dataset}.py")
        return

    # Memory-mapped loading: the data stays on disk and is loaded on demand.
    # This lets us train on datasets much larger than RAM.
    train_data = torch.from_numpy(
        __import__('numpy').memmap(str(train_path), dtype=__import__('numpy').uint16, mode='r')
    ).long()
    val_data = torch.from_numpy(
        __import__('numpy').memmap(str(val_path), dtype=__import__('numpy').uint16, mode='r')
    ).long()
    logger.info(f"Data loaded: {len(train_data):,} train tokens, {len(val_data):,} val tokens")

    # ── Build model ──────────────────────────────────────────────────
    model_config = TransformerConfig(
        block_size=config.block_size,
        vocab_size=config.vocab_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
        dropout=config.dropout,
        bias=config.bias,
    )
    model = Transformer(model_config).to(device)

    # Optional: torch.compile for ~2x speedup (PyTorch 2.0+)
    if config.compile_model and hasattr(torch, 'compile'):
        logger.info("Compiling model with torch.compile...")
        model = torch.compile(model)

    # ── Create optimizer ─────────────────────────────────────────────
    optimizer = create_optimizer(model, config)

    # ── Bassin storage (MongoDB, optional) ───────────────────────────
    bassin_store = None
    if config.bassin_to_mongo and config.optimizer == 'delta2':
        bassin_store = BassinStore(
            config.mongo_uri, config.mongo_db, config.bassin_collection
        )
        logger.info("Bassin snapshots will be saved to MongoDB")

    # ── Output directory ─────────────────────────────────────────────
    out_dir = Path(config.output_dir) / config.experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config for reproducibility
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(config), f, indent=2)

    # Metrics log (append one JSON line per step)
    metrics_file = open(out_dir / "metrics.jsonl", "a")

    # ── Training loop ────────────────────────────────────────────────
    logger.info(f"Starting training: {config.max_steps} steps, optimizer={config.optimizer}")
    logger.info(f"Experiment: {config.experiment_name}")

    best_val_loss = float('inf')
    t0 = time.time()

    for step in range(config.max_steps):
        # ── Learning rate schedule ───────────────────────────────────
        lr = get_lr(step, config)
        for param_group in optimizer.param_groups:
            if config.optimizer == 'adam':
                param_group['lr'] = lr
            else:
                # For δ², the LR schedule controls the injection rate η
                param_group['eta'] = lr

        # ── Evaluation ───────────────────────────────────────────────
        if step % config.eval_interval == 0 or step == config.max_steps - 1:
            losses = estimate_loss(model, config, train_data, val_data, device)
            logger.info(
                f"step {step:5d} | train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

            # Save best model
            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'step': step,
                    'val_loss': best_val_loss,
                    'config': vars(config),
                }, out_dir / "best.pt")

        # ── Forward + backward (with gradient accumulation) ──────────
        # Gradient accumulation: instead of updating weights every batch,
        # we accumulate gradients over multiple batches and then update.
        # This simulates a larger batch size without needing more memory.
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(config.grad_accum_steps):
            x, y = get_batch('train', config, train_data, val_data, device)

            with ctx:
                logits, loss = model(x, y)
                # Scale loss by accumulation steps so the total gradient
                # magnitude is the same regardless of how many micro-steps.
                loss = loss / config.grad_accum_steps

            loss.backward()  # THIS computes δ₂ (the gradient) for every parameter
            accum_loss += loss.item()

        # ── Gradient clipping (safety net) ───────────────────────────
        # Even with δ², we clip extreme gradients to prevent NaN.
        # This is a belt-and-suspenders safety measure, not part of the
        # δ² theory. It should rarely activate if hyperparameters are right.
        # `clip_grad_norm_` returns the TOTAL gradient norm — we capture it
        # because the controller can use it as a confidence signal.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # ── Optimizer step ───────────────────────────────────────────
        # THIS IS WHERE THE DIFFERENCE HAPPENS.
        #   Adam:        subtracts normalized gradient from weights.
        #   δ²:          sign-squares the friction, accumulates in bassin, injects bounded tension.
        #   Controller:  picks Adam or δ² each step based on a confidence signal,
        #                so it needs the loss / grad_norm passed in to decide.
        if isinstance(optimizer, DialecticalController):
            # Pass the signals the controller might use.
            # It only reads the one its `signal` config asked for; the others
            # are ignored. Cheap enough to always pass both.
            optimizer.step(loss=accum_loss, grad_norm=float(grad_norm))
        else:
            # Standard optimizers (AdamW, DeltaSquaredOptimizer) take no args.
            optimizer.step()

        # ── Logging ──────────────────────────────────────────────────
        if step % config.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()

            metrics = {
                "step": step,
                "loss": accum_loss,
                "lr": lr,
                "time_ms": dt * 1000,
                "optimizer": config.optimizer,
            }

            # If using δ², also log bassin statistics
            if config.optimizer == 'delta2' and isinstance(optimizer, DeltaSquaredOptimizer):
                bassin_stats = optimizer.get_bassin_stats()
                metrics["bassin"] = bassin_stats

                # Log a compact summary
                logger.info(
                    f"step {step:5d} | loss {accum_loss:.4f} | lr {lr:.2e} | "
                    f"bassin mean={bassin_stats.get('abs_mean', 0):.6f} "
                    f"max={bassin_stats.get('max', 0):.4f}"
                )
            else:
                logger.info(
                    f"step {step:5d} | loss {accum_loss:.4f} | lr {lr:.2e} | "
                    f"{dt*1000:.0f}ms"
                )

            # Write metrics to JSONL file
            metrics_file.write(json.dumps(metrics, default=str) + "\n")
            metrics_file.flush()

        # ── Bassin snapshot to MongoDB ───────────────────────────────
        if (bassin_store is not None
                and step > 0
                and step % config.bassin_snapshot_interval == 0):
            bassin_stats = optimizer.get_bassin_stats()
            bassin_store.save_snapshot(
                step=step,
                experiment=config.experiment_name,
                bassin_stats=bassin_stats,
                negation_summary={},  # TODO: per-layer classification
            )

        # ── Checkpoint ───────────────────────────────────────────────
        if step > 0 and step % config.save_interval == 0:
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'step': step,
                'config': vars(config),
            }, out_dir / f"checkpoint_{step}.pt")
            logger.info(f"Saved checkpoint at step {step}")

    # ── Final save ───────────────────────────────────────────────────
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'step': config.max_steps,
        'config': vars(config),
    }, out_dir / "final.pt")

    metrics_file.close()
    logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    logger.info(f"Output: {out_dir}")


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anamnesis-δ² Training")

    # The most important flag: which optimizer to use
    parser.add_argument("--optimizer", choices=["adam", "delta2"], default="delta2",
                        help="Which optimizer: 'adam' (baseline) or 'delta2' (experiment)")
    parser.add_argument("--experiment", type=str, default=None,
                        help="Experiment name (default: auto-generated from optimizer)")

    # Override any config value from CLI
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-layer", type=int, default=None)
    parser.add_argument("--n-head", type=int, default=None)
    parser.add_argument("--n-embd", type=int, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--d2-gamma", type=float, default=None)
    parser.add_argument("--d2-eta", type=float, default=None)
    parser.add_argument("--d2-w-bar-mode", type=str, default=None)

    args = parser.parse_args()

    # Build config with CLI overrides
    config = TrainingConfig()
    config.optimizer = args.optimizer

    if args.experiment:
        config.experiment_name = args.experiment
    else:
        config.experiment_name = f"{config.optimizer}_{config.dataset}"

    # Apply CLI overrides
    for key in ['max_steps', 'batch_size', 'n_layer', 'n_head', 'n_embd',
                'block_size', 'dataset', 'device']:
        cli_val = getattr(args, key.replace('-', '_'), None)
        if cli_val is not None:
            setattr(config, key, cli_val)

    if args.d2_gamma is not None:
        config.d2_gamma = args.d2_gamma
    if args.d2_eta is not None:
        config.d2_eta = args.d2_eta
    if args.d2_w_bar_mode is not None:
        config.d2_w_bar_mode = args.d2_w_bar_mode

    train(config)
