"""
continual.py — Continual-learning benchmark runner.

WHAT THIS FILE IS, FOR THE DUMMIES:
====================================

This is the ACTUAL EXPERIMENT. The whole δ² project lives or dies on
whether this script produces good numbers.

The setup:

    Take a dataset (MNIST, CIFAR, or text) and split it into N "tasks"
    that the model has to learn ONE AFTER THE OTHER. After each task,
    we measure the model's accuracy on ALL tasks seen so far.

The goal: maintain high accuracy on EVERY task as the sequence progresses.

The enemy: catastrophic forgetting. By default, when you train a network
on Task 2, it forgets Task 1. That's the well-known bug we're trying to
fix.

The competitors:

    Method            What it does                            Where it shines
    ──────────────────────────────────────────────────────────────────────────
    "adam"          | Vanilla baseline (will forget badly)   | Single task only
    "ewc"           | Elastic Weight Consolidation           | Smooth task seqs
    "gem"           | Gradient Episodic Memory               | Adversarial seqs
    "delta2"        | δ² alone                                | TBD (this is us)
    "controller"    | Adam ↔ δ² dialectical switching         | TBD (this is us)


THE TWO STANDARD BENCHMARKS:
=============================

1. PERMUTED-MNIST:
   Take MNIST (60k images of digits 0-9). Apply a different random pixel
   permutation to each task. Task 1 = original. Task 2 = pixels permuted
   one way. Task 3 = different permutation. Etc.
   The labels stay the same (still 0-9). Only the inputs change.
   This is "domain incremental learning" — same task, different distributions.

2. SPLIT-MNIST:
   Take MNIST and split it by class. Task 1 = digits 0,1. Task 2 = 2,3.
   Task 3 = 4,5. Task 4 = 6,7. Task 5 = 8,9.
   Each task is a 2-class classification problem on different classes.
   This is "class incremental learning" — new classes appear over time.

Both are standard in the continual-learning literature. EWC and GEM both
reported numbers on these. We need to do the same to be comparable.


THE METRICS:
=============

After all tasks have been trained, compute on every task's test set:

  - Average Accuracy (ACC):
        Mean of test accuracies across all tasks at the end of training.
        Higher = better. Range [0, 1].

  - Backward Transfer (BWT):
        How much each task's accuracy DROPPED between when we finished
        training that task and the end of all training.
        BWT = (1/(T-1)) × Σ (acc_at_end_of_training[i] - acc_when_finished_task_i)
        Negative = forgetting (BAD). Zero = no forgetting. Positive = retroactive
        improvement (good but rare).

  - Forward Transfer (FWT):
        How much pre-training on previous tasks helped task t before any
        training on task t. Indicates whether task knowledge generalizes.
        Higher = better.

We log all three. The headline number is BWT — that's where δ² is supposed
to win against vanilla Adam.


HOW TO RUN:
============

After preparing the data once:

    python d2/data/prepare_continual_mnist.py

Run an experiment:

    # Vanilla baseline (will forget catastrophically)
    python d2/experiments/continual.py --method adam --benchmark permuted_mnist --tasks 5 --epochs 1

    # EWC baseline
    python d2/experiments/continual.py --method ewc --benchmark permuted_mnist --tasks 5 --epochs 1

    # GEM baseline
    python d2/experiments/continual.py --method gem --benchmark permuted_mnist --tasks 5 --epochs 1

    # δ² alone
    python d2/experiments/continual.py --method delta2 --benchmark permuted_mnist --tasks 5 --epochs 1

    # δ² + Adam controller (the "speculative moment")
    python d2/experiments/continual.py --method controller --benchmark permuted_mnist --tasks 5 --epochs 1

Then compare (to be implemented in compare_continual.py):

    python d2/experiments/compare_continual.py \\
        d2/output/continual_adam.json \\
        d2/output/continual_ewc.json \\
        d2/output/continual_gem.json \\
        d2/output/continual_delta2.json \\
        d2/output/continual_controller.json


WHAT THIS FILE DOES STEP BY STEP:
==================================

1. Parse command-line args (method, benchmark, tasks, epochs)
2. Load the chosen benchmark dataset, split into N tasks
3. Build a small MLP model (continual-learning literature uses small MLPs
   for MNIST — fair comparison with EWC/GEM)
4. Set up the optimizer based on `method`:
       adam     → AdamW
       ewc      → AdamW + EWCRegularizer
       gem      → AdamW + GEMConstraint
       delta2   → DeltaSquaredOptimizer
       controller → DialecticalController
5. For each task in sequence:
       a. Train the model on that task for `epochs` epochs
       b. After the task, evaluate on ALL tasks seen so far
       c. Record the accuracies in a matrix (rows = tasks trained, cols = tasks evaluated)
       d. If using EWC: consolidate (compute Fisher, snapshot weights)
       e. If using GEM: archive memory examples
6. After all tasks, compute ACC / BWT / FWT and write to a JSON file.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Allow importing sibling modules from d2/
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Local imports (siblings inside d2/)
from optimizer import DeltaSquaredOptimizer
from controller import DialecticalController, build_controller_from_model
from experiments.continual_baselines import (
    EWCRegularizer,
    GEMConstraint,
    NoBaselineWrapper,
    SAMOptimizer,
)

logger = logging.getLogger("anamnesis.d2.continual")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")


# ============================================================================
# A SMALL MLP MODEL — what the continual-learning lit uses for MNIST
# ============================================================================

class SmallMLP(nn.Module):
    """
    A small fully-connected network — 2 hidden layers, ReLU, ~150k params.

    Why so small? Because the continual-learning literature standardized
    on small MLPs for MNIST experiments. Comparable numbers across papers
    require comparable models. Using a transformer here would be unfair.
    """
    def __init__(self, input_dim: int = 784, hidden_dim: int = 100, n_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x):
        # Flatten if needed (MNIST images come as [B, 1, 28, 28])
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.net(x)


# ============================================================================
# BENCHMARK DATA LOADERS (placeholders — wired through prepare_continual_mnist.py)
# ============================================================================

def get_permuted_mnist_tasks(n_tasks: int, batch_size: int, data_dir: str):
    """
    Build N tasks of permuted MNIST.

    Each task uses the same underlying MNIST dataset but with a different
    random pixel permutation applied to every image.

    Returns a list of (train_loader, test_loader) tuples, one per task.

    NOTE: This depends on `torchvision.datasets.MNIST`. If torchvision is
    not installed, the prep script `prepare_continual_mnist.py` should be
    run first (it caches MNIST as plain .pt files we can load without
    torchvision).
    """
    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise ImportError(
            "torchvision is required for the MNIST benchmarks. "
            "Install with: pip install torchvision"
        ) from e

    transform = transforms.Compose([transforms.ToTensor()])

    # Load MNIST once — we'll create N permuted views of it
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    # Generate N permutations of the 784 input pixels
    tasks = []
    rng = torch.Generator().manual_seed(42)
    for task_idx in range(n_tasks):
        perm = torch.randperm(784, generator=rng)

        # Wrap dataset with the permutation
        # We use a tiny custom Dataset that applies the perm on the fly
        class PermutedDataset(torch.utils.data.Dataset):
            def __init__(self, base, perm):
                self.base = base
                self.perm = perm
            def __len__(self):
                return len(self.base)
            def __getitem__(self, idx):
                x, y = self.base[idx]
                x = x.view(-1)[self.perm].view(1, 28, 28)
                return x, y

        train_loader = DataLoader(
            PermutedDataset(train_ds, perm),
            batch_size=batch_size, shuffle=True, num_workers=0,
        )
        test_loader = DataLoader(
            PermutedDataset(test_ds, perm),
            batch_size=batch_size, shuffle=False, num_workers=0,
        )
        tasks.append((train_loader, test_loader))

    logger.info(f"Built {n_tasks} permuted-MNIST tasks (60k train / 10k test each)")
    return tasks


def get_split_mnist_tasks(n_tasks: int, batch_size: int, data_dir: str):
    """
    Build N tasks of split MNIST.

    Splits MNIST classes into N groups (default 5 tasks of 2 classes each).
    Each task is a 2-class classification problem on a different pair.

    NOTE: We keep the full 10-class output head; the task simply contains
    examples from only 2 classes. This is the "class incremental" setup.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError as e:
        raise ImportError(
            "torchvision required. pip install torchvision"
        ) from e

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_ds = datasets.MNIST(data_dir, train=False, download=True, transform=transform)

    classes_per_task = 10 // n_tasks
    tasks = []
    for task_idx in range(n_tasks):
        lo = task_idx * classes_per_task
        hi = lo + classes_per_task
        # Filter dataset to just the classes for this task
        train_idx = [i for i, (_, y) in enumerate(train_ds) if lo <= y < hi]
        test_idx = [i for i, (_, y) in enumerate(test_ds) if lo <= y < hi]

        train_subset = torch.utils.data.Subset(train_ds, train_idx)
        test_subset = torch.utils.data.Subset(test_ds, test_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        tasks.append((train_loader, test_loader))

    logger.info(f"Built {n_tasks} split-MNIST tasks ({classes_per_task} classes each)")
    return tasks


# ============================================================================
# OPTIMIZER FACTORY (mirrors train.py but in continual-learning context)
# ============================================================================

def build_optimizer(method: str, model: nn.Module, lr: float):
    """
    Build the right optimizer for the chosen method.

    method ∈ {"adam", "ewc", "gem", "delta2", "controller"}

    Returns: (optimizer, baseline) — the baseline is a wrapper holding
    EWC / GEM state if applicable, or a no-op NoBaselineWrapper.
    """
    if method == "adam":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        baseline = NoBaselineWrapper()

    elif method == "ewc":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        baseline = EWCRegularizer(model, lambda_reg=1000.0, fisher_n_samples=200)

    elif method == "gem":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
        baseline = GEMConstraint(memory_size=256)

    elif method == "sam":
        # SAM wraps a base optimizer (AdamW). The baseline wrapper is a
        # no-op because SAM lives in the optimizer object itself, not in
        # a separate per-step regularizer/constraint.
        base_opt = torch.optim.AdamW(model.parameters(), lr=lr)
        opt = SAMOptimizer(model.parameters(), base_optimizer=base_opt, rho=0.05)
        baseline = NoBaselineWrapper()

    elif method == "delta2":
        # Path A — standalone replacement form (the original — empirically
        # fails to learn at default hyperparameters, ACC ≈ 0.099 on permuted
        # MNIST per d2/output/sweep_2026-04-26/).
        opt = DeltaSquaredOptimizer(
            model.parameters(),
            alpha1=1e-5, alpha2=1e-4,
            gamma=0.99, eta=1e-3,
            bound_fn="tanh",
            w_bar_mode="ema",
            additive_mode=False,
        )
        baseline = NoBaselineWrapper()

    elif method == "delta2_additive":
        # Path B — additive form: gradient descent does the actual learning,
        # δ² adds a bounded tension nudge on top from accumulated past
        # contradictions. Closer to the philosophical claim and avoids the
        # bloat / no-learning trap of the standalone form. base_lr matches
        # the lr we'd use for plain Adam on the same task (1e-3).
        opt = DeltaSquaredOptimizer(
            model.parameters(),
            alpha1=1e-5, alpha2=1e-4,
            gamma=0.99, eta=1e-3,
            bound_fn="tanh",
            w_bar_mode="ema",
            additive_mode=True,
            base_lr=lr,
        )
        baseline = NoBaselineWrapper()

    elif method == "controller":
        # Build a tiny config-shaped object the controller factory expects
        class _ControllerConfig:
            adam_lr = lr
            adam_beta1 = 0.9
            adam_beta2 = 0.999
            adam_weight_decay = 0.0
            d2_alpha1 = 1e-5
            d2_alpha2 = 1e-4
            d2_gamma = 0.99
            d2_eta = 1e-3
            d2_bound_fn = "tanh"
            d2_clip_value = 1.0
            d2_w_bar_mode = "ema"
            d2_w_bar_ema_decay = 0.999
            controller_signal = "loss"
            controller_warmup_steps = 50
            controller_loss_window = 30
            controller_loss_stable_threshold = 0.01
            controller_grad_norm_threshold = 1.0
            controller_entropy_threshold = 1.5
        opt = build_controller_from_model(model, _ControllerConfig)
        baseline = NoBaselineWrapper()

    else:
        raise ValueError(f"Unknown method: {method!r}")

    return opt, baseline


# ============================================================================
# TRAINING / EVALUATION
# ============================================================================

def train_one_task(
    model, train_loader, optimizer, baseline,
    method: str,
    epochs: int,
    device: str,
):
    """
    Train the model on one task for `epochs` epochs.
    Applies any baseline-specific behavior (EWC penalty, GEM projection,
    SAM two-pass forward/backward).
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)

            # ── SAM: two-pass training ──────────────────────────────
            # SAM needs forward+backward TWICE per step:
            #   1. compute ∇L(W); perturb to W+ε
            #   2. compute ∇L(W+ε); undo perturbation; base_optimizer.step()
            # Roughly 2× the wall-clock cost of standard training.
            if method == "sam":
                optimizer.zero_grad()
                loss = F.cross_entropy(model(x), y)
                loss.backward()
                optimizer.first_step(zero_grad=True)
                # Second forward+backward at the perturbed point
                F.cross_entropy(model(x), y).backward()
                optimizer.second_step(zero_grad=True)
                total_loss += loss.item()
                n_batches += 1
                continue

            optimizer.zero_grad()
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            # ── EWC: add penalty to loss ────────────────────────────
            if method == "ewc":
                penalty = baseline.penalty(model)
                loss = loss + penalty

            loss.backward()

            # ── GEM: project gradients before stepping ──────────────
            if method == "gem":
                baseline.project_gradients(model)

            # ── Optimizer step (controller takes loss arg) ──────────
            if isinstance(optimizer, DialecticalController):
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(loss=loss.item(), grad_norm=float(grad_norm))
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        logger.info(f"  Epoch {epoch+1}/{epochs}: avg loss = {avg_loss:.4f}")


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    """Return classification accuracy in [0, 1] on the given loader."""
    model.eval()
    correct = 0
    total = 0
    for batch in loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# ============================================================================
# METRIC COMPUTATION (ACC / BWT / FWT)
# ============================================================================

def compute_metrics(acc_matrix: list[list[float]]) -> dict:
    """
    Given a square accuracy matrix (rows = tasks trained, cols = tasks evaluated),
    compute the standard continual-learning metrics.

    acc_matrix[t][i] = accuracy on task i AFTER training on tasks 0..t

    Returns:
        {
            "acc":   final average accuracy across all tasks
            "bwt":   backward transfer (negative = forgetting)
            "fwt":   forward transfer (positive = generalization)
            "matrix": the full matrix for plotting
        }
    """
    T = len(acc_matrix)
    if T == 0:
        return {"acc": 0.0, "bwt": 0.0, "fwt": 0.0, "matrix": []}

    # ACC: mean of last row (accuracy on every task at the end of training)
    final_row = acc_matrix[-1]
    acc = sum(final_row) / T

    # BWT: average drop between "right after training task i" and "end of all training"
    # BWT_i = acc_matrix[T-1][i] - acc_matrix[i][i]
    # Average over i in [0, T-2] (exclude the last task which has no drop)
    if T >= 2:
        bwt_terms = [acc_matrix[T - 1][i] - acc_matrix[i][i] for i in range(T - 1)]
        bwt = sum(bwt_terms) / (T - 1)
    else:
        bwt = 0.0

    # FWT: average accuracy on task i BEFORE training on it, vs. random baseline
    # FWT_i = acc_matrix[i-1][i] - 0.1   (0.1 = random baseline for 10-class MNIST)
    # We approximate "random baseline" as 1/n_classes
    n_classes = 10
    random_baseline = 1.0 / n_classes
    if T >= 2:
        fwt_terms = [acc_matrix[i - 1][i] - random_baseline for i in range(1, T)]
        fwt = sum(fwt_terms) / (T - 1)
    else:
        fwt = 0.0

    return {"acc": acc, "bwt": bwt, "fwt": fwt, "matrix": acc_matrix}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    p = argparse.ArgumentParser(description="δ² continual-learning benchmark runner")
    p.add_argument("--method", choices=["adam", "ewc", "gem", "sam",
                                         "delta2", "delta2_additive", "controller"],
                   required=True, help="Which method to evaluate")
    p.add_argument("--benchmark", choices=["permuted_mnist", "split_mnist"],
                   default="permuted_mnist", help="Which benchmark to run")
    p.add_argument("--tasks", type=int, default=5, help="Number of sequential tasks")
    p.add_argument("--epochs", type=int, default=1, help="Epochs per task")
    p.add_argument("--batch-size", type=int, default=128, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--data-dir", default="./d2/data/mnist", help="MNIST cache dir")
    p.add_argument("--output", default="./d2/output/continual_result.json",
                   help="Where to write results JSON")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    logger.info(f"=== Continual benchmark: method={args.method} benchmark={args.benchmark} ===")
    logger.info(f"Tasks={args.tasks}, epochs/task={args.epochs}, lr={args.lr}, device={args.device}")

    # ── Build tasks ──────────────────────────────────────────────────────
    if args.benchmark == "permuted_mnist":
        tasks = get_permuted_mnist_tasks(args.tasks, args.batch_size, args.data_dir)
    else:
        tasks = get_split_mnist_tasks(args.tasks, args.batch_size, args.data_dir)

    # ── Build model + optimizer ──────────────────────────────────────────
    model = SmallMLP().to(args.device)
    optimizer, baseline = build_optimizer(args.method, model, args.lr)
    logger.info(f"Model: SmallMLP ({sum(p.numel() for p in model.parameters()):,} params)")
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    logger.info(f"Baseline: {type(baseline).__name__}")

    # ── Run the sequence ─────────────────────────────────────────────────
    # acc_matrix[t][i] = accuracy on task i after training tasks 0..t
    acc_matrix = []
    t_start = time.time()

    for t, (train_loader, _test_loader) in enumerate(tasks):
        logger.info(f"\n── Task {t+1}/{args.tasks} ──")
        train_one_task(model, train_loader, optimizer, baseline,
                       method=args.method, epochs=args.epochs, device=args.device)

        # ── EWC consolidation: compute Fisher and snapshot weights ──────
        if args.method == "ewc":
            baseline.consolidate(model, train_loader, device=args.device)

        # ── GEM: archive memory examples from this task ─────────────────
        if args.method == "gem":
            baseline.add_task_memory(t, train_loader, device=args.device)

        # ── Evaluate on ALL tasks seen so far ──────────────────────────
        row = []
        for i in range(args.tasks):
            test_loader = tasks[i][1]
            acc = evaluate(model, test_loader, args.device)
            row.append(acc)
            logger.info(f"  Test acc on task {i+1}: {acc:.4f}")
        acc_matrix.append(row)

    duration = time.time() - t_start

    # ── Compute final metrics ────────────────────────────────────────────
    metrics = compute_metrics(acc_matrix)
    metrics["method"] = args.method
    metrics["benchmark"] = args.benchmark
    metrics["tasks"] = args.tasks
    metrics["epochs_per_task"] = args.epochs
    metrics["seed"] = args.seed
    metrics["duration_seconds"] = duration

    # If controller was used, include its decision stats
    if isinstance(optimizer, DialecticalController):
        metrics["controller_stats"] = optimizer.get_stats()

    # ── Write to disk ────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Print summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info(f"  Method:   {args.method}")
    logger.info(f"  ACC:      {metrics['acc']:.4f}    (higher = better)")
    logger.info(f"  BWT:      {metrics['bwt']:+.4f}   (>= 0 = no forgetting)")
    logger.info(f"  FWT:      {metrics['fwt']:+.4f}   (positive = transfer)")
    logger.info(f"  Duration: {duration:.1f}s")
    logger.info(f"  Wrote:    {out_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
