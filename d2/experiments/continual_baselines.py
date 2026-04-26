"""
continual_baselines.py — EWC and GEM as comparison baselines for δ².

WHAT THIS FILE IS, FOR THE DUMMIES:
====================================

To prove δ² does anything new, we have to compare it against the existing
state-of-the-art continual-learning methods. The two we care about most are:

    EWC  — Elastic Weight Consolidation (Kirkpatrick et al., DeepMind, 2017)
    GEM  — Gradient Episodic Memory   (Lopez-Paz & Ranzato, FAIR, 2017)

Without these baselines, we can't claim δ² is better at the continual-learning
problem. We could only claim "δ² runs and produces some loss curve" — which
is much weaker than "δ² beats EWC and GEM on the standard benchmarks."

Both methods are about ten years old, well-understood, and not too hard to
implement from scratch. We do that here so we can run head-to-head with δ²
in the same training loop, on the same data, with the same model.


WHAT EACH METHOD DOES (recap):
===============================

EWC:
    "After training on Task A, figure out which weights mattered most.
     When training on Task B, add a penalty term that punishes changing
     those important weights."

    Important = high Fisher Information (the model is confident in that
    weight, so changing it would hurt the old task a lot).

    Concretely: after task A, freeze a snapshot W*_A and a Fisher matrix
    F_A. During task B, the loss becomes:

        L_total = L_taskB(W) + λ × Σ F_A[i] × (W[i] - W*_A[i])²

    The penalty pulls each weight back toward its post-task-A value, with
    a strength proportional to its Fisher-info importance.


GEM:
    "Keep a buffer of examples from past tasks. Before applying a gradient
     for the current task, check it doesn't increase the loss on any of
     those past examples. If it would, project the gradient onto the
     closest direction that doesn't."

    Concretely: maintain memory M_k for each past task k. Each step:
      1. Compute current-task gradient g
      2. For each past task k, compute g_k = gradient on M_k
      3. If g · g_k < 0 for some k (g would hurt task k), solve a small
         quadratic program to find the closest g̃ such that g̃ · g_j ≥ 0
         for all j
      4. Apply g̃ instead of g


WHY BOTH:
==========

EWC is fast and stateless-per-step (just an extra penalty in the loss).
GEM is heavier (memory buffer + per-step constraint check) but stronger
on hard task sequences. Together they bracket the continual-learning
literature: a regularization-based method (EWC) and a memory-based one
(GEM). δ² has elements of both — Fisher-weighted drift like EWC's δ₁,
gradient retention like GEM's buffer — so beating either in isolation is
a real signal.


SCOPE OF THIS FILE:
====================

We implement minimal, correct versions. Not the most performance-tuned
implementations. The goal is fair comparison, not state-of-the-art
performance for either baseline.

For EWC: we maintain Fisher per parameter, accumulated across all past
tasks. We add the penalty as part of the loss in the training loop.

For GEM: we maintain a small per-task memory buffer (default 256
examples per task), compute per-task gradients, and project via a tiny
QP (quadratic program) using torch's `torch.autograd.functional` for
gradient comparison.


HOW TO USE:
============

In `continual.py` (the benchmark runner), you'll see:

    if method == "ewc":
        baseline = EWCRegularizer(model, fisher_steps=200, lambda_reg=1000)
        ...
        for epoch in epochs:
            ...
            loss = task_loss(model, batch) + baseline.penalty(model)
            loss.backward()
            optimizer.step()
        baseline.consolidate(model, dataloader)  # at end of task

    if method == "gem":
        baseline = GEMConstraint(memory_size=256)
        ...
        for epoch in epochs:
            ...
            loss = task_loss(model, batch)
            loss.backward()
            baseline.project_gradients(model, past_tasks)
            optimizer.step()
        baseline.add_task_memory(task_id, dataloader)  # at end of task
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger("anamnesis.d2.continual_baselines")


# ============================================================================
# EWC — Elastic Weight Consolidation
# ============================================================================

class EWCRegularizer:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017).

    Maintains, per parameter:
      - W* (snapshot of weights at end of each past task, accumulated)
      - F  (Fisher Information diagonal, accumulated across tasks)

    Adds a quadratic penalty to the loss during training on a new task:
      penalty = λ × Σ F[i] × (W[i] - W*[i])²

    USAGE:
        ewc = EWCRegularizer(model, lambda_reg=1000)
        for task_idx, task_loader in enumerate(tasks):
            for batch in task_loader:
                loss = task_loss(model, batch)
                loss = loss + ewc.penalty(model)   # add EWC regularization
                loss.backward()
                optimizer.step()
            ewc.consolidate(model, task_loader)    # after task ends

    NOTE on `lambda_reg`: this is the most important hyperparameter.
    Too small (e.g., 1) and EWC has no effect — model forgets like vanilla SGD.
    Too large (e.g., 1e6) and the model can't learn the new task at all.
    Typical values: 100 to 10,000. We use 1000 by default.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lambda_reg: float = 1000.0,
        fisher_n_samples: int = 200,
    ):
        # The strength of the penalty term in the loss
        self.lambda_reg = lambda_reg

        # How many examples to use when estimating Fisher Information
        # (more = more accurate Fisher, but slower to consolidate)
        self.fisher_n_samples = fisher_n_samples

        # Per-parameter snapshots and Fisher info, both initially empty
        # (will fill in on first call to consolidate())
        self._w_star = {}    # name -> tensor (snapshot of weights at task end)
        self._fisher = {}    # name -> tensor (accumulated Fisher information)

        self._n_tasks_seen = 0

    @torch.no_grad()
    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Compute the EWC penalty term: λ × Σ F[i] × (W[i] - W*[i])²

        Returns a scalar tensor that should be ADDED to the task loss
        before calling .backward().

        Returns 0 if no tasks have been consolidated yet (no W* / F to
        regularize against — pre-first-task or first task in progress).
        """
        if self._n_tasks_seen == 0:
            # Before any task has finished, nothing to regularize against.
            # Return a scalar zero on the same device as the first parameter.
            for p in model.parameters():
                return torch.zeros((), device=p.device, dtype=p.dtype)
            return torch.zeros(())

        # Sum the squared-drift × fisher across every parameter
        loss = torch.zeros((), device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name not in self._w_star:
                # New parameter (e.g., added between tasks) — no anchor for it
                continue
            # (W - W*)² weighted by Fisher importance
            drift = param - self._w_star[name]
            loss = loss + (self._fisher[name] * drift.pow(2)).sum()

        return self.lambda_reg * loss

    def consolidate(
        self,
        model: torch.nn.Module,
        dataloader,
        device: str = "cuda",
    ):
        """
        After training on a task is done, compute Fisher info for that task
        and update W* + F. Call this ONCE at the end of each task.

        We compute Fisher by:
          1. Sampling N batches from the task dataloader
          2. For each batch, computing the squared gradient of the loss
          3. Averaging across batches → diagonal Fisher estimate

        Then we add this task's Fisher to the accumulated F (online EWC),
        and snapshot the current weights as W*.
        """
        model.eval()  # disable dropout etc., but we still want gradients

        # ── Initialize Fisher accumulator for this consolidation pass ────
        new_fisher = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_fisher[name] = torch.zeros_like(param)

        # ── Estimate Fisher by sampling gradients on task data ───────────
        n_done = 0
        for batch in dataloader:
            if n_done >= self.fisher_n_samples:
                break

            # Move batch to device (handles tuple batches like (x, y))
            if isinstance(batch, (tuple, list)):
                batch = [b.to(device) if torch.is_tensor(b) else b for b in batch]
            else:
                batch = batch.to(device)

            # Forward + loss
            model.zero_grad()
            x, y = batch[0], batch[1] if len(batch) >= 2 else None
            logits = model(x)
            if y is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            else:
                # Auto-regressive: assume model returns (logits, loss)
                loss = logits[1] if isinstance(logits, tuple) else logits.mean()

            loss.backward()

            # Accumulate squared gradients (Fisher diagonal estimate)
            for name, param in model.named_parameters():
                if param.grad is not None:
                    new_fisher[name] += param.grad.detach().pow(2)

            n_done += 1

        if n_done == 0:
            logger.warning("EWC consolidate: no batches processed, skipping")
            return

        # Normalize to make it an average per batch
        for name in new_fisher:
            new_fisher[name] /= n_done

        # ── Update accumulated Fisher and W* snapshot ────────────────────
        # Online EWC: F_total = F_old + F_new (sum across tasks)
        # W* = current weights (most recent task's endpoint)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self._fisher:
                self._fisher[name] = self._fisher[name] + new_fisher[name]
            else:
                self._fisher[name] = new_fisher[name]
            self._w_star[name] = param.detach().clone()

        self._n_tasks_seen += 1
        model.zero_grad()
        model.train()
        logger.info(
            f"EWC consolidated task {self._n_tasks_seen} "
            f"(used {n_done} batches for Fisher estimate)"
        )


# ============================================================================
# GEM — Gradient Episodic Memory
# ============================================================================

class GEMConstraint:
    """
    Gradient Episodic Memory (Lopez-Paz & Ranzato, 2017).

    Maintains a small memory buffer per past task. Before applying any
    gradient update for the current task, checks that the gradient
    doesn't increase the loss on any past task's memory — if it would,
    projects the gradient onto the closest direction that respects the
    no-increase constraint.

    USAGE:
        gem = GEMConstraint(memory_size=256)
        for task_idx, task_loader in enumerate(tasks):
            for batch in task_loader:
                loss = task_loss(model, batch)
                loss.backward()
                gem.project_gradients(model, loss_fn)   # project the gradients
                optimizer.step()
            gem.add_task_memory(task_idx, task_loader)  # archive this task

    NOTE on memory_size: 256 examples per task is the original paper's
    default and works well for MNIST-scale tasks. For larger tasks
    (CIFAR, language modeling), 1024 or more is typical.

    SIMPLIFIED VERSION:
    The full GEM paper uses a quadratic program (QP) to find the optimal
    projection. We use a simpler approximation: project g away from any
    past-task gradient g_k that has g · g_k < 0, one at a time. This is
    cheaper computationally and sufficient for MNIST-scale benchmarks.
    The full QP version can be added later if needed for fair comparison
    on harder tasks.
    """

    def __init__(self, memory_size: int = 256):
        # How many examples to retain per past task
        self.memory_size = memory_size

        # task_id -> list of (x, y) tensors held in memory
        self._task_memories = {}

    @torch.no_grad()
    def add_task_memory(self, task_id: int, dataloader, device: str = "cuda"):
        """
        After a task ends, sample `memory_size` examples and store them.
        These will be used at every future step to constrain gradient updates.
        """
        examples_x = []
        examples_y = []
        n_collected = 0

        for batch in dataloader:
            if n_collected >= self.memory_size:
                break
            if isinstance(batch, (tuple, list)):
                x, y = batch[0], batch[1]
            else:
                x = batch
                y = None
            x = x.to(device)
            if y is not None:
                y = y.to(device)

            # Random subsample within the batch if it's large
            n_take = min(x.size(0), self.memory_size - n_collected)
            examples_x.append(x[:n_take].detach().clone())
            if y is not None:
                examples_y.append(y[:n_take].detach().clone())
            n_collected += n_take

        if not examples_x:
            logger.warning(f"GEM: no examples collected for task {task_id}")
            return

        x_all = torch.cat(examples_x, dim=0)
        y_all = torch.cat(examples_y, dim=0) if examples_y else None
        self._task_memories[task_id] = (x_all, y_all)
        logger.info(
            f"GEM: stored {x_all.size(0)} memories for task {task_id} "
            f"(total tasks in memory: {len(self._task_memories)})"
        )

    def project_gradients(
        self,
        model: torch.nn.Module,
        loss_fn=None,
    ):
        """
        For each past task: check if the current gradient would hurt that
        task's memory. If so, project away from the conflicting direction.

        Call this AFTER loss.backward() and BEFORE optimizer.step().

        loss_fn: optional callable (model, x, y) -> scalar loss. If None,
                 we use cross-entropy on (model(x), y).
        """
        if not self._task_memories:
            # No past tasks → no constraint → leave gradients alone
            return

        # ── Capture current gradient as a flat vector ────────────────────
        current_grad = self._flatten_grads(model)
        if current_grad is None:
            return

        # ── For each past task, check + project if needed ────────────────
        for task_id, (x_mem, y_mem) in self._task_memories.items():
            past_grad = self._compute_grad_on_memory(
                model, x_mem, y_mem, loss_fn
            )
            if past_grad is None:
                continue

            # Inner product: do the gradients agree?
            #   > 0: yes (current update would also help past task) → fine
            #   ≈ 0: orthogonal → fine (no interference)
            #   < 0: NO (current update would HURT past task) → project away
            inner = torch.dot(current_grad, past_grad)

            if inner < 0:
                # Project current_grad onto the orthogonal complement of past_grad:
                #   g_proj = g - (g·g_past / g_past·g_past) × g_past
                # This removes the component that would harm the past task,
                # keeping only the component orthogonal to it.
                past_norm_sq = torch.dot(past_grad, past_grad)
                if past_norm_sq > 1e-12:  # avoid divide-by-zero
                    current_grad = current_grad - (inner / past_norm_sq) * past_grad

        # ── Write the projected gradient back to the parameters ─────────
        self._unflatten_grads(model, current_grad)

    @torch.no_grad()
    def _flatten_grads(self, model) -> Optional[torch.Tensor]:
        """
        Concatenate all parameter gradients into one flat tensor.
        Returns None if no parameter has a gradient yet.
        """
        grads = []
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                grads.append(p.grad.detach().flatten())
        if not grads:
            return None
        return torch.cat(grads)

    def _compute_grad_on_memory(
        self,
        model,
        x_mem,
        y_mem,
        loss_fn,
    ) -> Optional[torch.Tensor]:
        """
        Run forward + backward on stored memory examples and return the
        flat gradient. Restores model state cleanly afterward.
        """
        # Save current gradients (we'll restore them after computing past-task grad)
        saved_grads = [
            p.grad.detach().clone() if (p.requires_grad and p.grad is not None) else None
            for p in model.parameters()
        ]

        # Zero gradients to compute past-task gradient cleanly
        model.zero_grad()

        # Forward + loss on memory
        logits = model(x_mem)
        if loss_fn is not None:
            loss = loss_fn(model, x_mem, y_mem)
        elif y_mem is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_mem.view(-1),
            )
        else:
            loss = logits.mean()

        loss.backward()

        # Capture past-task gradient as flat vector
        past_grad = self._flatten_grads(model)

        # Restore current-task gradients
        for p, saved in zip(model.parameters(), saved_grads):
            if p.requires_grad:
                p.grad = saved

        return past_grad

    @torch.no_grad()
    def _unflatten_grads(self, model, flat_grad: torch.Tensor):
        """
        Reverse of _flatten_grads: split the flat tensor back into per-
        parameter gradient tensors and write them in place.
        """
        offset = 0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                n = p.numel()
                p.grad = flat_grad[offset:offset + n].view_as(p).clone()
                offset += n


# ============================================================================
# SAM — Sharpness-Aware Minimization
# ============================================================================

class SAMOptimizer(torch.optim.Optimizer):
    """
    Sharpness-Aware Minimization (Foret et al., 2020).

    SAM is the closest published cousin to δ² in spirit — both seek out
    contested directions in the loss surface rather than smoothing them
    away. SAM does it via per-step adversarial perturbation (find the
    worst nearby point, take the gradient there). δ² does it via
    persistent retention (signed-square the friction, accumulate in
    bassin, inject bounded tension).

    We include SAM as a CONTINUAL-LEARNING BASELINE alongside EWC and
    GEM. Flat minima generalize better, so SAM should retain past-task
    accuracy better than vanilla AdamW. Beating SAM on backward transfer
    (BWT) would mean δ²'s structured retention captures something flat
    minima alone don't.

    USAGE:
        # SAM wraps a base optimizer (typically SGD or AdamW)
        base_opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        sam = SAMOptimizer(model.parameters(), base_optimizer=base_opt, rho=0.05)

        for batch in loader:
            # Two forward+backward passes per step (SAM's cost)
            loss1 = task_loss(model, batch); loss1.backward()
            sam.first_step(zero_grad=True)        # perturb to worst point
            loss2 = task_loss(model, batch); loss2.backward()
            sam.second_step(zero_grad=True)       # step using perturbed grad

    NOTE on `rho`: this is the perturbation radius. 0.05 is the paper's
    default for vision tasks. Larger rho = more aggressive search for
    flat minima but harder to optimize. Smaller rho ≈ standard SGD.

    REFERENCES:
        Foret, P. et al. (2020) "Sharpness-Aware Minimization for
        Efficiently Improving Generalization." ICLR 2021.
    """

    def __init__(self, params, base_optimizer, rho: float = 0.05):
        if rho < 0:
            raise ValueError(f"rho must be >= 0, got {rho}")
        defaults = dict(rho=rho)
        super().__init__(params, defaults)

        # Store the inner optimizer (AdamW, SGD, etc.) — SAM delegates
        # the actual weight update to it; SAM only modifies WHERE the
        # gradient gets computed (at the perturbed point, not at W).
        self.base_optimizer = base_optimizer
        # Re-key our defaults onto the base optimizer's groups so they
        # share state cleanly.
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False):
        """
        Step 1 of SAM: move the weights to the worst nearby point.

        Compute the perturbation ε = ρ × ∇L(W) / ‖∇L(W)‖ and add it to
        each parameter. After this call, the parameters have temporarily
        moved to W + ε. The next backward pass will compute ∇L(W + ε).

        We save the perturbation in state so second_step can undo it.
        """
        # Compute total gradient norm across all parameters
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group.get("rho", 0.05) / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                # The perturbation: ε_i = (ρ / ‖∇L‖) × ∇L_i
                e_w = p.grad * scale
                p.add_(e_w)
                # Save so we can undo in step 2
                self.state[p]["e_w"] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False):
        """
        Step 2 of SAM: undo the perturbation, then run the base optimizer
        with the gradient computed at the perturbed point.

        At this point param.grad = ∇L(W + ε) from the second backward
        pass. We move parameters back to W (subtracting ε), then let the
        base optimizer (AdamW etc.) take its normal step using the
        perturbed-point gradient.
        """
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                # Undo the perturbation: W_perturbed - ε = W_original
                e_w = self.state[p].get("e_w")
                if e_w is not None:
                    p.sub_(e_w)
        # Now param.grad still holds ∇L(W + ε). The base optimizer uses it.
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        """Compute the total L2 norm of gradients across all parameters."""
        # Use the first parameter's device as the reference
        device = self.param_groups[0]["params"][0].device
        total = torch.zeros((), device=device)
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                total = total + p.grad.norm(p=2).pow(2)
        return total.sqrt()


# ============================================================================
# A simple "no-baseline" controller — just runs whatever optimizer you pass it
# ============================================================================

class NoBaselineWrapper:
    """
    A pass-through wrapper for runs that DON'T use any continual-learning
    baseline (raw Adam, raw δ², or controller).

    Has the same interface as EWCRegularizer / GEMConstraint so the
    benchmark runner can be uniform without conditional logic everywhere.

    Used when method ∈ {"adam", "delta2", "controller"}.
    """

    def penalty(self, model):
        # No regularization penalty
        for p in model.parameters():
            return torch.zeros((), device=p.device, dtype=p.dtype)
        return torch.zeros(())

    def consolidate(self, model, dataloader, device="cuda"):
        # Nothing to do
        pass

    def project_gradients(self, model, loss_fn=None):
        # Don't touch the gradients
        pass

    def add_task_memory(self, task_id, dataloader, device="cuda"):
        # No memory buffer to update
        pass
