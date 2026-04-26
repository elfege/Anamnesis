"""
optimizer.py — The δ² optimizer for Anamnesis-δ² (R2D2).

THIS IS THE NOVEL PART. Everything else (neural_network.py, train.py) is
standard ML machinery. This file contains the new idea.


WHAT THIS FILE IMPLEMENTS:
===========================

Standard gradient descent (what Adam does):
    W_next = W_now - α × gradient
    "Subtract the error. Converge. Stop changing."

The δ² update rule (what THIS optimizer does):
    W_next = W_now + η × f(B)
    where B accumulates signed-squared frictions over time.
    "Square the friction. Accumulate it. Inject it. Keep growing."


THE THREE KEY OPERATIONS:
==========================

1. SIGNED SQUARING  (Section 2.2 of the formal addendum)

   For each component of the friction vector:
       δ_sq[i] = sign(δ[i]) × δ[i]²

   This preserves DIRECTION (sign) while amplifying MAGNITUDE (square).
   A weight that should decrease keeps decreasing, but harder.
   A weight that should increase keeps increasing, but harder.

   This is the Aufhebung: the negation's CONTENT is preserved (direction)
   while its FORCE is amplified (squared magnitude).

   Example:
       δ = [-0.5, 0.3, -0.1, 0.8]
       signed_square(δ) = [-0.25, 0.09, -0.01, 0.64]
       Directions preserved. Big values amplified more.


2. TENSION RESERVOIR (Bassin de tenseurs potentiels, Section 3.2 Option C)

   B_next = γ × B_now + (1 - γ) × (α₁ × δ₁_sq + α₂ × δ₂_sq)

   B is an exponentially weighted moving average of past signed-squared
   frictions. It's like a MEMORY of all the tensions the system has
   encountered. γ controls how long memories last:
       γ = 0.99  →  long memory (tension builds up over ~100 steps)
       γ = 0.9   →  short memory (tension fades in ~10 steps)
       γ = 0.0   →  no memory (only current step matters)

   This is structurally similar to Adam's second moment (v_n), BUT:
   - Adam DIVIDES by √v to DAMPEN high-variance dimensions
   - δ² ADDS f(B) to INJECT accumulated tension into the weights
   Same math, opposite purpose. (See Section 3.3 of formal addendum.)


3. BOUNDED INJECTION

   W_next = W_now + η × f(B)

   f is a bounding function (tanh, sigmoid, clip) that prevents the
   accumulated tension from exploding the weights. Without f, the weights
   grow without limit (Jan Dlabal's convergence objection).

   tanh squashes any value to [-1, +1]:
       tanh(0.5) ≈ 0.46    (small tension → almost linear)
       tanh(5.0) ≈ 0.9999  (large tension → capped at 1)
       tanh(-3.0) ≈ -0.995  (negative tension → capped at -1)


THE TWO FRICTIONS:
===================

δ₁ — LOGICAL FRICTION (a priori, no data needed)
    "How far have the weights drifted from their reference state?"

    δ₁ = W_now - W̄

    W̄ can be:
    - The initial random weights (simplest, weakest)
    - An exponential moving average of past weights (better)
    - Fisher-weighted drift (strongest — weights the drift by how
      confident the model is about each parameter)

    δ₁ is nonzero whenever the system has a HISTORY. A freshly
    initialized model has δ₁ = 0 everywhere.


δ₂ — EMPIRICAL FRICTION (contingent, requires data)
    "How wrong is the model on THIS particular batch?"

    δ₂ = ∇L  (the gradient — exactly what standard ML computes)

    This is identical to what Adam uses. The novelty is not in computing
    δ₂ but in what we DO with it (sign-square and accumulate, rather
    than subtract).


COMPARISON TO ADAM:
====================

| Feature              | Adam (standard)                    | δ² (this file)                        |
|----------------------|------------------------------------|---------------------------------------|
| What it accumulates  | EMA of g and g²                    | EMA of sign(δ₁)δ₁² and sign(δ₂)δ₂²  |
| What it does with it | Divides by √v (dampens)            | Adds f(B) (injects)                   |
| Update direction     | Downhill (subtract)                | Along friction (signed-square + add)  |
| Goal                 | Minimize loss (converge)           | Grow from tension (elevate)           |
| When variance is high| Step gets SMALLER (stabilize)      | Injection gets LARGER (amplify)       |
| Convergence          | Yes — reaches local minimum        | No — bounded growth, no fixed point   |
"""

import math
from typing import Optional

import torch
from torch.optim import Optimizer


def signed_square(tensor):
    """
    The Aufhebung operation on a tensor.

    For each element: preserve direction, square magnitude.
        result[i] = sign(tensor[i]) × tensor[i]²

    This is NOT the same as tensor ** 2 (which loses sign) or
    tensor * abs(tensor) (which is the same thing, just written differently...
    actually wait, tensor * abs(tensor) IS signed squaring. Let's use that
    because it's branchless and faster on GPU).

    Args:
        tensor: any PyTorch tensor

    Returns:
        Same shape. Direction preserved, magnitude squared.

    Examples:
        signed_square(torch.tensor([-3.0, 2.0, -0.5]))
        → tensor([-9.0, 4.0, -0.25])
    """
    # element * |element| = sign(element) * element²
    # This works because:
    #   positive × positive = positive (correct: +3 × 3 = +9)
    #   negative × positive = negative (correct: -3 × 3 = -9)
    return tensor * tensor.abs()


class DeltaSquaredOptimizer(Optimizer):
    """
    The δ² optimizer — a dialectical alternative to gradient descent.

    Instead of subtracting gradients (Adam, SGD), this optimizer:
    1. Computes two friction terms (δ₁ logical, δ₂ empirical)
    2. Applies signed squaring to each (Aufhebung)
    3. Accumulates them in a tension reservoir (bassin)
    4. Injects bounded tension into the weights

    Usage:
        optimizer = DeltaSquaredOptimizer(model.parameters(), config)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()        # this computes δ₂ (the gradient)
            optimizer.step()       # this does the δ² update
            optimizer.zero_grad()

    The optimizer reads gradients from param.grad (set by loss.backward())
    and uses them as δ₂. δ₁ is computed internally from the weight drift.

    Args:
        params:        model parameters (from model.parameters())
        alpha1:        learning rate for logical friction δ₁
        alpha2:        learning rate for empirical friction δ₂
        gamma:         bassin retention factor (EMA decay for tension reservoir)
        eta:           bassin injection rate
        bound_fn:      bounding function name ("tanh", "clip", "sigmoid")
        clip_value:    clip range when bound_fn="clip"
        w_bar_mode:    how to compute W̄ ("init", "ema", "fisher")
        w_bar_ema_decay: EMA decay rate for W̄ when mode="ema"
    """

    def __init__(
        self,
        params,
        alpha1: float = 1e-5,
        alpha2: float = 1e-4,
        gamma: float = 0.99,
        eta: float = 1e-3,
        bound_fn: str = "tanh",
        clip_value: float = 1.0,
        w_bar_mode: str = "ema",
        w_bar_ema_decay: float = 0.999,
        # ── PATH (B) — additive mode ──────────────────────────────────────
        # When True, do NOT replace the gradient step. Instead:
        #     W_next = W_now − base_lr·grad   +   η·tanh(B)
        #            └── learning ──────────┘ └── δ² nudge ─┘
        # This keeps the learning signal Adam-like (so the model actually
        # fits each task) and adds the δ² tension on top as a structural
        # memory of past contradictions. Closer to the philosophical claim
        # ("contradictions are productive content alongside learning, not
        # a replacement for it"). When False, behaves like the original
        # standalone-replacement formulation (which empirically didn't
        # learn — see d2/output/sweep_2026-04-26/ ACC ≈ 0.099 for δ²).
        additive_mode: bool = False,
        base_lr: float = 1e-3,
    ):
        # Validate inputs
        if alpha1 < 0.0:
            raise ValueError(f"alpha1 must be >= 0, got {alpha1}")
        if alpha2 < 0.0:
            raise ValueError(f"alpha2 must be >= 0, got {alpha2}")
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"gamma must be in [0, 1), got {gamma}")
        if base_lr < 0.0:
            raise ValueError(f"base_lr must be >= 0, got {base_lr}")

        # Store hyperparameters in the format PyTorch expects.
        # "defaults" is a dict that gets attached to every parameter group.
        defaults = dict(
            alpha1=alpha1,
            alpha2=alpha2,
            gamma=gamma,
            eta=eta,
            bound_fn=bound_fn,
            clip_value=clip_value,
            w_bar_mode=w_bar_mode,
            w_bar_ema_decay=w_bar_ema_decay,
            additive_mode=additive_mode,
            base_lr=base_lr,
        )
        super().__init__(params, defaults)

    @torch.no_grad()  # don't track gradients for the optimizer's own math
    def step(self, closure=None):
        """
        Perform one δ² update step.

        This is called AFTER loss.backward() has populated param.grad
        for every parameter. The gradients in param.grad ARE δ₂.

        The update for each parameter tensor:
            1. δ₂ = param.grad                     (empirical friction — already computed)
            2. δ₁ = param - W̄                      (logical friction — drift from reference)
            3. δ₁_sq = signed_square(δ₁)           (Aufhebung of logical friction)
            4. δ₂_sq = signed_square(δ₂)           (Aufhebung of empirical friction)
            5. B = γ × B + (1-γ) × (α₁δ₁² + α₂δ₂²)  (update tension reservoir)
            6. param += η × f(B)                    (inject bounded tension)

        Returns:
            loss value if closure is provided, else None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Walk through every parameter group and every parameter
        for group in self.param_groups:
            alpha1 = group['alpha1']
            alpha2 = group['alpha2']
            gamma = group['gamma']
            eta = group['eta']
            bound_fn = group['bound_fn']
            clip_value = group['clip_value']
            w_bar_mode = group['w_bar_mode']
            w_bar_ema_decay = group['w_bar_ema_decay']

            for p in group['params']:
                # Skip parameters that don't need gradients
                # (e.g., frozen layers, or parameters not involved in this loss)
                if p.grad is None:
                    continue

                # ── Get or initialize state for this parameter ───────────
                # PyTorch optimizers maintain a "state" dict per parameter.
                # On the first call, we initialize the bassin and reference.
                state = self.state[p]

                if len(state) == 0:
                    # First time seeing this parameter — initialize everything
                    state['step'] = 0

                    # B: the tension reservoir (bassin de tenseurs potentiels)
                    # Starts at zero — no accumulated tension yet.
                    state['bassin'] = torch.zeros_like(p.data)

                    # W̄: the reference state for computing logical friction δ₁
                    if w_bar_mode == 'init':
                        # Option (a): W̄ = initial weights (frozen copy)
                        state['w_bar'] = p.data.clone()
                    elif w_bar_mode == 'ema':
                        # Option (b): W̄ = exponential moving average of weights
                        # Starts as a copy of current weights
                        state['w_bar'] = p.data.clone()
                    elif w_bar_mode == 'fisher':
                        # Option (c): Fisher-weighted drift
                        # W̄ is still EMA, but δ₁ gets weighted by Fisher info
                        state['w_bar'] = p.data.clone()
                        state['fisher_ema'] = torch.zeros_like(p.data)
                    else:
                        raise ValueError(f"Unknown w_bar_mode: {w_bar_mode}")

                state['step'] += 1
                bassin = state['bassin']
                w_bar = state['w_bar']

                # ── δ₂: empirical friction ───────────────────────────────
                # This is just the gradient, already computed by loss.backward()
                delta2 = p.grad

                # ── δ₁: logical friction ─────────────────────────────────
                # How far have the weights drifted from their reference state?
                delta1 = p.data - w_bar

                # If using Fisher weighting, scale δ₁ by the model's confidence
                if w_bar_mode == 'fisher':
                    fisher_ema = state['fisher_ema']
                    # Update Fisher EMA: running average of gradient² (a proxy
                    # for the diagonal Fisher Information Matrix)
                    fisher_ema.mul_(w_bar_ema_decay).addcmul_(
                        delta2, delta2, value=1.0 - w_bar_ema_decay
                    )
                    # Weight δ₁ by sqrt(Fisher): parameters the model is
                    # confident about (high Fisher) get amplified drift
                    delta1 = delta1 * (fisher_ema.sqrt() + 1e-8)

                # ── Signed squaring (Aufhebung) ──────────────────────────
                delta1_sq = signed_square(delta1)
                delta2_sq = signed_square(delta2)

                # ── Update the tension reservoir (bassin) ────────────────
                # B = γ × B + (1 - γ) × (α₁ × δ₁² + α₂ × δ₂²)
                #
                # This is an exponential moving average. Each step, the bassin
                # retains γ fraction of its old value and absorbs (1-γ) fraction
                # of the new friction. Over time, the bassin reflects the
                # HISTORY of frictions, not just the current one.
                new_friction = alpha1 * delta1_sq + alpha2 * delta2_sq
                bassin.mul_(gamma).add_(new_friction, alpha=1.0 - gamma)

                # ── Bound the bassin ─────────────────────────────────────
                # Without bounding, the accumulated tension would eventually
                # explode the weights (Dlabal's convergence objection).
                # f(B) squashes the bassin to a bounded range.
                if bound_fn == 'tanh':
                    # tanh: squashes to [-1, +1]. Smooth, differentiable.
                    # Small tensions pass through almost linearly.
                    # Large tensions get capped.
                    bounded = torch.tanh(bassin)
                elif bound_fn == 'clip':
                    # Hard clamp to [-clip_value, +clip_value].
                    # Less smooth than tanh but simpler.
                    bounded = torch.clamp(bassin, -clip_value, clip_value)
                elif bound_fn == 'sigmoid':
                    # sigmoid: squashes to [0, 1]. Shifts everything positive.
                    # Less natural for signed updates. Included for experiments.
                    bounded = torch.sigmoid(bassin) - 0.5  # center around 0
                else:
                    raise ValueError(f"Unknown bound_fn: {bound_fn}")

                # ── Apply the update ──────────────────────────────────────
                # Two modes:
                #
                #   STANDALONE (additive_mode=False, original δ² formulation):
                #     W_next = W_now + η · f(B)
                #     The δ² nudge IS the update. No gradient descent step
                #     happens. Empirically this fails to learn — the bounded
                #     injection carries insufficient signal to fit even
                #     task 1 (see d2/output/sweep_2026-04-26/, ACC ≈ 0.099).
                #
                #   ADDITIVE (additive_mode=True, path b):
                #     W_next = W_now − base_lr · grad   +   η · f(B)
                #     Standard gradient descent does the actual learning;
                #     δ² adds a tension nudge on top as structural memory
                #     of past contradictions. Stays true to the philosophy
                #     ("contradictions productive *alongside* learning, not
                #     a replacement for it") and avoids the bloat / no-learning
                #     trap of the standalone form.
                if group['additive_mode']:
                    base_lr = group['base_lr']
                    # Gradient descent step (Adam-style sign convention: subtract grad)
                    p.data.add_(delta2, alpha=-base_lr)
                    # δ² tension nudge on top
                    p.data.add_(bounded, alpha=eta)
                else:
                    # Original standalone replacement form (kept for comparison)
                    p.data.add_(bounded, alpha=eta)

                # ── Update W̄ (reference state for δ₁) ───────────────────
                # If using EMA mode, W̄ slowly tracks the current weights.
                # This means δ₁ measures RECENT drift, not total drift.
                if w_bar_mode in ('ema', 'fisher'):
                    w_bar.mul_(w_bar_ema_decay).add_(
                        p.data, alpha=1.0 - w_bar_ema_decay
                    )

        return loss

    def get_bassin_snapshot(self):
        """
        Return a snapshot of the tension reservoir for all parameters.

        This is used by bassin.py to store snapshots in MongoDB for
        later analysis and for inference-time bassin recall.

        Returns:
            dict mapping parameter name → bassin tensor (detached CPU copy)
        """
        snapshot = {}
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                state = self.state.get(p, {})
                if 'bassin' in state:
                    snapshot[f"param_{i}"] = state['bassin'].detach().cpu().clone()
        return snapshot

    def get_bassin_stats(self):
        """
        Return summary statistics of the tension reservoir.

        Useful for logging and dashboards — you don't want to store
        the full bassin every step, but you do want to track its behavior.

        Returns:
            dict with mean/max/min tension, nonzero fraction, etc.
        """
        all_tensions = []
        for group in self.param_groups:
            for p in group['params']:
                state = self.state.get(p, {})
                if 'bassin' in state:
                    all_tensions.append(state['bassin'].detach().float())

        if not all_tensions:
            return {"empty": True}

        # Concatenate all bassin tensors into one big vector
        flat = torch.cat([t.flatten() for t in all_tensions])

        return {
            "mean": flat.mean().item(),
            "std": flat.std().item(),
            "max": flat.max().item(),
            "min": flat.min().item(),
            "abs_mean": flat.abs().mean().item(),
            "nonzero_frac": (flat.abs() > 1e-8).float().mean().item(),
            "total_params": flat.numel(),
        }
