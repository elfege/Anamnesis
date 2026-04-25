"""
controller.py — The Dialectical Controller (Adam ↔ δ² switching).

WHAT THIS FILE IS, FOR THE DUMMIES:
====================================

Most ML training uses ONE optimizer the whole time. SGD, or Adam, or AdamW.
Same rule every step. Same formula every batch. Boring but reliable.

This file does something different: it has TWO optimizers and a referee
that picks which one to use at each step, based on the state of the model.

  ┌──────────────────────────┐
  │  ┌───────────────────┐   │
  │  │  Look at model    │   │
  │  │  state right now  │   │
  │  └─────────┬─────────┘   │
  │            │             │
  │            v             │
  │  ┌─────────────────────┐ │
  │  │  Confident?         │ │
  │  │  (loss is stable    │ │
  │  │   or low entropy)   │ │
  │  └────┬─────────┬──────┘ │
  │       │         │         │
  │   yes │         │ no      │
  │       v         v         │
  │  ┌────────┐ ┌─────────┐  │
  │  │  Adam  │ │   δ²    │  │
  │  │ stabi- │ │ grow-   │  │
  │  │ lize   │ │ from-   │  │
  │  └────────┘ │ contra- │  │
  │             │ diction │  │
  │             └─────────┘  │
  └──────────────────────────┘

WHY HAVE TWO?
==============

Pure Adam: subtracts error, converges to a local minimum, becomes "dead"
           (no further change possible). Great for stabilizing what you
           already learned. Bad when you need to break out and explore.

Pure δ²:   adds tension, never converges, keeps growing. Great for
           exploring contested directions. Bad when you need to nail down
           a specific answer.

The controller's job: get the best of both. When the model has a clear
direction (low loss, low gradient noise, low output entropy), let Adam
stabilize. When the model is in doubt (high loss, high gradient noise,
high output entropy), let δ² inject tension and explore.

THE PHILOSOPHICAL POINT:
========================

In Hegel's terms (motivating, not load-bearing for the code):

    Adam  =  the moment of the UNDERSTANDING — fix the position
    δ²    =  the moment of REASON           — contradict it productively
    Controller = the SPECULATIVE moment      — preserve both, switch when needed

Pure Adam alone = entendement only (annihilate opposition, converge to dead state).
Pure δ² alone = Reason without ground (grow without terrain to grow from).
The complete framework: a controller that contains BOTH, picking the right one
for the moment. That IS the Aufhebung at the optimizer level.

But you don't need Hegel to read this code. The code just says:
    if model is confident: stabilize (Adam)
    else:                   grow from contradiction (δ²)

WHAT THE CONTROLLER DECIDES ON:
================================

The controller looks at "confidence signals" each step. We support three:

1. Loss-based:
       Is the loss stable (small change recently)? → Adam.
       Is the loss spiking or oscillating?         → δ².

   Reason: a stable loss means the model has settled. A spiking loss means
   it's encountering surprise — exactly when we want δ² to retain those
   surprises in the bassin.

2. Gradient-based:
       Is the gradient norm small? → Adam (we're near a minimum).
       Is the gradient norm large? → δ² (we're being pulled around).

3. Entropy-based (only available if the model exposes output logits):
       Is the output distribution peaked (low entropy)? → Adam.
       Is it flat (high entropy = uncertain)?           → δ².

By default we use the loss-based signal — it's always available and easy
to reason about. The other two are wired in for future experiments.

WHAT MAKES THIS A CONTRIBUTION (HONEST):
=========================================

A controller that learned its decisions from data would scale with compute
(Sutton-friendly). What we ship here is a HAND-DESIGNED controller — a
threshold-based if/else, not a learned one. That's a starting point, not
the final claim. The claim ladder:

    Step 1 (this file):       hand-coded if/else controller
    Step 2 (future work):     learned controller (small MLP picking optimizer)
    Step 3 (further future):  the controller's parameters trained jointly
                              with the model — meta-learning

Step 1 is what we test against EWC and GEM in the continual-learning
benchmarks. Steps 2 and 3 are the "compute-scalable controller" answer to
Sutton's Bitter Lesson.
"""

import logging
from collections import deque
from typing import Optional

import torch

from optimizer import DeltaSquaredOptimizer

logger = logging.getLogger("anamnesis.d2.controller")


# ============================================================================
# WHICH OPTIMIZER WAS PICKED — used for logging and metrics
# ============================================================================

class StepDecision:
    """
    A simple data-bag describing the controller's choice for one step.

    Why a class instead of a tuple/dict? Because it shows up in logs and
    metrics; having named fields prevents typos like `decision["adma"]`.
    """
    def __init__(self, used_optimizer: str, signal_value: float, reason: str):
        # used_optimizer: either "adam" or "delta2" — what actually ran this step
        # signal_value:   the number the controller looked at to decide
        # reason:         human-readable explanation ("loss stable", "gradient spike", etc.)
        self.used_optimizer = used_optimizer
        self.signal_value = signal_value
        self.reason = reason

    def to_dict(self):
        return {
            "used_optimizer": self.used_optimizer,
            "signal_value": self.signal_value,
            "reason": self.reason,
        }


# ============================================================================
# THE DIALECTICAL CONTROLLER
# ============================================================================

class DialecticalController:
    """
    Manages two optimizers and decides which one runs each step.

    Holds:
      - `adam`:   a standard torch.optim.AdamW for stabilization moments
      - `delta2`: a DeltaSquaredOptimizer for growth-from-contradiction moments
      - `signal`: which kind of confidence signal we look at
                  ("loss" | "grad_norm" | "entropy")
      - thresholds for the if/else decision

    USAGE:
        controller = DialecticalController(
            adam=adam_optim,
            delta2=delta2_optim,
            signal="loss",
            loss_window=50,
            loss_stable_threshold=0.01,
        )

        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            decision = controller.step(loss=loss.item())
            controller.zero_grad()
            # decision.used_optimizer tells you what ran ("adam" or "delta2")

    WHY HOLD BOTH AT ONCE?
    =======================

    Each optimizer carries its own internal state (Adam keeps moving averages
    of m and v; δ² keeps the bassin reservoir B and the reference W̄). If we
    instantiated them on demand, we'd lose that state every time we switched.
    So we keep BOTH alive at all times, and only one of them updates the
    weights on any given step. The other one's state stays frozen until it's
    its turn again.

    The trade-off: roughly 2× the optimizer state (which is small — a few MB
    for our model size). The benefit: clean switching with full state retention.
    """

    def __init__(
        self,
        adam: torch.optim.Optimizer,
        delta2: DeltaSquaredOptimizer,
        signal: str = "loss",
        loss_window: int = 50,                 # how many recent losses to average over
        loss_stable_threshold: float = 0.01,   # if recent loss change < this, "stable"
        grad_norm_stable_threshold: float = 1.0,
        entropy_stable_threshold: float = 1.5,
        warmup_steps: int = 100,               # always use Adam for the first N steps
    ):
        self.adam = adam
        self.delta2 = delta2
        self.signal = signal
        self.loss_window = loss_window
        self.loss_stable_threshold = loss_stable_threshold
        self.grad_norm_stable_threshold = grad_norm_stable_threshold
        self.entropy_stable_threshold = entropy_stable_threshold
        self.warmup_steps = warmup_steps

        # Sliding window of recent losses — used to detect "stable vs spiking"
        # deque with maxlen=N drops the oldest item when full, so we always
        # have at most `loss_window` recent values.
        self._loss_history = deque(maxlen=loss_window)

        # Counters for logging — how many steps each optimizer has run
        self._step = 0
        self._adam_steps = 0
        self._delta2_steps = 0

        # Last decision for inspection — useful when training metrics are logged
        self._last_decision: Optional[StepDecision] = None

        if signal not in ("loss", "grad_norm", "entropy"):
            raise ValueError(
                f"signal must be one of 'loss', 'grad_norm', 'entropy'; got {signal!r}"
            )
        logger.info(
            f"DialecticalController initialized: signal={signal}, "
            f"warmup_steps={warmup_steps}, loss_window={loss_window}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def zero_grad(self, set_to_none: bool = True):
        """Clear gradients on BOTH optimizers' parameters.

        Why both? Because both optimizers were constructed over the SAME
        parameter set (the model's). Calling zero_grad on either one would
        clear the same gradients. We call both for symmetry and explicitness.
        """
        self.adam.zero_grad(set_to_none=set_to_none)
        self.delta2.zero_grad(set_to_none=set_to_none)

    def step(
        self,
        loss: Optional[float] = None,
        grad_norm: Optional[float] = None,
        entropy: Optional[float] = None,
    ) -> StepDecision:
        """
        Decide which optimizer to use this step, then run it.

        Args:
            loss:      scalar loss value — required if signal="loss"
            grad_norm: total gradient norm — required if signal="grad_norm"
            entropy:   output distribution entropy — required if signal="entropy"

        Returns:
            StepDecision describing what was done and why.
        """
        self._step += 1

        # ── Phase 1: Warmup ──────────────────────────────────────────────
        # During warmup we always use Adam. δ² needs a "terrain" to operate
        # on — if we throw it at a freshly initialized random network, the
        # bassin fills with noise. Let Adam find a basin first, then start
        # alternating.
        if self._step <= self.warmup_steps:
            self.adam.step()
            self._adam_steps += 1
            decision = StepDecision(
                used_optimizer="adam",
                signal_value=0.0,
                reason=f"warmup ({self._step}/{self.warmup_steps})",
            )
            self._last_decision = decision
            return decision

        # ── Phase 2: Pick optimizer based on confidence signal ──────────
        decision = self._decide(loss=loss, grad_norm=grad_norm, entropy=entropy)

        if decision.used_optimizer == "adam":
            self.adam.step()
            self._adam_steps += 1
        else:  # "delta2"
            self.delta2.step()
            self._delta2_steps += 1

        self._last_decision = decision
        return decision

    # ── Decision logic ────────────────────────────────────────────────────

    def _decide(
        self,
        loss: Optional[float],
        grad_norm: Optional[float],
        entropy: Optional[float],
    ) -> StepDecision:
        """
        The actual if/else: look at the signal, return a StepDecision.

        Three branches based on `self.signal`. Each branch:
          1. Reads its signal from the args
          2. Updates any rolling history needed
          3. Compares against the threshold
          4. Returns either ("adam", "stable") or ("delta2", "spiking/uncertain")
        """
        if self.signal == "loss":
            if loss is None:
                raise ValueError(
                    "signal='loss' requires the `loss` arg passed to step()"
                )
            return self._decide_by_loss(loss)

        if self.signal == "grad_norm":
            if grad_norm is None:
                raise ValueError(
                    "signal='grad_norm' requires the `grad_norm` arg passed to step()"
                )
            return self._decide_by_grad_norm(grad_norm)

        if self.signal == "entropy":
            if entropy is None:
                raise ValueError(
                    "signal='entropy' requires the `entropy` arg passed to step()"
                )
            return self._decide_by_entropy(entropy)

        # Should never reach here because of __init__ validation
        raise ValueError(f"Unknown signal: {self.signal}")

    def _decide_by_loss(self, loss: float) -> StepDecision:
        """
        Compare the current loss against a rolling average of recent losses.

        If the loss has been stable (small change), use Adam — we've found
        a basin, descend further into it.

        If the loss is spiking or oscillating, use δ² — we're encountering
        contradictions that should be retained, not smoothed away.
        """
        self._loss_history.append(loss)

        # Need enough history to compute a meaningful average. Until then,
        # default to Adam.
        if len(self._loss_history) < self.loss_window:
            return StepDecision(
                used_optimizer="adam",
                signal_value=loss,
                reason=f"warming up loss window ({len(self._loss_history)}/{self.loss_window})",
            )

        # Compute mean and std of recent losses
        recent = list(self._loss_history)
        mean_loss = sum(recent) / len(recent)
        variance = sum((x - mean_loss) ** 2 for x in recent) / len(recent)
        std_loss = variance ** 0.5

        # "Stable" = current loss is close to recent mean.
        # We measure this as relative deviation: |loss - mean| / (std + epsilon).
        # If this is small, the loss is in its usual range. If it's large,
        # something surprising just happened.
        deviation = abs(loss - mean_loss) / (std_loss + 1e-8)

        if deviation < self.loss_stable_threshold * 10:
            # Stable: use Adam to keep descending into the basin
            return StepDecision(
                used_optimizer="adam",
                signal_value=deviation,
                reason=f"loss stable (deviation={deviation:.3f})",
            )
        else:
            # Spiking: use δ² to retain the surprise in the bassin
            return StepDecision(
                used_optimizer="delta2",
                signal_value=deviation,
                reason=f"loss spike (deviation={deviation:.3f})",
            )

    def _decide_by_grad_norm(self, grad_norm: float) -> StepDecision:
        """
        Decide based on total gradient norm.

        Small gradient norm = we're near a minimum, gradients are small →
        Adam (descend the last little bit).

        Large gradient norm = we're being pulled hard, gradients are big →
        δ² (the contradictions are real, retain them).
        """
        if grad_norm < self.grad_norm_stable_threshold:
            return StepDecision(
                used_optimizer="adam",
                signal_value=grad_norm,
                reason=f"grad norm small ({grad_norm:.3f})",
            )
        else:
            return StepDecision(
                used_optimizer="delta2",
                signal_value=grad_norm,
                reason=f"grad norm large ({grad_norm:.3f})",
            )

    def _decide_by_entropy(self, entropy: float) -> StepDecision:
        """
        Decide based on output distribution entropy.

        Low entropy = the model has a confident answer → Adam.
        High entropy = the model is uncertain → δ² (this is exactly the
                       moment when contradictions are most informative).

        Entropy units depend on log base — we assume natural log here, so
        for a vocab of size V, entropy ranges from 0 (one-hot) to ln(V)
        (uniform).
        """
        if entropy < self.entropy_stable_threshold:
            return StepDecision(
                used_optimizer="adam",
                signal_value=entropy,
                reason=f"entropy low ({entropy:.3f}) — confident",
            )
        else:
            return StepDecision(
                used_optimizer="delta2",
                signal_value=entropy,
                reason=f"entropy high ({entropy:.3f}) — uncertain",
            )

    # ── Inspection / metrics ──────────────────────────────────────────────

    def get_stats(self) -> dict:
        """
        Return controller statistics for logging and dashboards.

        Useful in metrics.jsonl: how often is each optimizer chosen?
        Are we mostly stable (Adam-heavy) or mostly contested (δ²-heavy)?
        """
        total = max(self._step, 1)
        return {
            "total_steps": self._step,
            "adam_steps": self._adam_steps,
            "delta2_steps": self._delta2_steps,
            "adam_fraction": self._adam_steps / total,
            "delta2_fraction": self._delta2_steps / total,
            "last_decision": self._last_decision.to_dict() if self._last_decision else None,
        }


# ============================================================================
# CONVENIENCE FACTORY
# ============================================================================

def build_controller_from_model(
    model: torch.nn.Module,
    config,           # TrainingConfig from config.py
) -> DialecticalController:
    """
    Convenience builder: create both Adam and δ² optimizers over the model's
    parameters and wrap them in a DialecticalController.

    This is what train.py calls when config.optimizer == "controller".

    The Adam half uses standard hyperparameters from config.adam_*.
    The δ² half uses standard hyperparameters from config.d2_*.
    """
    # ── Build Adam over decay/no-decay groups (standard practice) ─────────
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

    optim_groups = [
        {"params": decay_params, "weight_decay": config.adam_weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    adam = torch.optim.AdamW(
        optim_groups,
        lr=config.adam_lr,
        betas=(config.adam_beta1, config.adam_beta2),
    )

    # ── Build δ² over the SAME parameter set ─────────────────────────────
    # Both optimizers see the same parameters; only one of them runs each
    # step (selected by the controller). State is independent.
    delta2 = DeltaSquaredOptimizer(
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

    # ── Read controller settings from config (with sane defaults) ─────────
    signal = getattr(config, "controller_signal", "loss")
    loss_window = getattr(config, "controller_loss_window", 50)
    loss_stable_threshold = getattr(config, "controller_loss_stable_threshold", 0.01)
    grad_norm_stable_threshold = getattr(config, "controller_grad_norm_threshold", 1.0)
    entropy_stable_threshold = getattr(config, "controller_entropy_threshold", 1.5)
    warmup_steps = getattr(config, "controller_warmup_steps", 100)

    return DialecticalController(
        adam=adam,
        delta2=delta2,
        signal=signal,
        loss_window=loss_window,
        loss_stable_threshold=loss_stable_threshold,
        grad_norm_stable_threshold=grad_norm_stable_threshold,
        entropy_stable_threshold=entropy_stable_threshold,
        warmup_steps=warmup_steps,
    )
