# Anamnesis-δ²

A dialectical optimization framework with structured negative retention.

## Status

Formalization complete (see [formal definitions](https://github.com/elfege/RESEARCH/blob/master/formal_definitions_RS_and_delta_squared.md)).
Reference implementation scaffolded in this directory. Empirical benchmarks
against Adam on WikiText-103 pending — first run will test δ² in isolation as
the simplest claim; the full dialectical design (§Composition with Adam below)
is the intended framework, of which standalone δ² is one moment.

## The Idea in One Paragraph

Standard adaptive optimizers (Adam, RMSProp, AdaGrad) maintain an exponential
moving average of squared gradients and use it to **dampen** the update —
high-variance directions get smaller steps. The δ² optimizer uses the *same
machinery* with **inverted purpose**: it treats high-friction directions as
productive content and **injects** bounded tension into them. A "tension
reservoir" (bassin de tenseurs potentiels) retains past squared frictions so
the model can recall them at inference time when its current output
distribution is high-entropy — i.e., when it is in doubt.

## The Formulas

```
Adam  family:  W_next = W_now − α × m / √v                       (suppress noise)
                       v = β·v_{n-1} + (1-β)·g²                   (EMA of squared grads)

δ²:            W_next = W_now + η × tanh(B)                       (inject bounded tension)
                       B = γ·B_{n-1} + (1-γ)·(α₁ δ₁² + α₂ δ₂²)   (EMA of signed-squared frictions)
                       δ_sq = sign(δ) ⊙ (δ ⊙ δ)                   (signed squaring — preserves direction)
```

Same bookkeeping (EMA of squared gradients), opposite operation on it.

Where:
- **δ₁** = logical friction: `F(W)^(1/2) ⊙ (W − W̄)` — the Fisher-weighted drift from a reference state. Three candidates for W̄ (random init, EMA, Fisher-weighted) discussed in the formal definitions.
- **δ₂** = empirical friction: the standard gradient `∇L` — same quantity Adam uses.
- **B** = tension reservoir: exponential moving average of the signed-squared frictions.
- **tanh** = bounding function (not free-form; keeps weight updates in (−η, +η) to prevent explosion — answers the naive-divergence objection).

## Composition with Adam (the honest claim)

δ² is not a replacement for Adam. In the negation taxonomy the two are in
**opposition** (same quantity, opposed intents) — not annihilation. The
complete framework preserves both:

```
# Training-time dialectical switching
if confidence(W, batch) > threshold:     # model's commitments are stable
    W = adam_step(W, ∇L)                  # stabilize toward this basin
else:                                    # model is contested
    W = delta_squared_step(W, ∇L)        # grow from the contradiction
    bassin.store(∇L, context=batch)      # retain the negation
```

The paper's first experiment tests δ² alone (simpler claim, cleaner signal).
The architecture anticipates the full dialectical composition.

## Relationship to Existing Literature

This work sits at the intersection of three lines of existing research. It is
a combination and reinterpretation, not a claim of unprecedented novelty:

- **GEM** (Gradient Episodic Memory, Lopez-Paz & Ranzato 2017) — retains
  past-task gradients and projects new updates against them. The bassin
  generalizes this.
- **EWC** (Elastic Weight Consolidation, Kirkpatrick et al. 2017) — uses
  Fisher Information to weight parameter drift. This is the direct ancestor
  of the δ₁ definition.
- **SAM** (Sharpness-Aware Minimization, Foret et al. 2020) — seeks contested
  directions rather than pure minima. Philosophically adjacent to δ²'s
  amplification of high-friction regions.

The δ² contribution on top of these: (i) a taxonomy of negation types used to
classify what the bassin retains, (ii) signed squaring (direction-preserving
amplification), (iii) a symmetric training-time / inference-time gating
structure, and (iv) the near-real-time learning loop described below, which
closes the circuit between inference and training without requiring offline
fine-tuning cycles.

### Fixed taxonomy, learned distribution

A clarification to preempt a common misreading: the four categories of
negation (inessential difference, essential difference, opposition,
annihilation) are **fixed structural categories** derived from the
underlying Hegelian framework — they are *a priori* and do not evolve
from data. The system does **not** learn new categories. It is
**not** a Piagetian/constructivist system in which schemas accommodate
to novel experience.

What the system *does* learn is the **distribution over those fixed
categories across its own weight space and input regions**. A region of
weight space that initially produces "essential difference" frictions
might, after enough accumulated bassin entries, start producing
"annihilation" frictions in a recurring pattern. That pattern signals
to the controller — not "I need a new category" but "this region is
structurally unstable; the dialogue/search trigger should fire here."

**Categories fixed; distribution over categories learned.** This is
coherent with the Hegelian framework (categories necessary, not
contingent) while still adaptive behaviorally.

See also: [`docs/bitter_lesson/_README_on_the_bitter_lesson.md`](../docs/bitter_lesson/_README_on_the_bitter_lesson.md)
for the full discussion of what is and isn't novel here, the
heuristic/epistemogenetic distinction, and Sutton's "Bitter Lesson"
critique applied to this project.

## Near Real-Time Learning

A core goal of the architecture — not an afterthought — is that the model
updates from live conversation, not only from pre-batched training data. The
infrastructure for this is already present in the Anamnesis app (chat session
persistence, feedback collection, episode embedding) and composes with the
δ² loop as follows:

```
# At inference (every user turn)
    response = generate(context, bassin)       # standard pass with bassin recall

# User feedback arrives (thumbs down, correction, "no, I meant X")
    friction = embed(expected) − embed(response)    # = δ₂ at the output level
    bassin.store(friction,
                 context=context,
                 negation_type=classify(friction), # inessential | essential | opposition | annihilation
                 tension_score=||friction||²)

# Periodically (every N turns, or nightly)
    if bassin.accumulated_high_tension() > threshold:
        batch = bassin.sample_by_tension()     # draw the strongest retained frictions
        W = delta_squared_step(W, batch)       # a live weight update
        log_checkpoint(W)                      # reversible — can roll back
```

**Why this is more than chat-based fine-tuning:** standard RLHF / DPO pipelines
also use feedback, but only as unstructured +/− signal. The δ² pipeline
classifies *what kind* of negation the feedback represents (opposition vs
annihilation vs essential difference) and weights the update accordingly. A
user who says "technically correct but missing context" produces a different
update than a user who says "factually wrong" — same valence, different type.

**Why it is *near* real-time rather than real-time:** a live gradient step per
user turn would be unstable (one user could corrupt the weights). The bassin
acts as a rate-limiter: frictions accumulate, classified and tension-scored,
and the actual weight update happens when the reservoir has enough structured
signal to justify it. This is also what lets the update be reversible — each
δ²-step checkpoint can be rolled back if the bassin sample was dominated by
one bad-faith interaction.

The chat-session + feedback + episode-embedding infrastructure that the
Anamnesis app already ships (see `app/routes/chat.py`, `app/database.py`,
`app/training_pipeline.py`) is the substrate this loop runs on. No additional
persistence layer is required.

## Quick Start

```bash
# 1. Prepare data
pip install datasets tiktoken
python d2/data/prepare_wikitext.py

# 2. Run baseline (Adam)
python d2/train.py --optimizer adam --experiment adam_wikitext

# 3. Run experiment (δ²)
python d2/train.py --optimizer delta2 --experiment delta2_wikitext

# 4. Compare
python d2/experiments/compare.py \
    --adam d2/output/adam_wikitext/metrics.jsonl \
    --delta2 d2/output/delta2_wikitext/metrics.jsonl
```

## File Structure

```
d2/
├── neural_network.py     # Transformer architecture (the body — standard, not novel)
├── optimizer.py           # δ² optimizer (THE novel part)
├── bassin.py              # Negation classifier + tension storage + uncertainty detection
├── train.py               # Training loop (works with both Adam and δ²)
├── inference.py           # Generation with bassin recall on uncertainty
├── config.py              # All hyperparameters
├── data/
│   └── prepare_wikitext.py   # Download and tokenize WikiText-103
└── experiments/
    └── compare.py            # Compare δ² vs Adam results
```

## Theoretical Background

- Essay 1: [The Arithmetic Aufhebung](https://github.com/elfege/RESEARCH/blob/master/essai_aufhebung_arithmetique.md)
- Essay 2: [Contingency as Reservoir of Determination](https://github.com/elfege/RESEARCH/blob/master/essai_contingence_et_determination.md)
- Formal Definitions: [RS and δ²](https://github.com/elfege/RESEARCH/blob/master/formal_definitions_RS_and_delta_squared.md)
- Full research repository: [github.com/elfege/RESEARCH](https://github.com/elfege/RESEARCH)

## Author

Elfège Arthur Leylavergne — Software Engineer · Ph.D. Philosophy (Logic & Epistemology, specialization Hegel's *Science of Logic*), Université de Nantes, 2014.
