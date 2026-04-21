# Anamnesis-δ² (R2D2)

A dialectical optimization framework with structured negative retention.

## The Idea in One Paragraph

Standard neural network training uses gradient descent: subtract error, converge, stop. The δ² optimizer does something different: it **squares** the friction (preserving direction, amplifying magnitude) and **accumulates** it in a tension reservoir (bassin de tenseurs potentiels). Instead of forgetting what was discarded, the system remembers it. At inference time, when the model is uncertain, it recalls past tensions to inform its response. Convergence is functional death; δ² aims for bounded growth.

## The Formula

```
Standard:  W_next = W_now - α × ∇L                              (subtract error)
δ²:        W_next = W_now + η × f(B)                             (inject bounded tension)
           B_next = γ × B + (1-γ) × (α₁ × δ₁_sq + α₂ × δ₂_sq) (accumulate friction)
           δ_sq   = sign(δ) ⊙ (δ ⊙ δ)                           (signed squaring)
```

Where:
- **δ₁** = logical friction (how far weights drifted from baseline — a priori)
- **δ₂** = empirical friction (the gradient — same as standard ML)
- **B** = tension reservoir (exponential moving average of signed-squared frictions)
- **f** = bounding function (tanh) preventing weight explosion

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

- Essay 1: [The Arithmetic Aufhebung](../../0_ACADEMICS/essai_aufhebung_arithmetique.md)
- Essay 2: [Contingency as Reservoir of Determination](../../0_ACADEMICS/essai_contingence_et_determination.md)
- Formal Definitions: [RS and δ²](../../0_ACADEMICS/formal_definitions_RS_and_delta_squared.md)

## Author

Elfège Arthur Leylavergne — Ph.D. Philosophy (Hegel/Logic), Université de Nantes, 2014.
