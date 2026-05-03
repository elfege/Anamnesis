#!/usr/bin/env bash
# Pareto sweep v2 — η in the range where δ² actually does something.
#
# Yesterday's sweep (η ∈ {1e-7..1e-4}) produced bit-identical metrics across
# all 4 etas because the δ²-additive nudge term η·tanh(B) was 2-5 orders of
# magnitude smaller than the gradient term (base_lr=1e-2). Effectively no δ².
#
# This sweep targets η ∈ {1e-3, 3e-3, 1e-2, 3e-2, 1e-1} where the δ² nudge
# is comparable to or larger than the gradient step — the range where δ²'s
# claim of structured retention should be measurable.
#
# 5 etas × 3 seeds = 15 runs, ~1 min each = ~15 min total on 1660 SUPER.

set -uo pipefail

OUT_DIR=/workspace/checkpoints_bench/pareto_v2_2026-05-04
LOG=/workspace/checkpoints_personal/_pareto_v2_2026-05-04.log
mkdir -p "$OUT_DIR"

exec >"$LOG" 2>&1
echo "=== pareto v2 start $(date -Iseconds) ==="

run() {
    local label="$1"; shift
    echo "─── $label  $(date -Iseconds) ───"
    "$@"
    echo "─── $label  exit=$?  $(date -Iseconds) ───"
}

for eta in 1e-3 3e-3 1e-2 3e-2 1e-1; do
    for seed in 0 1 2; do
        run "η=${eta} seed=${seed}" python -m d2.experiments.continual \
            --method delta2_additive \
            --benchmark permuted_mnist \
            --tasks 5 --epochs 1 \
            --d2-eta "$eta" \
            --output "$OUT_DIR/permuted_mnist_eta_${eta}_seed_${seed}.json" \
            --device cuda --seed "$seed" \
            || echo "  failed, continuing"
    done
done

# Also a single Adam baseline for comparison
run "adam baseline (seed=0)" python -m d2.experiments.continual \
    --method adam \
    --benchmark permuted_mnist \
    --tasks 5 --epochs 1 \
    --output "$OUT_DIR/permuted_mnist_adam_seed_0.json" \
    --device cuda --seed 0

echo "=== pareto v2 end $(date -Iseconds) ==="
ls -la "$OUT_DIR/"
