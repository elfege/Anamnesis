#!/usr/bin/env bash
# Pareto v4 — α₂ at the values where δ² actually bends.
#
# v3 found α₂=10 produces only tiny shifts (BWT -0.040 vs -0.048 at lower).
# Per-element gradient² on this MLP+MNIST setup is much smaller than initially
# estimated. To make tanh(bassin) actually bend (B reaches ~0.1+), need α₂ in
# the range {1e2, 1e3, 1e4}. v4 covers that.
#
# 3 alphas × 3 seeds = 9 runs.

set -uo pipefail

OUT_DIR=/workspace/checkpoints_bench/pareto_v4_2026-05-04
LOG=/workspace/checkpoints_personal/_pareto_v4_2026-05-04.log
mkdir -p "$OUT_DIR"

exec >"$LOG" 2>&1
echo "=== pareto v4 (alpha high) start $(date -Iseconds) ==="

run() {
    local label="$1"; shift
    echo "─── $label  $(date -Iseconds) ───"
    "$@"
    echo "─── $label  exit=$?  $(date -Iseconds) ───"
}

for alpha2 in 1e2 1e3 1e4; do
    alpha1=$(python -c "print(float('$alpha2') * 0.1)")
    for seed in 0 1 2; do
        run "α₂=${alpha2} seed=${seed}" python -m d2.experiments.continual \
            --method delta2_additive --benchmark permuted_mnist --tasks 5 --epochs 1 \
            --d2-eta 1e-2 --d2-alpha1 "$alpha1" --d2-alpha2 "$alpha2" \
            --output "$OUT_DIR/permuted_mnist_alpha2_${alpha2}_seed_${seed}.json" \
            --device cuda --seed "$seed" \
            || echo "  failed, continuing"
    done
done

echo "=== pareto v4 end $(date -Iseconds) ==="
ls -la "$OUT_DIR/" | wc -l
