#!/usr/bin/env bash
# Pareto v3 — sweep α₂ (bassin growth rate) instead of η.
#
# Findings from v1 + v2: across η ∈ {1e-7..1e-1} the δ² nudge produced
# essentially identical metrics. Reason: with α₁=1e-5, α₂=1e-4, γ=0.99,
# the steady-state bassin magnitude is ~1e-4. tanh(1e-4) ≈ 1e-4, so the
# injection η·tanh(B) is at most η × 1e-4 — i.e. 1e-5 even at η=1e-1,
# which is 1000× smaller than the gradient term (base_lr=1e-2 × grad ≈ 1e-2).
#
# To make δ² actually matter, the bassin must grow to non-trivial magnitudes
# where tanh starts to bend. That requires α₁/α₂ several orders of magnitude
# larger than the current defaults. This sweep targets α₂ × 100 to × 100,000:
#   α₂ ∈ {1e-2, 1e-1, 1e0, 1e1}  (current default × 100, × 1000, × 10000, × 100000)
# Held fixed: α₁ = α₂ × 0.1, η = 1e-2, base_lr = 1e-2 (eta == base_lr in v3).
#
# 4 alphas × 3 seeds = 12 runs, ~1 min each = ~12 min on 1660 SUPER.
# Adam baseline (3 seeds) for variance comparison.

set -uo pipefail

OUT_DIR=/workspace/checkpoints_bench/pareto_v3_2026-05-04
LOG=/workspace/checkpoints_personal/_pareto_v3_2026-05-04.log
mkdir -p "$OUT_DIR"

exec >"$LOG" 2>&1
echo "=== pareto v3 (alpha sweep) start $(date -Iseconds) ==="

run() {
    local label="$1"; shift
    echo "─── $label  $(date -Iseconds) ───"
    "$@"
    echo "─── $label  exit=$?  $(date -Iseconds) ───"
}

for alpha2 in 1e-2 1e-1 1e0 1e1; do
    # alpha1 scaled to keep ratio consistent with the original 1e-5/1e-4 = 0.1
    alpha1=$(python -c "print(float('$alpha2') * 0.1)")
    for seed in 0 1 2; do
        run "α₂=${alpha2} α₁=${alpha1} seed=${seed}" python -m d2.experiments.continual \
            --method delta2_additive --benchmark permuted_mnist --tasks 5 --epochs 1 \
            --d2-eta 1e-2 --d2-alpha1 "$alpha1" --d2-alpha2 "$alpha2" \
            --output "$OUT_DIR/permuted_mnist_alpha2_${alpha2}_seed_${seed}.json" \
            --device cuda --seed "$seed" \
            || echo "  failed, continuing"
    done
done

# Adam baseline for variance comparison
for seed in 0 1 2; do
    run "adam seed=${seed}" python -m d2.experiments.continual \
        --method adam --benchmark permuted_mnist --tasks 5 --epochs 1 \
        --output "$OUT_DIR/permuted_mnist_adam_seed_${seed}.json" \
        --device cuda --seed "$seed" \
        || echo "  failed, continuing"
done

echo "=== pareto v3 end $(date -Iseconds) ==="
ls -la "$OUT_DIR/" | wc -l
