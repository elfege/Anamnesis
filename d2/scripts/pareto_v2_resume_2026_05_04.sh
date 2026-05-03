#!/usr/bin/env bash
# Resume Pareto v2 sweep — agent #3 restarted the d² container mid-run.
# Already done (8 runs): η ∈ {1e-3, 3e-3} × 3 seeds + η=1e-2 seeds 0,1.
# Missing (8 runs): η=1e-2 seed 2, η=3e-2 × 3, η=1e-1 × 3, adam baseline.

set -uo pipefail

OUT_DIR=/workspace/checkpoints_bench/pareto_v2_2026-05-04
LOG=/workspace/checkpoints_personal/_pareto_v2_resume_2026-05-04.log
mkdir -p "$OUT_DIR"

exec >"$LOG" 2>&1
echo "=== pareto v2 resume start $(date -Iseconds) ==="

run() {
    local label="$1"; shift
    echo "─── $label  $(date -Iseconds) ───"
    "$@"
    echo "─── $label  exit=$?  $(date -Iseconds) ───"
}

# 1e-2 seed 2 (was killed mid-run)
run "η=1e-2 seed=2" python -m d2.experiments.continual \
    --method delta2_additive --benchmark permuted_mnist --tasks 5 --epochs 1 \
    --d2-eta 1e-2 \
    --output "$OUT_DIR/permuted_mnist_eta_1e-2_seed_2.json" \
    --device cuda --seed 2 || echo "  failed, continuing"

for eta in 3e-2 1e-1; do
    for seed in 0 1 2; do
        run "η=${eta} seed=${seed}" python -m d2.experiments.continual \
            --method delta2_additive --benchmark permuted_mnist --tasks 5 --epochs 1 \
            --d2-eta "$eta" \
            --output "$OUT_DIR/permuted_mnist_eta_${eta}_seed_${seed}.json" \
            --device cuda --seed "$seed" \
            || echo "  failed, continuing"
    done
done

run "adam baseline (seed=0)" python -m d2.experiments.continual \
    --method adam --benchmark permuted_mnist --tasks 5 --epochs 1 \
    --output "$OUT_DIR/permuted_mnist_adam_seed_0.json" \
    --device cuda --seed 0

echo "=== pareto v2 resume end $(date -Iseconds) ==="
ls -la "$OUT_DIR/" | wc -l
