#!/usr/bin/env bash
# Overnight 2026-05-03 → 2026-05-04 — three runs, all on server 1660 SUPER, sequential.
#
# 1. Personal corpus × δ²-additive on gpt2-large (LoRA on c_attn)
# 2. Personal corpus × Adam control on gpt2-large (same recipe)
# 3. Pareto sweep: δ²-additive on permuted-MNIST across η ∈ {1e-7, 1e-6, 1e-5, 1e-4}
#
# Why gpt2-large: tokenizer matches the .bin files (anamnesis_to_tokens.py uses
# tiktoken GPT-2 BPE). Llama-3.2-3B-Instruct would need HF_TOKEN AND its tokenizer
# would mismatch the existing .bin token IDs. gpt2-large is 774M params, fp16,
# fits comfortably in ~3GB free VRAM on the 1660 SUPER, no gating, no token.
#
# Output: /workspace/checkpoints_personal/_overnight_2026-05-04.log + per-run subdirs.

set -uo pipefail

LOG=/workspace/checkpoints_personal/_overnight_2026-05-04.log
PARETO_DIR=/workspace/checkpoints_bench/pareto_2026-05-04
mkdir -p /workspace/checkpoints_personal "$PARETO_DIR"

exec >"$LOG" 2>&1
echo "=== overnight start $(date -Iseconds) ==="
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader 2>/dev/null || true

# Reduce memory fragmentation when allocating LoRA + activations on tight VRAM.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Switched gpt2-large -> gpt2-medium (355M) after first attempt OOMed: gpt2-large
# loaded fine but the FIRST forward pass had no room for activations on the 1660
# SUPER (only ~3GB free). gpt2-medium fits with margin: ~700MB for the model in
# fp16 + ~500MB activations + LoRA = comfortable.
COMMON=(
    --base-model gpt2-medium
    --lora-target-modules c_attn
    --dtype fp16
    --no-load-in-4bit
    --data-dir /workspace/data_personal/anamnesis_chronological
    --output-dir /workspace/checkpoints_personal
    --steps-per-task 300
    --block-size 256
    --batch-size 1
    --eval-interval 50
    --eval-batches 4
)

run() {
    local label="$1"; shift
    echo
    echo "─── $label  $(date -Iseconds) ───"
    "$@"
    local rc=$?
    echo "─── $label  exit=$rc  $(date -Iseconds) ───"
    return $rc
}

# Run 1: δ²
run "personal_delta2_v1" python -m d2.finetune_lora \
    "${COMMON[@]}" \
    --experiment personal_delta2_v1 \
    --optimizer delta2 \
    --d2-eta 1e-6 || echo "δ² arm failed — continuing to Adam"

# Run 2: Adam control
run "personal_adam_v1" python -m d2.finetune_lora \
    "${COMMON[@]}" \
    --experiment personal_adam_v1 \
    --optimizer adam \
    --lr 1e-4 || echo "Adam arm failed — continuing to Pareto sweep"

# Run 3: Pareto sweep on permuted-MNIST
for eta in 1e-7 1e-6 1e-5 1e-4; do
    out="$PARETO_DIR/permuted_mnist_eta_${eta}.json"
    run "pareto_eta_${eta}" python -m d2.experiments.continual \
        --method delta2_additive \
        --benchmark permuted_mnist \
        --tasks 5 \
        --epochs 1 \
        --d2-eta "$eta" \
        --output "$out" \
        --device cuda \
        --seed 0 \
        || echo "pareto η=${eta} failed — continuing"
done

echo
echo "=== overnight end $(date -Iseconds) ==="
echo "Outputs:"
ls -la /workspace/checkpoints_personal/personal_*/ 2>/dev/null || true
ls -la "$PARETO_DIR/" 2>/dev/null || true
