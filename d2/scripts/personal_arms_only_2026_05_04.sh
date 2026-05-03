#!/usr/bin/env bash
# Personal-corpus arms only — δ² then Adam. Pareto sweep already done.
# Run on server 1660 SUPER, gpt2-medium + LoRA on c_attn, fp16, batch=1 block=256.
#
# Why these settings: gpt2-large OOMed at 1st forward pass on the 1660 SUPER
# (only ~3 GB free). gpt2-medium (355M, ~700MB in fp16) leaves room for
# activations + LoRA + optimizer state.
#
# Bitsandbytes was uninstalled before this script runs (it was broken after
# torch was upgraded by torchvision install; we don't need 4-bit on this
# small model anyway). peft.import_utils.is_bnb_available() now returns False
# so peft skips the bnb dispatch branch.

set -uo pipefail

LOG=/workspace/checkpoints_personal/_personal_arms_2026-05-04.log
mkdir -p /workspace/checkpoints_personal

exec >"$LOG" 2>&1
echo "=== personal arms start $(date -Iseconds) ==="
nvidia-smi --query-gpu=memory.free,memory.used --format=csv,noheader 2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

run "personal_delta2_v1" python -m d2.finetune_lora \
    "${COMMON[@]}" \
    --experiment personal_delta2_v1 \
    --optimizer delta2 \
    --d2-eta 1e-6 || echo "δ² arm failed — continuing to Adam"

run "personal_adam_v1" python -m d2.finetune_lora \
    "${COMMON[@]}" \
    --experiment personal_adam_v1 \
    --optimizer adam \
    --lr 1e-4 || echo "Adam arm failed"

echo
echo "=== personal arms end $(date -Iseconds) ==="
ls -la /workspace/checkpoints_personal/personal_*/ 2>/dev/null || true
