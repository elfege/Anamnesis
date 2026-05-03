#!/usr/bin/env bash
# Bench WikiText-103 — δ² (additive) vs Adam, next-token prediction.
#
# This is the bench-track LM workload. It mirrors the permuted-MNIST sweeps
# (same delta2_additive optimizer path), but the workload is a real language
# modeling task on a public, reproducible corpus.
#
# Pipeline:
#   1. Tokenized .bin files prepared by d2/data/prepare_wikitext.py (PREREQ).
#      If d2/data/wikitext/train.bin is missing, this script aborts before
#      firing any GPU work — DO NOT silently re-prepare on the GPU host.
#   2. Run d2/train.py with the GPT-2-small architecture (~124M params),
#      block_size 512, batch_size 4, 2000 steps. Output cross-entropy
#      train/val + final perplexity. Cost on a 3090: ~25 min.
#   3. Two arms: --optimizer delta2 (η=1e-6) and --optimizer adam (lr=3e-4).
#
# Output: /workspace/checkpoints_bench/wikitext103_2026-05-04/{delta2,adam}/
#
# WHY THIS SCRIPT EXISTS:
#   The morning report (README_morning_2026-05-04.md) makes it clear the
#   permuted-MNIST sweeps showed no measurable δ² benefit. WikiText-103 is
#   the canonical "next sanity check": if δ² genuinely helps with structured
#   forgetting, an LM workload should show it more than a one-shot MLP.
#   This script is REGISTERED in the Training Catalog but NOT auto-fired.
#   The user picks when (and where) to run it from the dashboard.
#
# Estimated runtime:
#   - 1660 SUPER (server):     ~2 hours per arm
#   - 3090 (office):           ~25 min per arm
#   - A100 PCIe (RunPod):      ~10 min per arm

set -uo pipefail

OUT_DIR=/workspace/checkpoints_bench/wikitext103_2026-05-04
LOG=/workspace/checkpoints_bench/_bench_wikitext103_2026-05-04.log
DATA_DIR=/workspace/d2/data/wikitext
mkdir -p "$OUT_DIR" "$(dirname "$LOG")"

exec >"$LOG" 2>&1
echo "=== bench wikitext103 start $(date -Iseconds) ==="

# Prereq check: data must already be tokenized
if [ ! -f "$DATA_DIR/train.bin" ] || [ ! -f "$DATA_DIR/val.bin" ]; then
    echo "ERROR: WikiText-103 .bin files missing under $DATA_DIR"
    echo "Run d2/data/prepare_wikitext.py once on the GPU host:"
    echo "    docker exec anamnesis-d2 python /app/d2/data/prepare_wikitext.py"
    echo "(downloads ~500MB, tokenizes with tiktoken GPT-2 BPE, ~5 min)."
    echo "=== bench wikitext103 abort (no data) $(date -Iseconds) ==="
    exit 2
fi

nvidia-smi --query-gpu=name,memory.free,memory.used --format=csv,noheader 2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run() {
    local label="$1"; shift
    echo
    echo "─── $label  $(date -Iseconds) ───"
    "$@"
    local rc=$?
    echo "─── $label  exit=$rc  $(date -Iseconds) ───"
    return $rc
}

# Arm 1: δ²-additive
run "wikitext_delta2" python -m d2.train \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR/delta2" \
    --block-size 512 \
    --batch-size 4 \
    --steps 2000 \
    --eval-interval 200 \
    --eval-batches 20 \
    --optimizer delta2 \
    --d2-eta 1e-6 \
    --device cuda --seed 0 \
    || echo "δ² arm failed — continuing to Adam"

# Arm 2: Adam baseline
run "wikitext_adam" python -m d2.train \
    --data-dir "$DATA_DIR" \
    --output-dir "$OUT_DIR/adam" \
    --block-size 512 \
    --batch-size 4 \
    --steps 2000 \
    --eval-interval 200 \
    --eval-batches 20 \
    --optimizer adam \
    --lr 3e-4 \
    --device cuda --seed 0 \
    || echo "Adam arm failed"

echo
echo "=== bench wikitext103 end $(date -Iseconds) ==="
ls -la "$OUT_DIR/" 2>/dev/null || true
