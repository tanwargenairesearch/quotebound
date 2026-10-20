#!/usr/bin/env bash
# The receipts recipe end to end: CORD v2's labels, a LoRA adapter on Qwen3.5-4B, every value copied from the receipt,
# scored with and without the adapter, with exact drafts, and with the constraint off.
#
#   recipes/receipts/run.sh MODEL_DIR WORK_DIR
#
# MODEL_DIR is a Qwen3.5 checkpoint (hf download Qwen/Qwen3.5-4B --local-dir MODEL_DIR). WORK_DIR gets the prepared
# JSONL, the adapter (run/) and the four answer files. prepare.py needs pyarrow and huggingface_hub (pip install
# pyarrow huggingface_hub). Set QUOTEBOUND to use a binary other than build/quotebound. Training and generation resume
# where they stopped, so the script can be run again after an interruption.
set -euo pipefail
[[ $# -eq 2 ]] || { sed -n '2,10p' "$0"; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
qb=${QUOTEBOUND:-$here/../../build/quotebound}
model=$1
work=$2
mkdir -p "$work"

[[ -f $work/gold.jsonl ]] || python3 -I "$here/prepare.py" --out "$work"

"$qb" train --model "$model" --train "$work/train.jsonl" --dev "$work/dev.jsonl" --out "$work/run" --epochs 2 \
  --eval-every 25 --save-every 25 --gen-dev 4 | tee -a "$work/train.out"

constrained=(--schema "$here/schema.json" --quote-source user)
for v in base tuned tuned_drafts tuned_free; do
  extra=()
  case $v in tuned*) extra+=(--adapter "$work/run/best") ;; esac
  case $v in *_drafts) extra+=(--draft-train "$work/train.jsonl" --draft-k 8) ;; esac
  case $v in *_free) ;; *) extra+=("${constrained[@]}") ;; esac
  "$qb" generate --model "$model" --batch "$work/eval.jsonl" --max-tokens 768 "${extra[@]}" --out "$work/$v.jsonl"
done

python3 -I "$here/score.py" --data "$work" "$work"/{base,tuned,tuned_drafts,tuned_free}.jsonl
python3 -I "$here/score.py" --data "$work" --same "$work/tuned.jsonl" "$work/tuned_drafts.jsonl"
