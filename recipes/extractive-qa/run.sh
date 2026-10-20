#!/usr/bin/env bash
# The extractive-QA recipe end to end: SQuAD v1.1, a LoRA adapter on Qwen3.5-4B, answers bound to the passage's
# words, scored with and without the adapter and with and without exact drafts.
#
#   recipes/extractive-qa/run.sh MODEL_DIR WORK_DIR
#
# MODEL_DIR is a Qwen3.5 checkpoint (hf download Qwen/Qwen3.5-4B --local-dir MODEL_DIR). WORK_DIR gets SQuAD, the
# prepared JSONL, the adapter (run/) and the four answer files. Set QUOTEBOUND to use a binary other than
# build/quotebound. Training and generation resume where they stopped, so the script can be run again after an
# interruption.
set -euo pipefail
[[ $# -eq 2 ]] || { sed -n '2,10p' "$0"; exit 2; }
here=$(cd "$(dirname "$0")" && pwd)
qb=${QUOTEBOUND:-$here/../../build/quotebound}
model=$1
work=$2
mkdir -p "$work/squad"

for f in train-v1.1.json dev-v1.1.json; do
  [[ -f $work/squad/$f ]] || curl -fsSL -o "$work/squad/$f" "https://rajpurkar.github.io/SQuAD-explorer/dataset/$f"
done
shasum -a 256 -c <<EOF
3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955  $work/squad/train-v1.1.json
95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9  $work/squad/dev-v1.1.json
EOF
python3 -I "$here/prepare.py" --squad "$work/squad" --out "$work"

"$qb" train --model "$model" --train "$work/train.jsonl" --dev "$work/dev.jsonl" --out "$work/run" --epochs 1 \
  --eval-every 40 --save-every 40 --dev-max 200 --gen-dev 4 | tee -a "$work/train.out"

for v in base base_drafts tuned tuned_drafts; do
  extra=()
  case $v in tuned*) extra+=(--adapter "$work/run/best") ;; esac
  case $v in *_drafts) extra+=(--draft-train "$work/train.jsonl" --draft-k 8) ;; esac
  "$qb" generate --model "$model" --batch "$work/eval.jsonl" --schema "$here/schema.json" --quote-source user \
    --max-tokens 64 ${extra[@]+"${extra[@]}"} --out "$work/$v.jsonl"  # macOS's bash 3.2: an empty array under set -u
done

python3 -I "$here/score.py" --data "$work" "$work"/{base,base_drafts,tuned,tuned_drafts}.jsonl
python3 -I "$here/score.py" --data "$work" --same "$work/base.jsonl" "$work/base_drafts.jsonl"
python3 -I "$here/score.py" --data "$work" --same "$work/tuned.jsonl" "$work/tuned_drafts.jsonl"
