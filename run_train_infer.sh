#!/usr/bin/env bash
set -euo pipefail

# Final train + inference launcher for finetune_qlora_multibranch.py.
# Configure path-like settings explicitly before running, for example:
#   DATA_DIR=/path/to/data MODEL_ID=/path/to/base_model OUTPUT_ROOT=/path/to/outputs \
#   HF_CACHE_DIR=/path/to/hf_cache LOG_DIR=/path/to/logs bash run_train_infer.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${PY_SCRIPT:-${SCRIPT_DIR}/finetune_qlora_multibranch.py}"

DATA_DIR="${DATA_DIR:-/path/to/data}"
MODEL_ID="${MODEL_ID:-/path/to/base_model_or_hf_model_id}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/path/to/outputs}"
HF_CACHE_DIR="${HF_CACHE_DIR:-/path/to/hf_cache}"
LOG_DIR="${LOG_DIR:-/path/to/logs}"
VAL_CSV="${VAL_CSV:-auto}"

require_config() {
  local name="$1"
  local value="${!name:-}"
  if [[ -z "${value}" || "${value}" == /path/to* ]]; then
    echo "[ERROR] ${name} is not configured."
    echo "Set ${name} explicitly before running this script."
    exit 1
  fi
}

require_config DATA_DIR
require_config MODEL_ID
require_config OUTPUT_ROOT
require_config HF_CACHE_DIR
require_config LOG_DIR

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/${TIMESTAMP}}"
SUBMISSION_FILE="${SUBMISSION_FILE:-${OUTPUT_DIR}/submission.csv}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/run_${TIMESTAMP}.log}"

# Final-submission settings. Override with: FOO=bar bash run_train_infer.sh
IMG_SIZE="${IMG_SIZE:-224}"
LEGACY_RESIZE="${LEGACY_RESIZE:-false}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LORA_TARGETS="${LORA_TARGETS:-attn}"
USE_DORA="${USE_DORA:-false}"
ANSWER_PREFIX="${ANSWER_PREFIX:-The correct answer is:}"
PROMPT_METADATA="${PROMPT_METADATA:-none}"
USE_CAPTION="${USE_CAPTION:-true}"
AUGMENT="${AUGMENT:-false}"
NO_GRAD_CKPT="${NO_GRAD_CKPT:-false}"
USE_COT="${USE_COT:-false}"
USE_MARGIN_LOSS="${USE_MARGIN_LOSS:-false}"
MARGIN="${MARGIN:-0.5}"
SAVE_EPOCH_CKPTS="${SAVE_EPOCH_CKPTS:-false}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
CKPT_SPLIT="${CKPT_SPLIT:-0.1}"
MAX_TRAINABLE_PARAMS="${MAX_TRAINABLE_PARAMS:-5000000}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
LEARNING_RATE="${LEARNING_RATE:-2e-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
TRAIN_LIMIT="${TRAIN_LIMIT:-0}"
VAL_LIMIT="${VAL_LIMIT:-0}"
SEED="${SEED:-42}"

mkdir -p "${HF_CACHE_DIR}" "${OUTPUT_DIR}" "${LOG_DIR}"

if [[ ! -f "${DATA_DIR}/train.csv" || ! -f "${DATA_DIR}/test.csv" ]]; then
  echo "[ERROR] Missing ${DATA_DIR}/train.csv or ${DATA_DIR}/test.csv"
  exit 1
fi

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Cannot find Python script: ${PY_SCRIPT}"
  exit 1
fi

export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"

echo "Run ID : ${TIMESTAMP}"
echo "Data   : ${DATA_DIR}"
echo "Model  : ${MODEL_ID}"
echo "Output : ${OUTPUT_DIR}"
echo "Log    : ${LOG_FILE}"

ARGS=(
  --data_dir "${DATA_DIR}" \
  --model_id "${MODEL_ID}" \
  --output_dir "${OUTPUT_DIR}" \
  --submission_file "${SUBMISSION_FILE}" \
  --img_size "${IMG_SIZE}" \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --lora_targets "${LORA_TARGETS}" \
  --answer_prefix "${ANSWER_PREFIX}" \
  --prompt_metadata "${PROMPT_METADATA}" \
  --lr_scheduler "${LR_SCHEDULER}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --ckpt_split "${CKPT_SPLIT}" \
  --max_trainable_params "${MAX_TRAINABLE_PARAMS}" \
  --num_epochs "${NUM_EPOCHS}" \
  --learning_rate "${LEARNING_RATE}" \
  --grad_accum_steps "${GRAD_ACCUM_STEPS}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --logging_steps "${LOGGING_STEPS}" \
  --train_limit "${TRAIN_LIMIT}" \
  --val_limit "${VAL_LIMIT}" \
  --seed "${SEED}" \
  --val_csv "${VAL_CSV}"
)

[[ "${LEGACY_RESIZE}" = true ]] && ARGS+=(--legacy_resize)
[[ "${USE_DORA}" = true ]] && ARGS+=(--use_dora)
[[ "${USE_CAPTION}" = true ]] && ARGS+=(--use_caption)
[[ "${AUGMENT}" = true ]] && ARGS+=(--augment)
[[ "${NO_GRAD_CKPT}" = true ]] && ARGS+=(--no_grad_ckpt)
[[ "${USE_COT}" = true ]] && ARGS+=(--use_cot)
[[ "${USE_MARGIN_LOSS}" = true ]] && ARGS+=(--use_margin_loss --margin "${MARGIN}")
[[ "${SAVE_EPOCH_CKPTS}" = true ]] && ARGS+=(--save_epoch_ckpts)

python "${PY_SCRIPT}" "${ARGS[@]}" "$@" | tee "${LOG_FILE}"

echo "Done. Outputs : ${OUTPUT_DIR}"
echo "      Log     : ${LOG_FILE}"
