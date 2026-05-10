#!/usr/bin/env bash
set -euo pipefail

# Inference-only launcher for infer_qlora.py.
# Configure path-like settings explicitly before running, for example:
#   DATA_DIR=/path/to/data MODEL_ID=/path/to/base_model ADAPTER_DIR=/path/to/adapter_best \
#   HF_CACHE_DIR=/path/to/hf_cache LOG_DIR=/path/to/logs bash run_infer.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${PY_SCRIPT:-${SCRIPT_DIR}/infer_qlora.py}"

DATA_DIR="${DATA_DIR:-/path/to/data}"
MODEL_ID="${MODEL_ID:-/path/to/base_model_or_hf_model_id}"
ADAPTER_DIR="${ADAPTER_DIR:-/path/to/adapter_best}"
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
require_config ADAPTER_DIR
require_config HF_CACHE_DIR
require_config LOG_DIR

# Inference settings must match the training run that produced the adapter.
IMG_SIZE="${IMG_SIZE:-224}"
LEGACY_RESIZE="${LEGACY_RESIZE:-false}"
ANSWER_PREFIX="${ANSWER_PREFIX:-The correct answer is:}"
PROMPT_METADATA="${PROMPT_METADATA:-none}"
VAL_LIMIT="${VAL_LIMIT:-0}"
USE_CAPTION="${USE_CAPTION:-true}"
USE_COT="${USE_COT:-false}"
ADAPTER_DIR_CAPTION="${ADAPTER_DIR_CAPTION:-}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SUBMISSION_FILE="${SUBMISSION_FILE:-${ADAPTER_DIR}/submission_${TIMESTAMP}.csv}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/infer_${TIMESTAMP}.log}"

mkdir -p "${HF_CACHE_DIR}" "${LOG_DIR}"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "[ERROR] Cannot find Python script: ${PY_SCRIPT}"
  exit 1
fi

if [[ ! -f "${DATA_DIR}/test.csv" ]]; then
  echo "[ERROR] Missing ${DATA_DIR}/test.csv"
  exit 1
fi

if [[ ! -d "${ADAPTER_DIR}" ]]; then
  echo "[ERROR] Adapter directory not found: ${ADAPTER_DIR}"
  exit 1
fi

export HF_HOME="${HF_CACHE_DIR}"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}"

echo "Run ID      : ${TIMESTAMP}"
echo "Data        : ${DATA_DIR}"
echo "Model       : ${MODEL_ID}"
echo "Adapter dir : ${ADAPTER_DIR}"
echo "Submission  : ${SUBMISSION_FILE}"
echo "Log         : ${LOG_FILE}"

ARGS=(
  --adapter_dir "${ADAPTER_DIR}" \
  --data_dir "${DATA_DIR}" \
  --model_id "${MODEL_ID}" \
  --submission_file "${SUBMISSION_FILE}" \
  --img_size "${IMG_SIZE}" \
  --answer_prefix "${ANSWER_PREFIX}" \
  --prompt_metadata "${PROMPT_METADATA}" \
  --val_limit "${VAL_LIMIT}" \
  --val_csv "${VAL_CSV}"
)

[[ "${LEGACY_RESIZE}" = true ]] && ARGS+=(--legacy_resize)
[[ "${USE_COT}" = true ]] && ARGS+=(--use_cot)
[[ "${USE_CAPTION}" = true ]] && ARGS+=(--use_caption)
[[ -n "${ADAPTER_DIR_CAPTION}" ]] && ARGS+=(--adapter_dir_caption "${ADAPTER_DIR_CAPTION}")

python "${PY_SCRIPT}" "${ARGS[@]}" "$@" | tee "${LOG_FILE}"

echo "Done. Submission : ${SUBMISSION_FILE}"
echo "      Log        : ${LOG_FILE}"
