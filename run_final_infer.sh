# Generate a submission from the final/best adapter checkpoint.
# Usage:
#   DATA_DIR=/path/to/data MODEL_ID=/path/to/base_model ADAPTER_DIR=/path/to/adapter_best \
#   HF_CACHE_DIR=/path/to/hf_cache LOG_DIR=/path/to/logs bash run_final_infer.sh

if [ -z "${ADAPTER_DIR}" ]; then
  echo "[ERROR] ADAPTER_DIR is not set."
  echo "Usage: DATA_DIR=/path/to/data MODEL_ID=/path/to/base_model ADAPTER_DIR=/path/to/adapter_best HF_CACHE_DIR=/path/to/hf_cache LOG_DIR=/path/to/logs bash run_final_infer.sh"
  exit 1
fi

export USE_CAPTION="${USE_CAPTION:-true}"
export IMG_SIZE="${IMG_SIZE:-224}"
export PROMPT_METADATA="${PROMPT_METADATA:-none}"
export USE_COT="${USE_COT:-false}"

bash run_infer.sh "$@"
