# Reproduce the final submitted training configuration.
# Path-like settings are required by run_train_infer.sh:
#   DATA_DIR=/path/to/data MODEL_ID=/path/to/base_model OUTPUT_ROOT=/path/to/outputs \
#   HF_CACHE_DIR=/path/to/hf_cache LOG_DIR=/path/to/logs bash run_final_train.sh

export USE_CAPTION="${USE_CAPTION:-true}"
export LORA_TARGETS="${LORA_TARGETS:-attn}"
export LORA_R="${LORA_R:-16}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export IMG_SIZE="${IMG_SIZE:-224}"
export PROMPT_METADATA="${PROMPT_METADATA:-none}"
export AUGMENT="${AUGMENT:-false}"
export USE_DORA="${USE_DORA:-false}"
export USE_COT="${USE_COT:-false}"
export USE_MARGIN_LOSS="${USE_MARGIN_LOSS:-false}"

bash run_train_infer.sh "$@"
