# Deep Learning Final Kaggle Competition

Team member:

**Weikai Qu (wq2105)**

**Yifan Hu (yh6416)**

### Experiment

##### Loss Experiment

1. finetune_qlora_lettertoken_klogits.py: multiclass cross-entropy (softmax CE)

Usage:

``` bash
    python finetune_qlora_lettertoken_klogits.py \
  --data_dir data \
  --model_id HuggingFaceTB/SmolVLM-500M-Instruct \
  --output_dir outputs/qlora_lettertoken \
  --submission_file klogits_epoch15.csv \
  --img_size 512 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --num_epochs 15 \
  --learning_rate 1.5e-4 \
  --grad_accum_steps 8 \
  --logging_steps 20 --train_batch_size 4
```

1. finetune_qlora_lettertoken_klogits_weighted.py: class-weighted cross-entropy.

   Usage:

   ``` bash
       python finetune_qlora_lettertoken_klogits_weighted.py \
     --data_dir data \
     --model_id HuggingFaceTB/SmolVLM-500M-Instruct \
     --output_dir outputs/qlora_lettertoken \
     --submission_file klogits_epoch15.csv \
     --img_size 224 \
     --lora_r 8 \
     --lora_alpha 16 \
     --lora_dropout 0.05 \
     --num_epochs 15 \
     --learning_rate 1.5e-4 \
     --grad_accum_steps 8 \
     --logging_steps 20 --train_batch_size 4 --eval_steps 200
   ```

2. finetune_qlora_lettertoken_ls.py: cross-entropy with label smoothing

   Usage:

   ``` bash
   python finetune_qlora_lettertoken_ls.py \
     --data_dir data \
     --model_id HuggingFaceTB/SmolVLM-500M-Instruct \
     --output_dir outputs/qlora_lettertoken_focal \
     --submission_file lettertoken_ls1.csv \
     --num_epochs 3 \
     --learning_rate 2e-4 \
     --train_batch_size 4 \
     --grad_accum_steps 8 \
     --label_smoothing 0.05
   ```

3. finetune_qlora_lettertoken_margin.py: cross-entropy plus hinge-style ranking margin

   Usage:

   ``` bash
   python finetune_qlora_lettertoken_margin.py \
     --data_dir data \
     --model_id HuggingFaceTB/SmolVLM-500M-Instruct \
     --output_dir outputs/qlora_lettertoken_margin \
     --submission_file lettertoken_margin01.csv \
     --num_epochs 3 \
     --learning_rate 2e-4 \
     --train_batch_size 4 \
     --grad_accum_steps 8 \
     --margin_lambda 0.1 --margin_m 0.1
   ```

4. finetune_qlora_lettertoken_focal.py: CE + focal loss

   Usage:

   ``` bash
   python finetune_qlora_lettertoken_focal.py \
     --data_dir data \
     --model_id HuggingFaceTB/SmolVLM-500M-Instruct \
     --output_dir outputs/qlora_lettertoken_focal \
     --submission_file my_submission_lettertoken_focal.csv \
     --num_epochs 3 \
     --learning_rate 2e-4 \
     --train_batch_size 4 \
     --grad_accum_steps 8 \
     --focal_gamma 2.0 \
     --focal_alpha 1.0
   ```

5. finetune_qlora_lettertoken.py: multiclass cross-entropy over letter-token log-probabilities

   Usage:

   ``` bash
       python finetune_qlora_lettertoken.py \
     --data_dir data \
     --model_id HuggingFaceTB/SmolVLM-500M-Instruct \
     --output_dir outputs/qlora_lettertoken \
     --submission_file my_submission_lettertoken.csv \
     --img_size 224 \
     --lora_r 8 \
     --lora_alpha 16 \
     --lora_dropout 0.05 \
     --num_epochs 15 \
     --learning_rate 1.5e-4 \
     --grad_accum_steps 8 \
     --logging_steps 20 --train_batch_size 4
   ```

6. finetune_qlora_likelihood.py: multiclass cross-entropy over summed token log-likelihoods

   Usage:

   ``` bash
   python finetune_qlora_likelihood.py \
     --data_dir data \
     --model_id HuggingFaceTB/SmolVLM-500M-Instruct \
     --output_dir outputs/qlora_likelihood \
     --submission_file my_submission_likelihood.csv \
     --img_size 224 \
     --lora_r 8 \
     --lora_alpha 16 \
     --num_epochs 1 \
     --learning_rate 2e-4 \
     --grad_accum_steps 8 \
     --logging_steps 20
   ```

### Final Reproduction

The final training and inference entry points are:

- `run_final_train.sh`: train the final configuration and write a submission from the best adapter.
- `run_final_infer.sh`: run inference only from an existing final/best adapter.

Install dependencies:

``` bash
pip install -r requirements.txt
```

Prepare the dataset directory so that it contains:

``` text
/path/to/data/
  train.csv
  test.csv
  val.csv        # optional, used only for per-category evaluation if present
  ...image files referenced by image_path...
```

The base model can be either a local model folder or a Hugging Face model id. For offline review, set `MODEL_ID` to a local folder containing the downloaded base model.

The two main training schemes used in the report can be launched as follows.

Train with caption-enhanced prompt:

``` bash
DATA_DIR=/path/to/data \
MODEL_ID=/path/to/base_model_or_HuggingFaceTB/SmolVLM-500M-Instruct \
OUTPUT_DIR=/path/to/outputs/caption_prompt \
OUTPUT_ROOT=/path/to/outputs \
HF_CACHE_DIR=/path/to/hf_cache \
LOG_DIR=/path/to/logs \
USE_CAPTION=true \
LORA_TARGETS=auto \
LORA_R=8 \
LORA_ALPHA=16 \
bash run_train_infer.sh
```

Inference from the caption-enhanced checkpoint:

``` bash
DATA_DIR=/path/to/data \
MODEL_ID=/path/to/base_model_or_HuggingFaceTB/SmolVLM-500M-Instruct \
ADAPTER_DIR=/path/to/outputs/caption_prompt/adapter_best \
HF_CACHE_DIR=/path/to/hf_cache \
LOG_DIR=/path/to/logs \
USE_CAPTION=true \
bash run_infer.sh
```

Train with caption + attention-only LoRA, `r=16`, `alpha=32`(our final submission configuration):

``` bash
DATA_DIR=/path/to/data \
MODEL_ID=/path/to/base_model_or_HuggingFaceTB/SmolVLM-500M-Instruct \
OUTPUT_DIR=/path/to/outputs/caption_attn_lora16_alpha32 \
OUTPUT_ROOT=/path/to/outputs \
HF_CACHE_DIR=/path/to/hf_cache \
LOG_DIR=/path/to/logs \
USE_CAPTION=true \
LORA_TARGETS=attn \
LORA_R=16 \
LORA_ALPHA=32 \
bash run_train_infer.sh
```

Inference from the caption + attention-only LoRA checkpoint:

``` bash
DATA_DIR=/path/to/data \
MODEL_ID=/path/to/base_model_or_HuggingFaceTB/SmolVLM-500M-Instruct \
ADAPTER_DIR=/path/to/outputs/caption_attn_lora16_alpha32/adapter_best \
HF_CACHE_DIR=/path/to/hf_cache \
LOG_DIR=/path/to/logs \
USE_CAPTION=true \
bash run_final_infer.sh
```

Run final training:

``` bash
DATA_DIR=/path/to/data \
MODEL_ID=/path/to/base_model_or_HuggingFaceTB/SmolVLM-500M-Instruct \
OUTPUT_ROOT=/path/to/outputs \
HF_CACHE_DIR=/path/to/hf_cache \
LOG_DIR=/path/to/logs \
bash run_final_train.sh
```

The script saves adapters under `/path/to/outputs/<timestamp>/`, including:

``` text
adapter_best/
adapter_last/
submission.csv
```

Download the best checkpoint archive from Google Drive and unzip it:

``` bash
pip install gdown

GOOGLE_DRIVE_URL=""
CKPT_ZIP=/path/to/best_checkpoint.zip
CKPT_DIR=/path/to/best_checkpoint

gdown "${GOOGLE_DRIVE_URL}" -O "${CKPT_ZIP}"
mkdir -p "${CKPT_DIR}"
unzip "${CKPT_ZIP}" -d "${CKPT_DIR}"
```

After unzipping, set `ADAPTER_DIR` to the extracted adapter folder. For example, if the archive contains `adapter_best/`, use:

``` bash
ADAPTER_DIR=/path/to/best_checkpoint/adapter_best
```

Run final inference from a saved adapter:

``` bash
DATA_DIR=/path/to/data \
MODEL_ID=/path/to/base_model_or_HuggingFaceTB/SmolVLM-500M-Instruct \
ADAPTER_DIR=/path/to/best_checkpoint/adapter_best \
HF_CACHE_DIR=/path/to/hf_cache \
LOG_DIR=/path/to/logs \
bash run_final_infer.sh
```

The inference script writes `submission_<timestamp>.csv` into `ADAPTER_DIR` by default. To choose an exact output file:

``` bash
DATA_DIR=/path/to/data \
MODEL_ID=/path/to/base_model_or_HuggingFaceTB/SmolVLM-500M-Instruct \
ADAPTER_DIR=/path/to/best_checkpoint/adapter_best \
SUBMISSION_FILE=/path/to/submission.csv \
HF_CACHE_DIR=/path/to/hf_cache \
LOG_DIR=/path/to/logs \
bash run_final_infer.sh
```

Optional overrides such as `NUM_EPOCHS`, `TRAIN_BATCH_SIZE`, `IMG_SIZE`, `USE_CAPTION`, and `VAL_CSV` can be passed as environment variables before the command. `VAL_CSV=auto` is the default and uses `/path/to/data/val.csv` if it exists.
