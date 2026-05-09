### ECE-GY 7123 Deeplearning Final Kaggle Competition

Team member:

**Weikai Qu (wq2105)**

**Yifan Hu ()**



### Experiment

##### Loss Experiment

1. finetune_qlora_lettertoken_klogits.py: multiclass cross-entropy (softmax CE)

Usage:

```
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

2. finetune_qlora_lettertoken_klogits_weighted.py: class-weighted cross-entropy.

   Usage:

   ```
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

3. finetune_qlora_lettertoken_ls.py: cross-entropy with label smoothing

   Usage:

   ```
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

4. finetune_qlora_lettertoken_margin.py: cross-entropy plus hinge-style ranking margin

   Usage:

   ```
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

5. finetune_qlora_lettertoken_focal.py: CE + focal loss

   Usage:

   ```
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

6. finetune_qlora_lettertoken.py: multiclass cross-entropy over letter-token log-probabilities

   Usage: 

   ```
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

7. finetune_qlora_likelihood.py: multiclass cross-entropy over summed token log-likelihoods

   Usage:

   ```
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

   