"""
QLoRA + letter-token-only scoring for ScienceQA-style MCQ.

Scheme:
- Build prefix ending with the answer prompt (default: "Answer:").
- Run ONE forward pass per sample.
- Use only the next-token logits at the last prefix position.
- Score each option by logit(letter_token | image, prefix).
- Train with CE over these K raw logit scores; infer by argmax.

Hyperparameter branches (all opt-in, default keeps original behaviour):

  Branch 1 – MLP-only LoRA:         --lora_targets mlp  (recommend --lora_r 6)
  Branch 2 – DoRA:                   --use_dora
  Branch 3 – Self-generated captions:--use_caption
  Branch 4 – Attn+MLP LoRA:         --lora_targets all  (recommend --lora_r 4)
  Branch 5 – Answer-prefix wording:  --answer_prefix "The correct answer is:"
  Branch 6 – Metadata in prompt:     --prompt_metadata subject  (or grade/topic)
  Branch 7 – Larger image:           --img_size 384  (SmolVLM native; enables tiling)
  Branch 8 – Data augmentation:      --augment
  Branch 9 – Disable grad ckpt:      --no_grad_ckpt  (enabled by default)
  Branch 10 – Chain-of-Thought:       --use_cot  (trains on solution field)
  Branch 11 – Margin loss:            --use_margin_loss --margin 0.5
           – Epoch ckpt saving:       --save_epoch_ckpts  (for weight averaging)

Image-processing modes:
  Default (--legacy_resize off): processor handles all resizing and tiling.
    --img_size controls processor's longest_edge (384 = SmolVLM native, enables tiling).
  --legacy_resize: old behaviour — manual PIL resize to img_size before the processor.

LR scheduling:
  --lr_scheduler {none,cosine,linear}  (default: cosine)
  --warmup_ratio FLOAT                 (default: 0.05)
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

try:
    from torchvision import transforms as T

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
CHOICE_LETTERS = "ABCDEFGHIJ"
MAX_TRAINABLE_PARAMETERS = 5_000_000


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def require_trainable_parameters_within_limit(
    model: torch.nn.Module, max_params: int = MAX_TRAINABLE_PARAMETERS
) -> int:
    n = count_trainable_parameters(model)
    if n > max_params:
        raise ValueError(
            f"Trainable parameters ({n:,}) exceed the limit ({max_params:,}). "
            "Lower --lora_r / --lora_alpha, or reduce --lora_targets, then retry."
        )
    print(f"Trainable parameter check OK: {n:,} / {max_params:,} (max).")
    return n


def resolve_data_dir(user_data_dir: str) -> Path:
    candidate = Path(user_data_dir)
    if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find train.csv and test.csv under data_dir: {candidate}"
    )


def resolve_resume_adapter_dir(resume_adapter_dir: str, output_dir: Path) -> Path | None:
    """Resolve adapter checkpoint directory for warm-start training."""
    if not resume_adapter_dir or not resume_adapter_dir.strip():
        return None
    raw = resume_adapter_dir.strip()
    p = Path(raw)
    if p.exists():
        return p
    p2 = output_dir / raw
    if p2.exists():
        return p2
    raise FileNotFoundError(
        f"resume_adapter_dir not found at '{p}' or '{p2}'."
    )


# ──────────────────────────────────────────────────────────────
# Image loading  (legacy vs. processor-native)
# ──────────────────────────────────────────────────────────────


def _load_image(path: Path, img_size: int, legacy_resize: bool) -> Image.Image:
    """
    Load an image as RGB PIL.
    legacy_resize=True  → manually resize to img_size×img_size (old behaviour).
    legacy_resize=False → return original; the processor handles resize + tiling.
    """
    image = Image.open(path).convert("RGB")
    if legacy_resize:
        image = image.resize((img_size, img_size), Image.BICUBIC)
    return image


def configure_processor_image_size(processor, img_size: int) -> None:
    """Point the processor's image_processor at the requested longest_edge."""
    try:
        processor.image_processor.size = {"longest_edge": img_size}
        print(
            f"Processor image size set to longest_edge={img_size} (tiling enabled by processor).")
    except AttributeError:
        print(
            "Warning: could not set processor image size programmatically. "
            "Pass --legacy_resize to use manual PIL resize instead."
        )


# ──────────────────────────────────────────────────────────────
# Prompt building  (Branch 3 / 5 / 6)
# ──────────────────────────────────────────────────────────────


def build_prompt(
    row: pd.Series,
    answer_prefix: str = "Answer:",
    prompt_metadata: str = "none",
    caption: str | None = None,
    use_cot: bool = False,
) -> str:
    context_parts: list[str] = []
    lecture = row.get("lecture", "")
    hint = row.get("hint", "")
    if pd.notna(lecture) and str(lecture).strip():
        context_parts.append(str(lecture).strip())
    if pd.notna(hint) and str(hint).strip():
        context_parts.append(str(hint).strip())
    context_str = "\n".join(context_parts)

    choices = row["choices"]
    choices_str = "\n".join(
        f"  {CHOICE_LETTERS[i]}. {choice}" for i, choice in enumerate(choices)
    )

    prompt = "<image>\n"

    # Branch 3: prepend self-generated caption
    if caption:
        prompt += f"Image description: {caption}\n\n"

    if context_str:
        prompt += f"Context:\n{context_str}\n\n"

    # Branch 6: inject one metadata field
    if prompt_metadata != "none":
        meta_val = row.get(prompt_metadata, "")
        if pd.notna(meta_val) and str(meta_val).strip():
            prompt += f"{prompt_metadata.capitalize()}: {str(meta_val).strip()}\n"

    prompt += f"Question: {row['question']}\n"
    prompt += f"Choices:\n{choices_str}\n"
    # Branch 10: CoT trigger instead of answer prefix
    if use_cot:
        prompt += "Let's think step by step.\n"
    else:
        prompt += answer_prefix  # Branch 5: configurable suffix
    return prompt


def build_cot_target(row: pd.Series) -> str:
    """Build target text for CoT training: solution + answer letter."""
    solution = row.get("solution", "")
    answer = int(row["answer"])
    solution_text = str(solution).strip() if pd.notna(
        solution) and str(solution).strip() else ""
    return f"{solution_text}\n\nThe correct answer is: {CHOICE_LETTERS[answer]}."


# ──────────────────────────────────────────────────────────────
# Tokenizer helpers
# ──────────────────────────────────────────────────────────────


def build_letter_token_map(processor) -> dict[str, int]:
    """Map each choice letter to ONE token id for next-token scoring."""
    token_map: dict[str, int] = {}
    for letter in CHOICE_LETTERS:
        token_ids = processor.tokenizer(
            f" {letter}", add_special_tokens=False
        ).input_ids
        if not token_ids:
            raise ValueError(
                f"Tokenizer produced no token for letter '{letter}'.")
        if len(token_ids) > 1:
            print(
                f"Warning: tokenizer splits ' {letter}' into {len(token_ids)} tokens; "
                "using the first token for letter-only scoring."
            )
        token_map[letter] = int(token_ids[0])
    return token_map


# ──────────────────────────────────────────────────────────────
# Branch 11: margin-based loss
# ──────────────────────────────────────────────────────────────


def margin_loss(scores: torch.Tensor, target: int, margin: float = 0.5) -> torch.Tensor:
    """
    Margin-based ranking loss for multi-choice scoring.
    Hinge: max(0, margin - score_correct + max(score_wrong)) + 0.1 * CE.
    """
    correct_score = scores[target]
    mask = torch.ones_like(scores, dtype=torch.bool)
    mask[target] = False
    max_wrong_score = scores[mask].max()
    hinge = F.relu(margin - correct_score + max_wrong_score)
    ce = F.cross_entropy(scores.unsqueeze(0), torch.tensor(
        [target], device=scores.device, dtype=torch.long))
    return hinge + 0.1 * ce


# ──────────────────────────────────────────────────────────────
# Branch 10: CoT answer parsing from generated text
# ──────────────────────────────────────────────────────────────


def _parse_answer_from_cot(generated_text: str, n_opt: int) -> int:
    """Parse the predicted answer letter from a CoT generation."""
    # Try pattern: "answer is: X" or "answer is X"
    m = re.search(r"answer\s+is\s*:?\s*([A-H])", generated_text, re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        idx = CHOICE_LETTERS.index(letter)
        if idx < n_opt:
            return idx

    # Fallback: last uppercase A-H letter in the text
    for ch in reversed(generated_text):
        if ch in CHOICE_LETTERS[:n_opt]:
            return CHOICE_LETTERS.index(ch)

    return 0  # ultimate fallback


# ──────────────────────────────────────────────────────────────
# LoRA target selection  (Branch 1 / 4)
# ──────────────────────────────────────────────────────────────

_ATTN_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
_MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]


def pick_lora_target_modules(model: torch.nn.Module, mode: str = "auto") -> list[str]:
    """
    mode:
      "auto" – all recognised projection names found in the model (original behaviour)
      "attn" – attention projections only  (Branch 1 baseline / higher rank)
      "mlp"  – MLP projections only        (Branch 1 main)
      "all"  – attn + MLP                  (Branch 4; use lower rank to stay under 5M)
    """
    if mode == "attn":
        wanted = set(_ATTN_MODULES)
    elif mode == "mlp":
        wanted = set(_MLP_MODULES)
    elif mode == "all":
        wanted = set(_ATTN_MODULES + _MLP_MODULES)
    else:  # "auto"
        wanted = set(_ATTN_MODULES + _MLP_MODULES)

    found: set[str] = set()
    for name, _ in model.named_modules():
        short_name = name.split(".")[-1]
        if short_name in wanted:
            found.add(short_name)

    if found:
        return sorted(found)
    return ["q_proj", "v_proj"]


# ──────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────


def maybe_build_qlora_model(model_id: str) -> tuple[torch.nn.Module, bool]:
    if not torch.cuda.is_available():
        print("CUDA not available. Falling back to full-precision without QLoRA.")
        m = AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True
        )
        return m, False
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    m = AutoModelForVision2Seq.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    m = prepare_model_for_kbit_training(m)
    return m, True


# ──────────────────────────────────────────────────────────────
# Branch 3: self-generated caption cache
# ──────────────────────────────────────────────────────────────


def build_caption_cache(
    model: torch.nn.Module,
    processor,
    df: pd.DataFrame,
    data_dir: Path,
    img_size: int,
    legacy_resize: bool,
    device: torch.device,
    batch_size: int = 8,
) -> dict[str, str]:
    """Generate image captions in batches (single forward pass per batch)."""
    prompt = (
        "<image>\nBriefly describe what is shown in this image, "
        "including any text, axis labels, or scientific content."
    )
    cache: dict[str, str] = {}
    was_training = model.training
    model.eval()

    unique_rels = [r for r in df["image_path"].unique()
                   if str(data_dir / r) not in cache]

    for i in tqdm(range(0, len(unique_rels), batch_size), desc="Generating captions"):
        batch_rels = unique_rels[i:i + batch_size]
        batch_images = [_load_image(data_dir / r, img_size, legacy_resize)
                        for r in batch_rels]
        batch_prompts = [prompt] * len(batch_rels)

        enc = processor(text=batch_prompts, images=batch_images,
                        return_tensors="pt", padding=True)
        for k, v in list(enc.items()):
            if torch.is_tensor(v):
                enc[k] = v.to(device, non_blocking=True)

        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )

        input_len = enc["input_ids"].shape[1]
        for j, rel in enumerate(batch_rels):
            caption = processor.tokenizer.decode(
                out[j][input_len:], skip_special_tokens=True
            ).strip()
            cache[str(data_dir / rel)] = caption

    if was_training:
        model.train()
    return cache


# ──────────────────────────────────────────────────────────────
# Branch 8: training-time augmentation
# ──────────────────────────────────────────────────────────────


def build_train_transform(img_size: int):
    if not HAS_TORCHVISION:
        print(
            "Warning: torchvision not found; --augment ignored. "
            "Install with: pip install torchvision"
        )
        return None
    return T.Compose(
        [
            T.RandomResizedCrop((img_size, img_size),
                                scale=(0.8, 1.0), antialias=True),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
    )


# ──────────────────────────────────────────────────────────────
# Core forward pass
# ──────────────────────────────────────────────────────────────


def option_letter_logprobs(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    row: pd.Series,
    device: torch.device,
    letter_token_map: dict[str, int],
    need_grad: bool,
    answer_prefix: str = "Answer:",
    prompt_metadata: str = "none",
    caption: str | None = None,
    use_cot: bool = False,
) -> torch.Tensor:
    """
    Return raw logit scores for each option letter. Shape: (K,).

    These are the unmodified next-token logits at the "Answer:" position, subsetted
    to the K option-letter token ids.  Using raw logits (not log_softmax) means
    F.cross_entropy in the training loop receives proper unnormalized inputs.
    """
    prompt = build_prompt(
        row,
        answer_prefix=answer_prefix,
        prompt_metadata=prompt_metadata,
        caption=caption,
        use_cot=use_cot,
    )
    n_opt = int(row["num_choices"])
    enc = processor(text=[prompt], images=[image],
                    return_tensors="pt", padding=True)
    for k, v in list(enc.items()):
        if torch.is_tensor(v):
            enc[k] = v.to(device, non_blocking=True)

    with torch.amp.autocast(
        "cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")
    ):
        if need_grad:
            out = model(**enc, return_dict=True, use_cache=False)
        else:
            with torch.inference_mode():
                out = model(**enc, return_dict=True, use_cache=False)

    logits = out.logits[0]  # (T, V)
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()

    # Raw next-token logits at the last prefix position — no softmax applied.
    # F.cross_entropy expects unnormalized logits; argmax is invariant to softmax.
    next_token_logits = logits[-1]

    return torch.stack(
        [next_token_logits[letter_token_map[CHOICE_LETTERS[j]]]
            for j in range(n_opt)]
    )  # (K,)


# ──────────────────────────────────────────────────────────────
# Validation accuracy (used for best-checkpoint tracking)
# ──────────────────────────────────────────────────────────────


def _compute_ckpt_accuracy(
    model: torch.nn.Module,
    processor,
    ckpt_df: pd.DataFrame,
    data_dir: Path,
    img_size: int,
    legacy_resize: bool,
    device: torch.device,
    letter_token_map: dict[str, int],
    answer_prefix: str,
    prompt_metadata: str,
    caption_cache: dict[str, str] | None,
) -> float:
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(len(ckpt_df)):
        row = ckpt_df.iloc[i]
        image = _load_image(
            data_dir / row["image_path"], img_size, legacy_resize)
        caption = (
            caption_cache.get(str(data_dir / row["image_path"]))
            if caption_cache is not None
            else None
        )
        scores = option_letter_logprobs(
            model, processor, image, row, device, letter_token_map,
            need_grad=False, answer_prefix=answer_prefix,
            prompt_metadata=prompt_metadata, caption=caption,
        )
        if int(scores.argmax().item()) == int(row["answer"]):
            correct += 1
    if was_training:
        model.train()
    return correct / len(ckpt_df)


def _compute_ckpt_accuracy_cot(
    model: torch.nn.Module,
    processor,
    ckpt_df: pd.DataFrame,
    data_dir: Path,
    img_size: int,
    legacy_resize: bool,
    device: torch.device,
    letter_token_map: dict[str, int],
    answer_prefix: str,
    prompt_metadata: str,
    caption_cache: dict[str, str] | None,
) -> float:
    """CoT checkpoint accuracy: generate reasoning, parse answer letter."""
    was_training = model.training
    model.eval()
    correct = 0
    for i in range(len(ckpt_df)):
        row = ckpt_df.iloc[i]
        image = _load_image(
            data_dir / row["image_path"], img_size, legacy_resize)
        caption = (
            caption_cache.get(str(data_dir / row["image_path"]))
            if caption_cache is not None
            else None
        )
        prompt = build_prompt(
            row, answer_prefix=answer_prefix,
            prompt_metadata=prompt_metadata, caption=caption, use_cot=True,
        )
        enc = processor(text=[prompt], images=[image],
                        return_tensors="pt", padding=True)
        for k, v in list(enc.items()):
            if torch.is_tensor(v):
                enc[k] = v.to(device, non_blocking=True)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
        input_len = enc["input_ids"].shape[1]
        generated = processor.tokenizer.decode(
            out[0][input_len:], skip_special_tokens=True,
        ).strip()
        pred = _parse_answer_from_cot(generated, int(row["num_choices"]))
        if pred == int(row["answer"]):
            correct += 1
    if was_training:
        model.train()
    return correct / len(ckpt_df)


# ──────────────────────────────────────────────────────────────
# Val-set evaluation with per-category accuracy analysis
# ──────────────────────────────────────────────────────────────


def _eval_val_and_analyze(
    model: torch.nn.Module,
    processor,
    val_df: pd.DataFrame,
    data_dir: Path,
    img_size: int,
    legacy_resize: bool,
    device: torch.device,
    letter_token_map: dict[str, int],
    answer_prefix: str,
    prompt_metadata: str,
    caption_cache: dict[str, str] | None,
    use_cot: bool,
) -> dict:
    """Evaluate on val.csv and print per-category accuracy breakdown."""
    from collections import Counter

    was_training = model.training
    model.eval()

    results: list[dict] = []
    for i in tqdm(range(len(val_df)), desc="Evaluating val"):
        row = val_df.iloc[i]
        image = _load_image(
            data_dir / row["image_path"], img_size, legacy_resize)
        caption = (
            caption_cache.get(str(data_dir / row["image_path"]))
            if caption_cache is not None
            else None
        )
        y_true = int(row["answer"])
        n_opt = int(row["num_choices"])

        if use_cot:
            prompt = build_prompt(
                row, answer_prefix=answer_prefix,
                prompt_metadata=prompt_metadata, caption=caption, use_cot=True,
            )
            enc = processor(text=[prompt], images=[image],
                            return_tensors="pt", padding=True)
            for k, v in list(enc.items()):
                if torch.is_tensor(v):
                    enc[k] = v.to(device, non_blocking=True)
            with torch.inference_mode():
                out_ids = model.generate(
                    **enc, max_new_tokens=256, do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
            input_len = enc["input_ids"].shape[1]
            generated = processor.tokenizer.decode(
                out_ids[0][input_len:], skip_special_tokens=True,
            ).strip()
            y_pred = _parse_answer_from_cot(generated, n_opt)
        else:
            scores = option_letter_logprobs(
                model, processor, image, row, device, letter_token_map,
                need_grad=False, answer_prefix=answer_prefix,
                prompt_metadata=prompt_metadata, caption=caption,
            )
            y_pred = int(scores.argmax().item())

        results.append({
            "correct": 1 if y_pred == y_true else 0,
            "pred": y_pred,
            "answer": y_true,
            "num_choices": n_opt,
            "grade": str(row.get("grade", "")),
            "subject": str(row.get("subject", "")),
            "task": str(row.get("task", "")),
            "category": str(row.get("category", "")),
        })

    if was_training:
        model.train()

    total = len(results)
    overall_acc = sum(r["correct"] for r in results) / total

    def _group_by(key):
        g: dict = {}
        for r in results:
            g.setdefault(r[key], []).append(r)
        return g

    def _acc_by(groups):
        return {k: (len(v), sum(r["correct"] for r in v) / len(v)) for k, v in groups.items()}

    def _print_table(title, stats, sort_by="total"):
        print(f"\n{'─'*60}")
        print(f"  {title}")
        print(f"{'─'*60}")
        rev = sort_by == "total"
        items = sorted(stats.items(), key=lambda x: -
                       x[1][0] if rev else x[1][1])
        print(f"  {'Key':34s}  {'N':>5s}  {'Acc':>7s}  {'Bar'}")
        for k, (n, acc) in items:
            bar = "█" * max(1, int(acc * 30))
            print(f"  {str(k):34s}  {n:5d}  {acc:6.2%}  {bar}")

    # ── print report ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  VAL SET EVALUATION  (n={total})")
    print(f"  Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"{'='*60}")

    _print_table("Accuracy by ground-truth answer",
                 _acc_by(_group_by("answer")))
    _print_table("Accuracy by num_choices", _acc_by(_group_by("num_choices")))
    _print_table("Accuracy by grade", _acc_by(_group_by("grade")))
    _print_table("Accuracy by subject", _acc_by(_group_by("subject")))
    _print_table("Accuracy by task", _acc_by(_group_by("task")))

    # Category: worst first to surface hard topics
    _print_table("Accuracy by category (worst first)",
                 _acc_by(_group_by("category")), sort_by="acc")

    # ── confusion matrix: prediction distribution per answer ────
    print(f"\n{'─'*60}")
    print(f"  Confusion matrix (rows=true, cols=pred)")
    print(f"{'─'*60}")
    # Build matrix
    answers = sorted(set(r["answer"] for r in results))
    cm: dict[int, Counter] = {a: Counter() for a in answers}
    for r in results:
        cm[r["answer"]][r["pred"]] += 1
    header = "ans\\pred " + "".join(f"  {a}   " for a in answers)
    print(f"  {header}")
    for a in answers:
        row_str = f"    {a}    " + "".join(
            f" {cm[a].get(p, 0):4d} " for p in answers
        )
        acc = sum(1 for r in results if r["answer"] == a and r["correct"])
        total_a = sum(1 for r in results if r["answer"] == a)
        print(f"  {row_str}   (acc={acc/total_a:.2%})")

    return {"overall_acc": overall_acc, "total": total}


def _resolve_val_csv(val_csv: str, data_dir: Path) -> Path | None:
    """Resolve the val.csv path from user argument."""
    if not val_csv or not val_csv.strip():
        return None
    if val_csv.strip().lower() == "auto":
        candidate = data_dir / "val.csv"
        if candidate.exists():
            return candidate
        print(
            f"[val-eval] 'auto' mode: {candidate} not found, skipping val evaluation.")
        return None
    p = Path(val_csv.strip())
    if p.exists():
        return p
    # Try relative to data_dir
    p2 = data_dir / val_csv.strip()
    if p2.exists():
        return p2
    print(
        f"[val-eval] val.csv not found at '{val_csv}' or '{p2}', skipping val evaluation.")
    return None


# ──────────────────────────────────────────────────────────────
# LR scheduler factory
# ──────────────────────────────────────────────────────────────


def _build_scheduler(
    opt: torch.optim.Optimizer,
    lr_scheduler: str,
    learning_rate: float,
    total_optimizer_steps: int,
    warmup_ratio: float,
):
    if lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=learning_rate,
            total_steps=total_optimizer_steps,
            pct_start=warmup_ratio,
            anneal_strategy="cos",
            div_factor=10.0,
            final_div_factor=100.0,
        )
    if lr_scheduler == "linear":
        warmup_steps = max(1, int(total_optimizer_steps * warmup_ratio))

        def _lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / warmup_steps
            remaining = total_optimizer_steps - step
            total_decay = total_optimizer_steps - warmup_steps
            return max(0.0, remaining / max(1, total_decay))

        return torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)
    return None  # "none"


# ──────────────────────────────────────────────────────────────
# Training loop
# ──────────────────────────────────────────────────────────────


def run_train(
    model: torch.nn.Module,
    processor,
    train_df: pd.DataFrame,
    ckpt_df: pd.DataFrame,
    data_dir: Path,
    img_size: int,
    legacy_resize: bool,
    output_dir: str,
    learning_rate: float,
    num_epochs: int,
    grad_accum_steps: int,
    train_batch_size: int,
    logging_steps: int,
    seed: int,
    letter_token_map: dict[str, int],
    answer_prefix: str = "Answer:",
    prompt_metadata: str = "none",
    caption_cache: dict[str, str] | None = None,
    augment: bool = False,
    grad_ckpt: bool = True,
    lr_scheduler: str = "cosine",
    warmup_ratio: float = 0.05,
    use_cot: bool = False,
    use_margin_loss: bool = False,
    margin: float = 0.5,
    save_epoch_ckpts: bool = False,
) -> None:
    device = next(model.parameters()).device
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.train()

    # Branch 9: gradient checkpointing (enabled by default)
    if grad_ckpt:
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except (AttributeError, ValueError):
            pass

    # Branch 8: augmentation transform
    train_transform = None
    if augment:
        if not legacy_resize:
            print(
                "Warning: --augment with processor-native resize applies transforms "
                "before passing to the processor (random crop uses img_size as target)."
            )
        train_transform = build_train_transform(img_size)

    trainable = [p for p in model.parameters() if p.requires_grad]
    trainable_names = {n for n, p in model.named_parameters()
                       if p.requires_grad}
    opt = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)

    n = len(train_df)
    steps_per_epoch = math.ceil(n / (train_batch_size * grad_accum_steps))
    total_optimizer_steps = steps_per_epoch * num_epochs

    scheduler = _build_scheduler(
        opt, lr_scheduler, learning_rate, total_optimizer_steps, warmup_ratio
    )
    if scheduler is not None:
        print(
            f"LR scheduler: {lr_scheduler}, warmup_ratio={warmup_ratio}, "
            f"total_optimizer_steps={total_optimizer_steps}"
        )

    eff_batch = train_batch_size * grad_accum_steps
    print(
        f"train_batch_size={train_batch_size}, grad_accum_steps={grad_accum_steps} "
        f"=> effective batch ~{eff_batch} samples per optimizer step."
    )
    mode_desc = "CoT" if use_cot else (
        "margin-loss" if use_margin_loss else "letter-token CE")
    print(f"Training mode: {mode_desc}")

    # Best-checkpoint tracking (in-memory; no extra disk I/O during training)
    best_ckpt_acc: float = -1.0
    best_state: dict | None = None
    _first_ckpt_eval = True

    total_steps = 0
    losses: list[float] = []

    for epoch in range(num_epochs):
        order = list(range(n))
        random.shuffle(order)
        num_micro = math.ceil(n / train_batch_size)
        pbar = tqdm(
            range(0, n, train_batch_size),
            total=num_micro,
            desc=f"Epoch {epoch+1}/{num_epochs} (letter-token CE, micro-batches)",
        )
        opt.zero_grad(set_to_none=True)
        accum_micro = 0

        for start in pbar:
            mb = order[start: start + train_batch_size]
            b_eff = len(mb)
            for idx in mb:
                row = train_df.iloc[idx]
                image = _load_image(
                    data_dir / row["image_path"], img_size, legacy_resize)
                if train_transform is not None:
                    image = train_transform(image)
                caption = (
                    caption_cache.get(str(data_dir / row["image_path"]))
                    if caption_cache is not None
                    else None
                )
                y = int(row["answer"])

                # ── Branch 10: CoT training (generative LM loss) ──
                if use_cot:
                    prompt = build_prompt(
                        row, answer_prefix=answer_prefix,
                        prompt_metadata=prompt_metadata, caption=caption, use_cot=True,
                    )
                    target = build_cot_target(row)
                    full_text = prompt + target
                    enc = processor(
                        text=[full_text], images=[image],
                        return_tensors="pt", padding=True,
                    )
                    for k, v in list(enc.items()):
                        if torch.is_tensor(v):
                            enc[k] = v.to(device, non_blocking=True)
                    # Mask prompt tokens so loss is only on target
                    # Text-only tokenization to compute prompt boundary (fast, no image re-processing)
                    target_ids = processor.tokenizer(
                        target, add_special_tokens=False).input_ids
                    prompt_len = enc["input_ids"].shape[1] - len(target_ids)
                    labels = enc["input_ids"].clone()
                    labels[0, :prompt_len] = -100

                    with torch.amp.autocast(
                        "cuda" if device.type == "cuda" else "cpu",
                        enabled=(device.type == "cuda"),
                    ):
                        out = model(**enc, labels=labels, return_dict=True)
                    loss = out.loss / (b_eff * grad_accum_steps)
                else:
                    scores = option_letter_logprobs(
                        model, processor, image, row, device, letter_token_map,
                        need_grad=True, answer_prefix=answer_prefix,
                        prompt_metadata=prompt_metadata, caption=caption,
                    )
                    # ── Branch 11: margin loss ──
                    if use_margin_loss:
                        loss = margin_loss(scores, y, margin)
                    else:
                        loss = F.cross_entropy(
                            scores.unsqueeze(0),
                            torch.tensor([y], device=device, dtype=torch.long),
                        )
                    loss = loss / (b_eff * grad_accum_steps)

                loss.backward()
                total_steps += 1
                losses.append(float(loss.item()))

            accum_micro += 1
            if accum_micro >= grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                if scheduler is not None:
                    scheduler.step()
                opt.zero_grad(set_to_none=True)
                accum_micro = 0

            if total_steps % logging_steps == 0 and losses:
                ma_val = float(
                    np.mean(losses[-min(logging_steps, len(losses)):]))
                last_val = float(losses[-1])
                pbar.set_postfix(last_loss=last_val, ma=ma_val)
                print(
                    f"  step {total_steps:5d} | last_loss={last_val:.4f} | ma_loss={ma_val:.4f}"
                )

        if accum_micro > 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            opt.zero_grad(set_to_none=True)

        # ── best-checkpoint evaluation ─────────────────────────
        if _first_ckpt_eval:
            print(
                f"[ckpt-eval] Evaluating on {len(ckpt_df)} samples held out from train.csv "
                f"for checkpoint selection. This is NOT the competition val set."
            )
            _first_ckpt_eval = False
        if use_cot:
            ckpt_acc = _compute_ckpt_accuracy_cot(
                model, processor, ckpt_df, data_dir, img_size, legacy_resize,
                device, letter_token_map, answer_prefix, prompt_metadata, caption_cache,
            )
        else:
            ckpt_acc = _compute_ckpt_accuracy(
                model, processor, ckpt_df, data_dir, img_size, legacy_resize,
                device, letter_token_map, answer_prefix, prompt_metadata, caption_cache,
            )
        print(
            f"Epoch {epoch+1} ckpt-split accuracy: {ckpt_acc:.4f} "
            f"(best so far: {max(best_ckpt_acc, ckpt_acc):.4f})"
        )
        if ckpt_acc > best_ckpt_acc:
            best_ckpt_acc = ckpt_acc
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if k in trainable_names
            }
            print(
                f"  → new best checkpoint saved (ckpt_acc={best_ckpt_acc:.4f})")

        # ── save per-epoch checkpoint for weight averaging ────
        if save_epoch_ckpts:
            saved_epoch = Path(output_dir) / f"adapter_epoch_{epoch+1}"
            saved_epoch.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(saved_epoch))
            processor.save_pretrained(str(saved_epoch))
            print(f"  Saved epoch-{epoch+1} adapter to {saved_epoch}")

        model.train()

    # ── save last-epoch checkpoint ─────────────────────────────
    saved_last = Path(output_dir) / "adapter_last"
    saved_last.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(saved_last))
    processor.save_pretrained(str(saved_last))
    print(f"Saved last-epoch adapter to {saved_last}")

    # ── restore best weights and save ─────────────────────────
    if best_state is not None:
        print(f"Restoring best checkpoint (ckpt_acc={best_ckpt_acc:.4f}).")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in best_state:
                    param.copy_(best_state[name].to(param.device))

    saved_best = Path(output_dir) / "adapter_best"
    saved_best.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(saved_best))
    processor.save_pretrained(str(saved_best))
    print(f"Saved best adapter to {saved_best}")


# ──────────────────────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────────────────────


@torch.inference_mode()
def predict_submission(
    model: torch.nn.Module,
    processor,
    test_df: pd.DataFrame,
    data_dir: Path,
    img_size: int,
    legacy_resize: bool,
    output_file: Path,
    letter_token_map: dict[str, int],
    answer_prefix: str = "Answer:",
    prompt_metadata: str = "none",
    caption_cache: dict[str, str] | None = None,
    use_cot: bool = False,
) -> None:
    device = next(model.parameters()).device
    model.eval()
    preds: list[int] = []
    ids: list[str] = []

    desc = "Predicting test (CoT generation)" if use_cot else "Predicting test (letter-token argmax)"
    for i in tqdm(range(len(test_df)), desc=desc):
        row = test_df.iloc[i]
        image = _load_image(
            data_dir / row["image_path"], img_size, legacy_resize)
        caption = (
            caption_cache.get(str(data_dir / row["image_path"]))
            if caption_cache is not None
            else None
        )
        if use_cot:
            prompt = build_prompt(
                row, answer_prefix=answer_prefix,
                prompt_metadata=prompt_metadata, caption=caption, use_cot=True,
            )
            enc = processor(text=[prompt], images=[image],
                            return_tensors="pt", padding=True)
            for k, v in list(enc.items()):
                if torch.is_tensor(v):
                    enc[k] = v.to(device, non_blocking=True)
            out_ids = model.generate(
                **enc,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            input_len = enc["input_ids"].shape[1]
            generated = processor.tokenizer.decode(
                out_ids[0][input_len:], skip_special_tokens=True,
            ).strip()
            preds.append(_parse_answer_from_cot(
                generated, int(row["num_choices"])))
        else:
            scores = option_letter_logprobs(
                model, processor, image, row, device, letter_token_map,
                need_grad=False, answer_prefix=answer_prefix,
                prompt_metadata=prompt_metadata, caption=caption,
            )
            preds.append(int(scores.argmax().item()))
        ids.append(row["id"])

    out = pd.DataFrame({"id": ids, "answer": preds})
    out.to_csv(output_file, index=False)
    print(f"Saved submission to: {output_file}")
    print(out.head())


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────


def main() -> None:
    p = argparse.ArgumentParser(
        description="QLoRA with letter-token-only option likelihood training/inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ── data / paths ──────────────────────────────────────────
    p.add_argument("--data_dir", type=str, required=True,
                   help="Dataset directory containing train.csv and test.csv.")
    p.add_argument("--model_id", type=str, required=True,
                   help="Base model Hugging Face ID or local model directory.")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory where adapters and run artifacts are saved.")
    p.add_argument("--submission_file", type=str, required=True,
                   help="Output CSV path for test-set predictions.")

    # ── image ─────────────────────────────────────────────────
    p.add_argument(
        "--img_size",
        type=int,
        default=224,
        help=(
            "Image resolution. With --legacy_resize: manual PIL resize to img_size×img_size. "
            "Without (default): sets processor's longest_edge — "
            "384 = SmolVLM native with tiling; 224 = smaller/faster."
        ),
    )
    p.add_argument(
        "--legacy_resize",
        action="store_true",
        help=(
            "Use old manual PIL resize before the processor. "
            "Default (off): processor handles all resizing and tiling, "
            "which preserves SmolVLM's tiling logic."
        ),
    )

    # ── LoRA / DoRA ───────────────────────────────────────────
    p.add_argument(
        "--resume_adapter_dir",
        type=str,
        default="",
        help=(
            "Warm-start from a previously saved PEFT adapter directory (e.g. "
            "/path/to/adapter_best). If set, LoRA/DoRA settings "
            "are loaded from that directory and related CLI flags are ignored. "
            "Note: optimizer/scheduler state is NOT restored."
        ),
    )

    p.add_argument(
        "--lora_targets",
        type=str,
        default="auto",
        choices=["auto", "attn", "mlp", "all"],
        help=(
            "'attn': q/k/v/o_proj (Branch 1 baseline). "
            "'mlp': gate/up/down_proj (Branch 1 main; recommend --lora_r 6). "
            "'all': attn+MLP (Branch 4; recommend --lora_r 4). "
            "'auto': all found modules."
        ),
    )
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--use_dora",
        action="store_true",
        help="Branch 2: DoRA (magnitude+direction decomposition) instead of standard LoRA.",
    )

    # ── prompt ────────────────────────────────────────────────
    p.add_argument(
        "--answer_prefix",
        type=str,
        default="Answer:",
        help='Branch 5: e.g. "The correct answer is:" vs default "Answer:"',
    )
    p.add_argument(
        "--prompt_metadata",
        type=str,
        default="none",
        choices=["none", "subject", "grade", "topic"],
        help="Branch 6: inject one metadata field into the prompt.",
    )
    p.add_argument(
        "--use_caption",
        action="store_true",
        help=(
            "Branch 3: auto-generate an image caption with the base model before "
            "training/inference and prepend it to each prompt."
        ),
    )

    # ── augmentation / memory ─────────────────────────────────
    p.add_argument(
        "--augment",
        action="store_true",
        help="Branch 8: random crop / rotation / colour jitter during training.",
    )
    p.add_argument(
        "--no_grad_ckpt",
        action="store_true",
        help="Branch 9: disable gradient checkpointing (enabled by default).",
    )

    # ── Branch 10 / 11 ────────────────────────────────────────
    p.add_argument(
        "--use_cot",
        action="store_true",
        help="Branch 10: Chain-of-Thought training using solution field for reasoning.",
    )
    p.add_argument(
        "--use_margin_loss",
        action="store_true",
        help="Branch 11: margin-based ranking loss instead of standard cross-entropy.",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.5,
        help="Margin value for Branch 11 margin loss.",
    )
    p.add_argument(
        "--save_epoch_ckpts",
        action="store_true",
        help="Save per-epoch adapter checkpoints for weight averaging.",
    )

    # ── LR scheduler ─────────────────────────────────────────
    p.add_argument(
        "--lr_scheduler",
        type=str,
        default="cosine",
        choices=["none", "cosine", "linear"],
        help=(
            "LR schedule. 'cosine': OneCycleLR with cosine annealing (recommended). "
            "'linear': linear warmup then linear decay. 'none': flat LR."
        ),
    )
    p.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.05,
        help="Fraction of total optimizer steps used for LR warmup.",
    )

    # ── training ──────────────────────────────────────────────
    p.add_argument("--max_trainable_params", type=int,
                   default=MAX_TRAINABLE_PARAMETERS)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Samples per micro-batch; effective batch ≈ train_batch_size × grad_accum_steps.",
    )
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--train_limit", type=int, default=0)
    p.add_argument("--val_limit", type=int, default=0,
                   help="If >0, only first N test rows.")
    p.add_argument(
        "--val_csv",
        type=str,
        default="",
        help="Path to val.csv for final per-category accuracy analysis. "
             'Use "auto" to auto-detect in data_dir. '
             "Skip if empty (default).",
    )
    p.add_argument(
        "--ckpt_split",
        type=float,
        default=0.1,
        help=(
            "Fraction of train.csv held out internally for best-checkpoint selection. "
            "Distinct from the competition val set."
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.train_batch_size < 1:
        raise ValueError("--train_batch_size must be >= 1")
    if not (0.0 < args.ckpt_split < 1.0):
        raise ValueError("--ckpt_split must be in (0, 1)")

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_adapter_path = resolve_resume_adapter_dir(
        args.resume_adapter_dir, output_dir)

    print(f"Using data directory: {data_dir}")
    if resume_adapter_path is not None:
        try:
            processor = AutoProcessor.from_pretrained(str(resume_adapter_path))
            print(
                f"Loaded processor from resume adapter dir: {resume_adapter_path}")
        except Exception as e:
            print(
                f"Warning: could not load processor from '{resume_adapter_path}': {e}. "
                f"Falling back to model_id processor: {args.model_id}"
            )
            processor = AutoProcessor.from_pretrained(args.model_id)
    else:
        processor = AutoProcessor.from_pretrained(args.model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if not args.legacy_resize:
        configure_processor_image_size(processor, args.img_size)

    letter_token_map = build_letter_token_map(processor)

    train_df = pd.read_csv(data_dir / "train.csv")
    test_df = pd.read_csv(data_dir / "test.csv")
    train_df["choices"] = train_df["choices"].apply(json.loads)
    test_df["choices"] = test_df["choices"].apply(json.loads)
    if args.train_limit > 0:
        train_df = train_df.iloc[: args.train_limit].copy()
    if args.val_limit > 0:
        test_df = test_df.iloc[: args.val_limit].copy()

    # ── ckpt split for best-checkpoint tracking ───────────────
    # Distinct from the competition val set; carved out of train.csv only.
    rng_split = np.random.RandomState(args.seed)
    ckpt_n = max(1, int(len(train_df) * args.ckpt_split))
    ckpt_idx = rng_split.choice(len(train_df), ckpt_n, replace=False)
    fit_idx = np.setdiff1d(np.arange(len(train_df)), ckpt_idx)
    ckpt_df = train_df.iloc[ckpt_idx].reset_index(drop=True)
    fit_df = train_df.iloc[fit_idx].reset_index(drop=True)
    print(
        f"Ckpt split: {len(fit_df)} fit / {len(ckpt_df)} ckpt "
        f"(internal checkpoint selection; NOT the competition val set)."
    )

    model, _ = maybe_build_qlora_model(args.model_id)
    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))

    # Branch 3: generate captions with base model before LoRA wrapping
    caption_cache: dict[str, str] | None = None
    if args.use_caption:
        device = next(model.parameters()).device
        print("Branch 3: building caption cache for train set…")
        caption_cache = build_caption_cache(
            model, processor, train_df, data_dir, args.img_size, args.legacy_resize, device,
            batch_size=args.train_batch_size,
        )
        print("Branch 3: building caption cache for test set…")
        caption_cache.update(
            build_caption_cache(
                model, processor, test_df, data_dir, args.img_size, args.legacy_resize, device,
                batch_size=args.train_batch_size,
            )
        )

    if resume_adapter_path is not None:
        print(
            f"Resuming training from adapter checkpoint: {resume_adapter_path}")
        if any(
            [
                args.use_dora,
                args.lora_targets != "auto",
                args.lora_r != 8,
                args.lora_alpha != 16,
                abs(args.lora_dropout - 0.05) > 1e-9,
            ]
        ):
            print(
                "Note: --resume_adapter_dir is set; LoRA/DoRA-related CLI flags are ignored "
                "and settings are loaded from adapter_config.json."
            )
        model = PeftModel.from_pretrained(
            model, str(resume_adapter_path), is_trainable=True
        )
    else:
        target_modules = pick_lora_target_modules(
            model, mode=args.lora_targets)
        print(f"LoRA target modules ({args.lora_targets}): {target_modules}")

        lora_kwargs: dict = dict(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        if args.use_dora:
            lora_kwargs["use_dora"] = True
            print("Branch 2: DoRA enabled.")

        model = get_peft_model(model, LoraConfig(**lora_kwargs))
    model.print_trainable_parameters()
    require_trainable_parameters_within_limit(
        model, max_params=args.max_trainable_params)

    # ── print all hyperparameters ────────────────────────────────
    print("=" * 60)
    print("Training hyperparameters:")
    print(f"  model_id: {args.model_id}")
    print(f"  resume_adapter_dir: {args.resume_adapter_dir}")
    print(f"  img_size: {args.img_size}")
    print(f"  legacy_resize: {args.legacy_resize}")
    print(f"  lora_targets: {args.lora_targets}")
    print(f"  lora_r: {args.lora_r}")
    print(f"  lora_alpha: {args.lora_alpha}")
    print(f"  lora_dropout: {args.lora_dropout}")
    print(f"  use_dora: {args.use_dora}")
    print(f"  answer_prefix: {args.answer_prefix}")
    print(f"  prompt_metadata: {args.prompt_metadata}")
    print(f"  use_caption: {args.use_caption}")
    print(f"  augment: {args.augment}")
    print(f"  grad_ckpt: {not args.no_grad_ckpt}")
    print(f"  use_cot: {args.use_cot}")
    print(f"  use_margin_loss: {args.use_margin_loss}")
    print(f"  margin: {args.margin}")
    print(f"  save_epoch_ckpts: {args.save_epoch_ckpts}")
    print(f"  lr_scheduler: {args.lr_scheduler}")
    print(f"  warmup_ratio: {args.warmup_ratio}")
    print(f"  learning_rate: {args.learning_rate}")
    print(f"  num_epochs: {args.num_epochs}")
    print(f"  train_batch_size: {args.train_batch_size}")
    print(f"  grad_accum_steps: {args.grad_accum_steps}")
    print(
        f"  effective_batch_size: {args.train_batch_size * args.grad_accum_steps}")
    print(f"  max_trainable_params: {args.max_trainable_params}")
    print(f"  seed: {args.seed}")
    print(f"  num_fit_samples: {len(fit_df)}")
    print(f"  num_ckpt_samples: {len(ckpt_df)}")
    print(f"  logging_steps: {args.logging_steps}")
    print("=" * 60)

    run_train(
        model=model,
        processor=processor,
        train_df=fit_df,
        ckpt_df=ckpt_df,
        data_dir=data_dir,
        img_size=args.img_size,
        legacy_resize=args.legacy_resize,
        output_dir=str(output_dir),
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        grad_accum_steps=args.grad_accum_steps,
        train_batch_size=args.train_batch_size,
        logging_steps=args.logging_steps,
        seed=args.seed,
        letter_token_map=letter_token_map,
        answer_prefix=args.answer_prefix,
        prompt_metadata=args.prompt_metadata,
        caption_cache=caption_cache,
        augment=args.augment,
        grad_ckpt=not args.no_grad_ckpt,
        lr_scheduler=args.lr_scheduler,
        warmup_ratio=args.warmup_ratio,
        use_cot=args.use_cot,
        use_margin_loss=args.use_margin_loss,
        margin=args.margin,
        save_epoch_ckpts=args.save_epoch_ckpts,
    )

    # ── val-set evaluation before generating submission ─────────
    val_path = _resolve_val_csv(args.val_csv, data_dir)
    if val_path is not None:
        print(f"\n[val-eval] Evaluating on: {val_path}")
        val_df = pd.read_csv(val_path)
        val_df["choices"] = val_df["choices"].apply(json.loads)
        device = next(model.parameters()).device
        _eval_val_and_analyze(
            model=model,
            processor=processor,
            val_df=val_df,
            data_dir=data_dir,
            img_size=args.img_size,
            legacy_resize=args.legacy_resize,
            device=device,
            letter_token_map=letter_token_map,
            answer_prefix=args.answer_prefix,
            prompt_metadata=args.prompt_metadata,
            caption_cache=caption_cache,
            use_cot=args.use_cot,
        )

    predict_submission(
        model=model,
        processor=processor,
        test_df=test_df,
        data_dir=data_dir,
        img_size=args.img_size,
        legacy_resize=args.legacy_resize,
        output_file=Path(args.submission_file),
        letter_token_map=letter_token_map,
        answer_prefix=args.answer_prefix,
        prompt_metadata=args.prompt_metadata,
        caption_cache=caption_cache,
        use_cot=args.use_cot,
    )


if __name__ == "__main__":
    main()
