"""
Standalone inference script: loads a QLoRA adapter checkpoint and generates a
submission CSV for the test set.

Usage:
  python infer_qlora.py --adapter_dir /path/to/adapter_best
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
CHOICE_LETTERS = "ABCDEFGHIJ"


# ── utilities (mirrored from training script) ────────────────────

def resolve_data_dir(user_data_dir: str) -> Path:
    candidate = Path(user_data_dir)
    if (candidate / "test.csv").exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find test.csv under data_dir: {candidate}")


def _load_image(path: Path, img_size: int, legacy_resize: bool) -> Image.Image:
    image = Image.open(path).convert("RGB")
    if legacy_resize:
        image = image.resize((img_size, img_size), Image.BICUBIC)
    return image


def configure_processor_image_size(processor, img_size: int) -> None:
    try:
        processor.image_processor.size = {"longest_edge": img_size}
        print(f"Processor image size set to longest_edge={img_size}.")
    except AttributeError:
        print("Warning: could not set processor image size programmatically.")


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
    if caption:
        prompt += f"Image description: {caption}\n\n"
    if context_str:
        prompt += f"Context:\n{context_str}\n\n"
    if prompt_metadata != "none":
        meta_val = row.get(prompt_metadata, "")
        if pd.notna(meta_val) and str(meta_val).strip():
            prompt += f"{prompt_metadata.capitalize()}: {str(meta_val).strip()}\n"
    prompt += f"Question: {row['question']}\n"
    prompt += f"Choices:\n{choices_str}\n"
    if use_cot:
        prompt += "Let's think step by step.\n"
    else:
        prompt += answer_prefix
    return prompt


def _parse_answer_from_cot(generated_text: str, n_opt: int) -> int:
    """Parse the predicted answer letter from a CoT generation."""
    m = re.search(r"answer\s+is\s*:?\s*([A-H])", generated_text, re.IGNORECASE)
    if m:
        letter = m.group(1).upper()
        idx = CHOICE_LETTERS.index(letter)
        if idx < n_opt:
            return idx
    for ch in reversed(generated_text):
        if ch in CHOICE_LETTERS[:n_opt]:
            return CHOICE_LETTERS.index(ch)
    return 0


def build_letter_token_map(processor) -> dict[str, int]:
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


# ── forward pass ─────────────────────────────────────────────────

def option_letter_logprobs(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    row: pd.Series,
    device: torch.device,
    letter_token_map: dict[str, int],
    answer_prefix: str = "Answer:",
    prompt_metadata: str = "none",
    caption: str | None = None,
    use_cot: bool = False,
) -> torch.Tensor:
    prompt = build_prompt(row, answer_prefix=answer_prefix,
                          prompt_metadata=prompt_metadata,
                          caption=caption, use_cot=use_cot)
    n_opt = int(row["num_choices"])
    enc = processor(text=[prompt], images=[image],
                    return_tensors="pt", padding=True)
    for k, v in list(enc.items()):
        if torch.is_tensor(v):
            enc[k] = v.to(device, non_blocking=True)

    with torch.amp.autocast(
        "cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")
    ):
        with torch.inference_mode():
            out = model(**enc, return_dict=True, use_cache=False)

    logits = out.logits[0]
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    next_token_logits = logits[-1]
    return torch.stack(
        [next_token_logits[letter_token_map[CHOICE_LETTERS[j]]]
            for j in range(n_opt)]
    )


# ── inference ────────────────────────────────────────────────────

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

    desc = "Predicting test (CoT)" if use_cot else "Predicting test"
    for i in tqdm(range(len(test_df)), desc=desc):
        row = test_df.iloc[i]
        image = _load_image(data_dir / row["image_path"], img_size, legacy_resize)
        caption = (
            caption_cache.get(str(data_dir / row["image_path"]))
            if caption_cache is not None
            else None
        )
        if use_cot:
            prompt = build_prompt(row, answer_prefix=answer_prefix,
                                  prompt_metadata=prompt_metadata,
                                  caption=caption, use_cot=True)
            enc = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
            for k, v in list(enc.items()):
                if torch.is_tensor(v):
                    enc[k] = v.to(device, non_blocking=True)
            out_ids = model.generate(
                **enc, max_new_tokens=256, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            generated = processor.tokenizer.decode(
                out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True,
            ).strip()
            preds.append(_parse_answer_from_cot(generated, int(row["num_choices"])))
        else:
            scores = option_letter_logprobs(
                model, processor, image, row, device, letter_token_map,
                answer_prefix=answer_prefix, prompt_metadata=prompt_metadata,
                caption=caption,
            )
            preds.append(int(scores.argmax().item()))
        ids.append(row["id"])

    out = pd.DataFrame({"id": ids, "answer": preds})
    out.to_csv(output_file, index=False)
    print(f"Saved submission to: {output_file}")
    print(out.head())


# ── model loading ────────────────────────────────────────────


def build_base_model(model_id: str):
    """Load the base SmolVLM model with QLoRA 4-bit quantization."""
    if not torch.cuda.is_available():
        print("CUDA not available. Loading in float32 without quantization.")
        return AutoModelForVision2Seq.from_pretrained(
            model_id, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        )
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    return AutoModelForVision2Seq.from_pretrained(
        model_id, quantization_config=bnb, device_map="auto", low_cpu_mem_usage=True,
    )


# ── caption cache (for --use_caption) ─────────────────────────


def _generate_single_caption(model, processor, image, device) -> str:
    prompt = (
        "<image>\nBriefly describe what is shown in this image, "
        "including any text, axis labels, or scientific content."
    )
    enc = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    for k, v in list(enc.items()):
        if torch.is_tensor(v):
            enc[k] = v.to(device, non_blocking=True)
    with torch.inference_mode():
        out = model.generate(
            **enc, max_new_tokens=64, do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
        )
    return processor.tokenizer.decode(
        out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True,
    ).strip()


def build_caption_cache(model, processor, df, data_dir, img_size, legacy_resize, device) -> dict[str, str]:
    cache: dict[str, str] = {}
    was_training = model.training
    model.eval()
    for img_rel in tqdm(df["image_path"].unique(), desc="Generating captions"):
        img_path = str(data_dir / img_rel)
        if img_path not in cache:
            image = _load_image(data_dir / img_rel, img_size, legacy_resize)
            cache[img_path] = _generate_single_caption(model, processor, image, device)
    if was_training:
        model.train()
    return cache


# ── entry point ──────────────────────────────────────────────────


def _resolve_val_csv(val_csv: str, data_dir: Path) -> Path | None:
    """Resolve the val.csv path from user argument."""
    if not val_csv or not val_csv.strip():
        return None
    if val_csv.strip().lower() == "auto":
        candidate = data_dir / "val.csv"
        if candidate.exists():
            return candidate
        print(f"[val-eval] 'auto' mode: {candidate} not found, skipping val evaluation.")
        return None
    p = Path(val_csv.strip())
    if p.exists():
        return p
    p2 = data_dir / val_csv.strip()
    if p2.exists():
        return p2
    print(f"[val-eval] val.csv not found at '{val_csv}' or '{p2}', skipping val evaluation.")
    return None


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
    caption_cache: dict[str, str] | None = None,
    use_cot: bool = False,
) -> dict:
    """Evaluate on val.csv and print per-category accuracy breakdown."""
    from collections import Counter

    was_training = model.training
    model.eval()

    results: list[dict] = []
    for i in tqdm(range(len(val_df)), desc="Evaluating val"):
        row = val_df.iloc[i]
        image = _load_image(data_dir / row["image_path"], img_size, legacy_resize)
        y_true = int(row["answer"])
        caption = (
            caption_cache.get(str(data_dir / row["image_path"]))
            if caption_cache is not None
            else None
        )

        if use_cot:
            prompt = build_prompt(row, answer_prefix=answer_prefix,
                                  prompt_metadata=prompt_metadata,
                                  caption=caption, use_cot=True)
            enc = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
            for k, v in list(enc.items()):
                if torch.is_tensor(v):
                    enc[k] = v.to(device, non_blocking=True)
            out_ids = model.generate(
                **enc, max_new_tokens=256, do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
            )
            generated = processor.tokenizer.decode(
                out_ids[0][enc["input_ids"].shape[1]:], skip_special_tokens=True,
            ).strip()
            y_pred = _parse_answer_from_cot(generated, int(row["num_choices"]))
        else:
            scores = option_letter_logprobs(
                model, processor, image, row, device, letter_token_map,
                answer_prefix=answer_prefix, prompt_metadata=prompt_metadata,
                caption=caption,
            )
            y_pred = int(scores.argmax().item())

        results.append({
            "correct": 1 if y_pred == y_true else 0,
            "pred": y_pred,
            "answer": y_true,
            "num_choices": int(row["num_choices"]),
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
        items = sorted(stats.items(), key=lambda x: -x[1][0] if rev else x[1][1])
        print(f"  {'Key':34s}  {'N':>5s}  {'Acc':>7s}  {'Bar'}")
        for k, (n, acc) in items:
            bar = "█" * max(1, int(acc * 30))
            print(f"  {str(k):34s}  {n:5d}  {acc:6.2%}  {bar}")

    print(f"\n{'='*60}")
    print(f"  VAL SET EVALUATION  (n={total})")
    print(f"  Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.2f}%)")
    print(f"{'='*60}")

    _print_table("Accuracy by ground-truth answer", _acc_by(_group_by("answer")))
    _print_table("Accuracy by num_choices", _acc_by(_group_by("num_choices")))
    _print_table("Accuracy by grade", _acc_by(_group_by("grade")))
    _print_table("Accuracy by subject", _acc_by(_group_by("subject")))
    _print_table("Accuracy by task", _acc_by(_group_by("task")))
    _print_table("Accuracy by category (worst first)", _acc_by(_group_by("category")), sort_by="acc")

    # Confusion matrix
    print(f"\n{'─'*60}")
    print(f"  Confusion matrix (rows=true, cols=pred)")
    print(f"{'─'*60}")
    answers = sorted(set(r["answer"] for r in results))
    cm: dict[int, Counter] = {a: Counter() for a in answers}
    for r in results:
        cm[r["answer"]][r["pred"]] += 1
    header = "ans\\pred " + "".join(f"  {a}   " for a in answers)
    print(f"  {header}")
    for a in answers:
        row_str = f"    {a}    " + "".join(f" {cm[a].get(p, 0):4d} " for p in answers)
        total_a = sum(1 for r in results if r["answer"] == a)
        acc = sum(1 for r in results if r["answer"] == a and r["correct"])
        print(f"  {row_str}   (acc={acc/total_a:.2%})")

    return {"overall_acc": overall_acc, "total": total}


def main() -> None:
    p = argparse.ArgumentParser(
        description="Inference-only: load adapter + generate submission CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--adapter_dir", type=str, required=True,
                   help="Path to the saved adapter directory (e.g. adapter_best).")
    p.add_argument("--data_dir", type=str, required=True,
                   help="Dataset directory containing test.csv.")
    p.add_argument("--model_id", type=str, required=True,
                   help="Base model Hugging Face ID or local model directory.")
    p.add_argument("--submission_file", type=str, required=True,
                   help="Output CSV path for test-set predictions.")

    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--legacy_resize", action="store_true")
    p.add_argument("--answer_prefix", type=str, default="Answer:",
                   help="Must match the prefix used during training.")
    p.add_argument("--prompt_metadata", type=str, default="none",
                   choices=["none", "subject", "grade", "topic"])
    p.add_argument("--val_limit", type=int, default=0,
                   help="If >0, only predict first N test rows.")
    p.add_argument(
        "--val_csv",
        type=str,
        default="",
        help="Path to val.csv for per-category accuracy evaluation. "
             'Use "auto" to auto-detect in data_dir. '
             "Skip if empty (default).",
    )
    p.add_argument(
        "--use_cot",
        action="store_true",
        help="Use Chain-of-Thought generation (must match training).",
    )
    p.add_argument(
        "--use_caption",
        action="store_true",
        help="Use image captions (must match training).",
    )
    p.add_argument(
        "--adapter_dir_caption",
        type=str,
        default="",
        help="Path to adapter for caption generation (if different from --adapter_dir).",
    )
    args = p.parse_args()

    adapter_dir = Path(args.adapter_dir)
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    data_dir = resolve_data_dir(args.data_dir)
    print(f"Data directory: {data_dir}")
    print(f"Adapter directory: {adapter_dir}")

    # ── load processor ───────────────────────────────────────
    processor = AutoProcessor.from_pretrained(adapter_dir)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if not args.legacy_resize:
        configure_processor_image_size(processor, args.img_size)

    letter_token_map = build_letter_token_map(processor)

    # ── load base model with QLoRA quantization ──────────────
    base_model = build_base_model(args.model_id)

    # ── load LoRA adapter ────────────────────────────────────
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    print("Adapter loaded successfully.")
    model.print_trainable_parameters()

    # ── read test.csv ────────────────────────────────────────
    test_df = pd.read_csv(data_dir / "test.csv")
    test_df["choices"] = test_df["choices"].apply(json.loads)
    if args.val_limit > 0:
        test_df = test_df.iloc[:args.val_limit].copy()
    print(f"Test samples: {len(test_df)}")

    # ── build caption cache if needed ─────────────────────────
    caption_cache: dict[str, str] | None = None
    if args.use_caption:
        device = next(model.parameters()).device
        if args.adapter_dir_caption:
            # Use a separate adapter for caption generation
            print(f"Loading caption adapter from: {args.adapter_dir_caption}")
            cap_base = build_base_model(args.model_id)
            cap_model = PeftModel.from_pretrained(cap_base, args.adapter_dir_caption)
        else:
            cap_model = model
        print("Building caption cache for test set…")
        caption_cache = build_caption_cache(
            cap_model, processor, test_df, data_dir, args.img_size, args.legacy_resize, device,
        )
        # Also build for val if evaluating
        val_path = _resolve_val_csv(args.val_csv, data_dir)
        if val_path is not None:
            val_df = pd.read_csv(val_path)
            val_df["choices"] = val_df["choices"].apply(json.loads)
            caption_cache.update(
                build_caption_cache(
                    cap_model, processor, val_df, data_dir, args.img_size, args.legacy_resize, device,
                )
            )
        if args.adapter_dir_caption:
            del cap_model, cap_base
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        print(f"Caption cache built: {len(caption_cache)} images.")

    # ── val-set evaluation (before submission) ───────────────────
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

    # ── predict ──────────────────────────────────────────────
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
