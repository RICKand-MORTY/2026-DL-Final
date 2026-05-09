"""
QLoRA + option log-likelihood (Scheme A) for ScienceQA-style MCQ.
For each example, s_j = sum over tokenized " {letter_j}" of
log p(token | image, prefix) after a fixed "Answer:" prefix;
train with CE(softmax(s), y); predict argmax_j s_j.
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
from transformers import (
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
)

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
CHOICE_LETTERS = "ABCDEFGHIJ"
MAX_TRAINABLE_PARAMETERS = 5_000_000


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def require_trainable_parameters_within_limit(
    model: torch.nn.Module, max_params: int = MAX_TRAINABLE_PARAMETERS
) -> int:
    n = count_trainable_parameters(model)
    if n > max_params:
        raise ValueError(
            f"Trainable parameters ({n:,}) exceed the limit ({max_params:,}). "
            "This includes all adapters, LoRA weights, and any other updated parameters. "
            "Lower --lora_r / --lora_alpha, or reduce LoRA target_modules, then retry."
        )
    print(f"Trainable parameter check OK: {n:,} / {max_params:,} (max).")
    return n


def resolve_data_dir(user_data_dir: str) -> Path:
    candidates = [Path(user_data_dir), Path("data"), Path("pixels-to-predictions")]
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
            return candidate
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        f"Could not find valid dataset directory. Searched: {searched}"
    )


def build_prompt(row: pd.Series, include_answer: bool = False) -> str:
    context_parts: list = []
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
    if context_str:
        prompt += f"Context:\n{context_str}\n\n"
    prompt += f"Question: {row['question']}\n"
    prompt += f"Choices:\n{choices_str}\n"
    prompt += "Answer:"

    if include_answer:
        answer_idx = int(row["answer"])
        prompt += f" {CHOICE_LETTERS[answer_idx]}"
    return prompt


def _forward_enc(model, enc, device, need_grad: bool):
    for k, v in list(enc.items()):
        if torch.is_tensor(v):
            enc[k] = v.to(device, non_blocking=True)
    with torch.amp.autocast(
        "cuda" if device.type == "cuda" else "cpu", enabled=(device.type == "cuda")
    ):
        if need_grad:
            return model(**{**enc}, return_dict=True, use_cache=False)
        with torch.inference_mode():
            return model(**{**enc}, return_dict=True, use_cache=False)


def _suffix_score_from_logits(lg, ids1, l_p, l_f) -> torch.Tensor:
    """Sum log p(ids[pos]) for pos in l_p..l_f-1 (causal LM: logits at pos-1)."""
    if lg.dtype in (torch.float16, torch.bfloat16):
        lg = lg.float()
    acc: torch.Tensor | None = None
    for pos in range(l_p, l_f):
        logp = F.log_softmax(lg[pos - 1], -1)
        v_a = logp[ids1[pos].long()]
        acc = v_a if acc is None else (acc + v_a)
    assert acc is not None
    return acc


def option_logprobs(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    row: pd.Series,
    device: torch.device,
    need_grad: bool = False,
) -> torch.Tensor:
    """
    Returns (K,) tensor. need_grad True for training (LoRA); False for eval.
    """
    prefix = build_prompt(row, include_answer=False)
    n_opt = int(row["num_choices"])
    with torch.no_grad():
        enc_p = processor(
            text=[prefix],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
    l_p = int(enc_p["input_ids"].shape[1])
    scores: list = []
    for j in range(n_opt):
        text_j = prefix + f" {CHOICE_LETTERS[j]}"
        enc = processor(
            text=[text_j],
            images=[image],
            return_tensors="pt",
            padding=True,
        )
        l_f = int(enc["input_ids"].shape[1])
        out = _forward_enc(model, enc, device, need_grad)
        lg = out.logits[0]
        ids1 = enc["input_ids"][0]
        s = _suffix_score_from_logits(
            lg, ids1.to(lg.device), l_p, l_f
        )
        scores.append(s)
    return torch.stack(scores)


@torch.inference_mode()
def compute_option_logprobs(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    row: pd.Series,
    device: torch.device,
) -> torch.Tensor:
    return option_logprobs(
        model, processor, image, row, device, need_grad=False
    )


def compute_option_logprobs_train(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    row: pd.Series,
    device: torch.device,
) -> torch.Tensor:
    return option_logprobs(
        model, processor, image, row, device, need_grad=True
    )


def pick_lora_target_modules(model: torch.nn.Module) -> list[str]:
    candidates: set = set()
    preferred = [
        "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"
    ]
    for name, _ in model.named_modules():
        short_name = name.split(".")[-1]
        if short_name in preferred:
            candidates.add(short_name)
    if candidates:
        return sorted(candidates)
    return ["q_proj", "v_proj"]


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


def run_train_likelihood(
    model: torch.nn.Module,
    processor,
    train_df: pd.DataFrame,
    data_dir: Path,
    img_size: int,
    output_dir: str,
    learning_rate: float,
    num_epochs: int,
    grad_accum_steps: int,
    train_batch_size: int,
    logging_steps: int,
    seed: int,
) -> None:
    device = next(model.parameters()).device
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model.train()
    try:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    except (AttributeError, ValueError):
        pass
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=0.01)
    n = len(train_df)
    total_steps = 0
    accum_loss: list[float] = []
    eff_batch = train_batch_size * grad_accum_steps
    print(
        f"train_batch_size={train_batch_size}, grad_accum_steps={grad_accum_steps} "
        f"=> effective batch ~{eff_batch} samples per optimizer step."
    )
    for epoch in range(num_epochs):
        order = list(range(n))
        random.shuffle(order)
        num_micro = math.ceil(n / train_batch_size)
        pbar = tqdm(
            range(0, n, train_batch_size),
            total=num_micro,
            desc=f"Epoch {epoch+1}/{num_epochs} (likelihood CE, micro-batches)",
        )
        opt.zero_grad(set_to_none=True)
        accum_micro = 0
        for start in pbar:
            mb = order[start : start + train_batch_size]
            b_eff = len(mb)
            for idx in mb:
                row = train_df.iloc[idx]
                image = Image.open(data_dir / row["image_path"]).convert("RGB")
                image = image.resize((img_size, img_size), Image.BICUBIC)
                y = int(row["answer"])
                sc = compute_option_logprobs_train(
                    model, processor, image, row, device
                )
                loss = F.cross_entropy(
                    sc.unsqueeze(0),
                    torch.tensor([y], device=device, dtype=torch.long),
                )
                (loss / (b_eff * grad_accum_steps)).backward()
                total_steps += 1
                accum_loss.append(float(loss.item()))
            accum_micro += 1
            if accum_micro >= grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                accum_micro = 0
            if total_steps % logging_steps == 0 and accum_loss:
                pbar.set_postfix(
                    last_loss=float(accum_loss[-1]),
                    ma=float(np.mean(accum_loss[-min(logging_steps, len(accum_loss)) :])),
                )
        if accum_micro > 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

    saved = Path(output_dir) / "adapter"
    saved.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(saved))
    processor.save_pretrained(str(saved))
    print(f"Saved LoRA and processor to {saved}")


@torch.inference_mode()
def predict_likelihood_submission(
    model: torch.nn.Module,
    processor,
    test_df: pd.DataFrame,
    data_dir: Path,
    img_size: int,
    output_file: Path,
) -> None:
    device = next(model.parameters()).device
    model.eval()
    preds: list[int] = []
    ids: list = []
    for i in tqdm(range(len(test_df)), desc="Predicting test (likelihood argmax)"):
        row = test_df.iloc[i]
        image = Image.open(data_dir / row["image_path"]).convert("RGB")
        image = image.resize((img_size, img_size), Image.BICUBIC)
        s = compute_option_logprobs(model, processor, image, row, device)
        preds.append(int(s.argmax().item()))
        ids.append(row["id"])
    out = pd.DataFrame({"id": ids, "answer": preds})
    out.to_csv(output_file, index=False)
    print(f"Saved submission to: {output_file}")
    print(out.head())


def main() -> None:
    p = argparse.ArgumentParser(
        description="QLoRA + option log-likelihood training and likelihood inference."
    )
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--model_id", type=str, default=MODEL_ID)
    p.add_argument("--output_dir", type=str, default="outputs/qlora_likelihood")
    p.add_argument(
        "--submission_file", type=str, default="my_submission_likelihood.csv"
    )
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--max_trainable_params", type=int, default=MAX_TRAINABLE_PARAMETERS
    )
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--grad_accum_steps", type=int, default=8)
    p.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Samples per micro-batch; effective batch ≈ train_batch_size * grad_accum_steps.",
    )
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--train_limit", type=int, default=0)
    p.add_argument(
        "--val_limit",
        type=int,
        default=0,
        help="If >0, only first N test rows for quick runs.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--test_only",
        action="store_true",
        help="Skip training; load LoRA from --checkpoint and only run test inference.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to saved PEFT adapter (e.g. outputs/qlora_likelihood/adapter).",
    )
    args = p.parse_args()
    if args.train_batch_size < 1:
        raise ValueError("--train_batch_size must be >= 1")
    if args.test_only and not (args.checkpoint and str(args.checkpoint).strip()):
        raise ValueError("--test_only requires a non-empty --checkpoint path to the adapter.")

    data_dir = resolve_data_dir(args.data_dir)
    output_dir = Path(args.output_dir)
    if not args.test_only:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using data directory: {data_dir}")
    if args.test_only:
        ckpt = Path(args.checkpoint)
        if not ckpt.is_dir():
            raise FileNotFoundError(f"Checkpoint not found or not a directory: {ckpt}")
        try:
            processor = AutoProcessor.from_pretrained(str(ckpt))
        except (OSError, ValueError) as e:
            print(
                f"Failed to load processor from {ckpt}, falling back to --model_id: {e!r}"
            )
            processor = AutoProcessor.from_pretrained(args.model_id)
    else:
        processor = AutoProcessor.from_pretrained(args.model_id)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if args.test_only:
        test_df = pd.read_csv(data_dir / "test.csv")
        test_df["choices"] = test_df["choices"].apply(json.loads)
        if args.val_limit > 0:
            test_df = test_df.iloc[: args.val_limit].copy()
    else:
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        train_df["choices"] = train_df["choices"].apply(json.loads)
        test_df["choices"] = test_df["choices"].apply(json.loads)
        if args.train_limit > 0:
            train_df = train_df.iloc[: args.train_limit].copy()
        if args.val_limit > 0:
            test_df = test_df.iloc[: args.val_limit].copy()

    model, _ = maybe_build_qlora_model(args.model_id)
    if not torch.cuda.is_available():
        model = model.to(torch.device("cpu"))

    if args.test_only:
        print(f"Loading LoRA from {args.checkpoint}")
        model = PeftModel.from_pretrained(model, str(Path(args.checkpoint)))
        model.eval()
    else:
        tmods = pick_lora_target_modules(model)
        print(f"LoRA target modules: {tmods}")
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=tmods,
            ),
        )
        model.print_trainable_parameters()
        require_trainable_parameters_within_limit(
            model, max_params=args.max_trainable_params
        )
        run_train_likelihood(
            model,
            processor,
            train_df,
            data_dir,
            args.img_size,
            str(output_dir),
            args.learning_rate,
            args.num_epochs,
            args.grad_accum_steps,
            args.train_batch_size,
            args.logging_steps,
            args.seed,
        )
    predict_likelihood_submission(
        model,
        processor,
        test_df,
        data_dir,
        args.img_size,
        Path(args.submission_file),
    )


if __name__ == "__main__":
    main()
