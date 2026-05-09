"""
QLoRA + K-way logit CE with class-imbalance weights from train.csv answer stats.

Same as finetune_qlora_lettertoken_klogits.py, but before training we print the
distribution of `answer` (option index 0..K-1), then build balanced per-class
weights and pass them to F.cross_entropy(..., weight=).

Weight formula (sklearn-style balanced): w_k = N / (C * n_k) for classes k
with count n_k > 0, where N=len(train), C=number of classes that appear; then
renormalize weights over present classes to mean 1.

PyTorch cross_entropy(..., weight=) requires weight.shape[0] == num_classes in the
logits vector. Each sample has K=num_choices logits, so we pass class_weights[:K]
(not the full length-10 vector).
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
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
CHOICE_LETTERS = "ABCDEFGHIJ"
MAX_TRAINABLE_PARAMETERS = 5_000_000
# Max possible distinct answer indices (A..J -> 0..9)
MAX_ANSWER_CLASSES = len(CHOICE_LETTERS)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def require_trainable_parameters_within_limit(
    model: torch.nn.Module, max_params: int = MAX_TRAINABLE_PARAMETERS
) -> int:
    n = count_trainable_parameters(model)
    if n > max_params:
        raise ValueError(
            f"Trainable parameters ({n:,}) exceed the limit ({max_params:,}). "
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


def print_train_answer_distribution(train_df: pd.DataFrame, num_classes: int) -> None:
    """Print counts and proportions for `answer` column (option indices)."""
    mx = int(train_df["answer"].max())
    if mx >= num_classes:
        raise ValueError(
            f"train answer max index is {mx}, but --num_weight_classes is {num_classes}. "
            "Increase --num_weight_classes or check train.csv."
        )
    y = train_df["answer"].astype(int).values
    counts = np.bincount(y, minlength=num_classes)[:num_classes]
    n = len(y)
    print(f"\n--- train.csv answer index distribution (N={n}) ---")
    for k in range(num_classes):
        c = int(counts[k])
        if c == 0:
            continue
        pct = 100.0 * c / n
        letter = CHOICE_LETTERS[k] if k < len(CHOICE_LETTERS) else "?"
        print(f"  answer=={k} ({letter}): {c:6d}  ({pct:5.2f}%)")
    zeros = int(np.sum(counts == 0))
    if zeros:
        print(f"  (indices with 0 samples in train: {zeros})")
    print("--- end distribution ---\n")


def compute_balanced_class_weights(
    train_df: pd.DataFrame, num_classes: int
) -> torch.Tensor:
    """
    Balanced weights: w_k = N / (C * n_k) for k with n_k > 0,
    C = number of classes with at least one sample. Missing classes get weight 1.0.
    Renormalize weights over present classes to arithmetic mean 1.0.
    """
    y = train_df["answer"].astype(int).values
    counts = np.bincount(y, minlength=num_classes)[:num_classes]
    N = len(y)
    present = counts > 0
    n_present = int(present.sum())
    if n_present == 0:
        raise ValueError("No training samples.")
    w = np.ones(num_classes, dtype=np.float64)
    for k in range(num_classes):
        if counts[k] > 0:
            w[k] = N / (n_present * counts[k])
    w[present] = w[present] / w[present].mean()
    return torch.tensor(w, dtype=torch.float32)


def build_prompt(row: pd.Series) -> str:
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
    if context_str:
        prompt += f"Context:\n{context_str}\n\n"
    prompt += f"Question: {row['question']}\n"
    prompt += f"Choices:\n{choices_str}\n"
    prompt += "Answer:"
    return prompt


def build_letter_token_map(processor) -> dict[str, int]:
    token_map: dict[str, int] = {}
    for letter in CHOICE_LETTERS:
        token_ids = processor.tokenizer(
            f" {letter}", add_special_tokens=False
        ).input_ids
        if not token_ids:
            raise ValueError(f"Tokenizer produced no token for letter '{letter}'.")
        if len(token_ids) > 1:
            print(
                f"Warning: tokenizer splits ' {letter}' into {len(token_ids)} tokens; "
                "using the first token for letter-only scoring."
            )
        token_map[letter] = int(token_ids[0])
    return token_map


def pick_lora_target_modules(model: torch.nn.Module) -> list[str]:
    candidates: set[str] = set()
    preferred = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "down_proj",
        "gate_proj",
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


def option_letter_logits(
    model: torch.nn.Module,
    processor,
    image: Image.Image,
    row: pd.Series,
    device: torch.device,
    letter_token_map: dict[str, int],
    need_grad: bool,
) -> torch.Tensor:
    prompt = build_prompt(row)
    n_opt = int(row["num_choices"])
    enc = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
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

    logits = out.logits[0]
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    next_token_logits = logits[-1]

    tok_ids = [
        letter_token_map[CHOICE_LETTERS[j]] for j in range(n_opt)
    ]
    idx = torch.tensor(tok_ids, device=next_token_logits.device, dtype=torch.long)
    return next_token_logits.index_select(0, idx)


def run_train(
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
    letter_token_map: dict[str, int],
    class_weights: torch.Tensor,
    use_sample_weights: bool,
) -> None:
    device = next(model.parameters()).device
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cw = class_weights.to(device) if use_sample_weights else None
    if use_sample_weights:
        print("Using balanced class weights in cross_entropy (mean over present≈1).")

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
    losses: list[float] = []
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
            desc=f"Epoch {epoch+1}/{num_epochs} (weighted K-logit CE)",
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
                n_opt = int(row["num_choices"])
                logits_k = option_letter_logits(
                    model,
                    processor,
                    image,
                    row,
                    device,
                    letter_token_map,
                    need_grad=True,
                )
                kwargs = {}
                if cw is not None:
                    # CE weight length must match logits_k.shape[0] == n_opt (not full 10).
                    kwargs["weight"] = cw[:n_opt]
                loss = F.cross_entropy(
                    logits_k.unsqueeze(0),
                    torch.tensor([y], device=device, dtype=torch.long),
                    **kwargs,
                )
                (loss / (b_eff * grad_accum_steps)).backward()
                total_steps += 1
                losses.append(float(loss.item()))
                if total_steps % logging_steps == 0 and losses:
                    pbar.set_postfix(
                        last_loss=float(losses[-1]),
                        ma=float(
                            np.mean(losses[-min(logging_steps, len(losses)) :])
                        ),
                    )

            accum_micro += 1
            if accum_micro >= grad_accum_steps:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                accum_micro = 0

        if accum_micro > 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

        tail = losses[-min(500, len(losses)) :]
        print(
            f"Epoch {epoch + 1}/{num_epochs} done — "
            f"mean(loss last≤500 steps)={float(np.mean(tail)):.5f}  "
            f"(global sample steps={total_steps})"
        )

    saved = Path(output_dir) / "adapter"
    saved.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(saved))
    processor.save_pretrained(str(saved))
    print(f"Saved LoRA and processor to {saved}")


@torch.inference_mode()
def predict_submission(
    model: torch.nn.Module,
    processor,
    test_df: pd.DataFrame,
    data_dir: Path,
    img_size: int,
    output_file: Path,
    letter_token_map: dict[str, int],
) -> None:
    device = next(model.parameters()).device
    model.eval()
    preds: list[int] = []
    ids: list[str] = []

    for i in tqdm(range(len(test_df)), desc="Predicting test (K-logit argmax)"):
        row = test_df.iloc[i]
        image = Image.open(data_dir / row["image_path"]).convert("RGB")
        image = image.resize((img_size, img_size), Image.BICUBIC)
        logits_k = option_letter_logits(
            model, processor, image, row, device, letter_token_map, need_grad=False
        )
        preds.append(int(logits_k.argmax().item()))
        ids.append(row["id"])

    out = pd.DataFrame({"id": ids, "answer": preds})
    out.to_csv(output_file, index=False)
    print(f"Saved submission to: {output_file}")
    print(out.head())


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "QLoRA + weighted K-way logit CE (balanced weights from train answer stats)."
        )
    )
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--model_id", type=str, default=MODEL_ID)
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qlora_lettertoken_klogits_weighted",
    )
    p.add_argument(
        "--submission_file",
        type=str,
        default="my_submission_lettertoken_klogits_weighted.csv",
    )
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_trainable_params", type=int, default=MAX_TRAINABLE_PARAMETERS)
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
    p.add_argument("--val_limit", type=int, default=0, help="If >0, only first N test rows.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--num_weight_classes",
        type=int,
        default=MAX_ANSWER_CLASSES,
        help="Length of weight vector (cover answer indices 0..C-1; default 10 for A..J).",
    )
    p.add_argument(
        "--no_class_balance",
        action="store_true",
        help="Disable class weights (same CE as unweighted klogits).",
    )
    p.add_argument(
        "--test_only",
        action="store_true",
        help="Skip training; load LoRA from --checkpoint and only run test inference.",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to saved PEFT adapter.",
    )
    args = p.parse_args()
    if args.train_batch_size < 1:
        raise ValueError("--train_batch_size must be >= 1")
    if args.test_only and not (args.checkpoint and str(args.checkpoint).strip()):
        raise ValueError("--test_only requires a non-empty --checkpoint path to the adapter.")
    if args.num_weight_classes < 1:
        raise ValueError("--num_weight_classes must be >= 1")

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
    letter_token_map = build_letter_token_map(processor)

    if args.test_only:
        test_df = pd.read_csv(data_dir / "test.csv")
        test_df["choices"] = test_df["choices"].apply(json.loads)
        if args.val_limit > 0:
            test_df = test_df.iloc[: args.val_limit].copy()
        class_weights = torch.ones(args.num_weight_classes)
    else:
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        train_df["choices"] = train_df["choices"].apply(json.loads)
        test_df["choices"] = test_df["choices"].apply(json.loads)
        if args.train_limit > 0:
            train_df = train_df.iloc[: args.train_limit].copy()
        if args.val_limit > 0:
            test_df = test_df.iloc[: args.val_limit].copy()

        print_train_answer_distribution(train_df, args.num_weight_classes)
        class_weights = compute_balanced_class_weights(train_df, args.num_weight_classes)
        print(
            "Balanced class weights (index k -> weight for gold label k):\n  "
            + ", ".join(f"{k}:{class_weights[k]:.4f}" for k in range(args.num_weight_classes))
        )

    model, _ = maybe_build_qlora_model(args.model_id)
    if not torch.cuda.is_available():
        model = model.to(torch.device("cpu"))

    if args.test_only:
        print(f"Loading LoRA from {args.checkpoint}")
        model = PeftModel.from_pretrained(model, str(Path(args.checkpoint)))
        model.eval()
    else:
        target_modules = pick_lora_target_modules(model)
        print(f"LoRA target modules: {target_modules}")
        model = get_peft_model(
            model,
            LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            ),
        )
        model.print_trainable_parameters()
        require_trainable_parameters_within_limit(
            model, max_params=args.max_trainable_params
        )

        run_train(
            model=model,
            processor=processor,
            train_df=train_df,
            data_dir=data_dir,
            img_size=args.img_size,
            output_dir=str(output_dir),
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            grad_accum_steps=args.grad_accum_steps,
            train_batch_size=args.train_batch_size,
            logging_steps=args.logging_steps,
            seed=args.seed,
            letter_token_map=letter_token_map,
            class_weights=class_weights,
            use_sample_weights=not args.no_class_balance,
        )

    predict_submission(
        model=model,
        processor=processor,
        test_df=test_df,
        data_dir=data_dir,
        img_size=args.img_size,
        output_file=Path(args.submission_file),
        letter_token_map=letter_token_map,
    )


if __name__ == "__main__":
    main()
