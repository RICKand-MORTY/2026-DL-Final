"""
QLoRA + letter-token scoring with K-way logit cross-entropy (MCQ).

Same pipeline as finetune_qlora_lettertoken.py, but training uses raw next-token
logits at the letter token ids for the K options (slice of lm_head output),
then F.cross_entropy — i.e. softmax only over those K logits. Inference scores
with the same K logits and argmax (equivalent ranking to softmax over K).

This avoids: full-vocab log_softmax → gather log p → cross_entropy (double
normalization). See finetune_qlora_lettertoken.py for the log-prob variant.
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
    """
    Map each choice letter to ONE token id used for next-token scoring.
    We encode with a leading space to match "Answer: A" style completions.
    """
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
    """
    (K,) raw next-token logits at option letter token ids after "Answer:".
    CrossEntropy(logits, y) applies softmax over these K values only.
    """
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

    logits = out.logits[0]  # (T, V)
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    next_token_logits = logits[-1]  # (V,)

    tok_ids = [
        letter_token_map[CHOICE_LETTERS[j]] for j in range(n_opt)
    ]
    idx = torch.tensor(tok_ids, device=next_token_logits.device, dtype=torch.long)
    return next_token_logits.index_select(0, idx)


@torch.inference_mode()
def evaluate_val_accuracy(
    model: torch.nn.Module,
    processor,
    val_df: pd.DataFrame,
    data_dir: Path,
    img_size: int,
    device: torch.device,
    letter_token_map: dict[str, int],
) -> float:
    """Fraction of val rows where argmax letter matches gold answer index."""
    was_training = model.training
    model.eval()
    n = len(val_df)
    if n == 0:
        return 0.0
    correct = 0
    for i in tqdm(range(n), desc="Val accuracy", leave=False):
        row = val_df.iloc[i]
        image = Image.open(data_dir / row["image_path"]).convert("RGB")
        image = image.resize((img_size, img_size), Image.BICUBIC)
        logits_k = option_letter_logits(
            model, processor, image, row, device, letter_token_map, need_grad=False
        )
        pred = int(logits_k.argmax().item())
        if pred == int(row["answer"]):
            correct += 1
    if was_training:
        model.train()
    return correct / n


def deterministic_epoch_order(epoch_idx: int, n: int, seed: int) -> list[int]:
    """Same order every run — required to resume mid-epoch."""
    rng = random.Random(seed + 1_000_003 * epoch_idx + 13_369)
    order = list(range(n))
    rng.shuffle(order)
    return order


def find_resume_microbatch_position(
    order: list[int], n: int, batch_size: int, resume_skip: int
) -> tuple[int, int]:
    """
    Skip the first `resume_skip` samples in epoch traversal order.
    Returns (outer_start, inner_j) for the first sample that still needs training.
    """
    if resume_skip <= 0:
        return 0, 0
    linear = 0
    for start in range(0, n, batch_size):
        mb = order[start : start + batch_size]
        for j in range(len(mb)):
            if linear == resume_skip:
                return start, j
            linear += 1
    raise ValueError(
        f"resume_skip={resume_skip} exceeds epoch length {n} (invalid checkpoint)."
    )


TRAINING_STATE_VERSION = 1


def build_training_state_dict(
    *,
    total_steps: int,
    optimizer_step: int,
    accum_micro: int,
    epoch_idx: int,
    samples_into_epoch: int,
    n_samples_epoch: int,
    train_batch_size: int,
    grad_accum_steps: int,
    seed: int,
    learning_rate: float,
    num_epochs: int,
    eval_steps: int,
) -> dict:
    return {
        "format_version": TRAINING_STATE_VERSION,
        "total_steps": total_steps,
        "optimizer_step": optimizer_step,
        "accum_micro": accum_micro,
        "epoch_idx": epoch_idx,
        "samples_into_epoch": samples_into_epoch,
        "n_samples_epoch": n_samples_epoch,
        "train_batch_size": train_batch_size,
        "grad_accum_steps": grad_accum_steps,
        "seed": seed,
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "eval_steps": eval_steps,
    }


def save_training_state_payload(
    dest_dir: Path,
    state: dict,
    optimizer: torch.optim.Optimizer,
) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    path = dest_dir / "training_state.json"
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    torch.save(optimizer.state_dict(), dest_dir / "optimizer.pt")


def load_resume_and_best_metrics(
    output_dir: Path,
) -> tuple[dict | None, float, int, int, int]:
    """Load optional best_* tracking from output_dir when resuming."""
    best_path = output_dir / "best_checkpoint.json"
    if not best_path.is_file():
        return None, -1.0, -1, -1, -1
    try:
        data = json.loads(best_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None, -1.0, -1, -1, -1
    return (
        data,
        float(data.get("best_val_accuracy", -1.0)),
        int(data.get("best_training_samples_seen", -1)),
        int(data.get("best_optimizer_step", -1)),
        int(data.get("best_epoch", -1)),
    )


def find_training_state_and_optimizer_dir(
    resume_adapter_dir: Path,
) -> tuple[Path | None, Path | None]:
    """
    training_state.json may live next to the adapter or only under output_dir
    (older runs). Optimizer.pt usually follows the same directory as training_state.json.
    """
    candidates: list[Path] = [resume_adapter_dir / "training_state.json"]
    if (
        resume_adapter_dir.name.startswith("sample_step_")
        and resume_adapter_dir.parent.name == "checkpoints"
    ):
        candidates.append(resume_adapter_dir.parent.parent / "training_state.json")
    seen: set[str] = set()
    for p in candidates:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        if p.is_file():
            return p, p.parent
    return None, None


def build_resume_state_from_eval_json(
    eval_path: Path,
    n: int,
    args,
) -> dict:
    """Fallback when training_state.json was never written (legacy checkpoints)."""
    ev = json.loads(eval_path.read_text(encoding="utf-8"))
    total_steps = int(ev["training_samples_seen"])
    optimizer_step = int(ev["optimizer_step"])
    epoch_idx = total_steps // n
    samples_into_epoch = total_steps - epoch_idx * n
    ep_meta = int(ev.get("epoch", epoch_idx + 1))
    if ep_meta != epoch_idx + 1:
        print(
            f"Warning: eval.json epoch={ep_meta} vs inferred epoch_idx+1={epoch_idx + 1} "
            "from total_steps — using arithmetic from training_samples_seen."
        )
    return {
        "format_version": TRAINING_STATE_VERSION,
        "total_steps": total_steps,
        "optimizer_step": optimizer_step,
        "accum_micro": 0,
        "epoch_idx": epoch_idx,
        "samples_into_epoch": samples_into_epoch,
        "n_samples_epoch": n,
        "train_batch_size": args.train_batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "num_epochs": args.num_epochs,
        "eval_steps": args.eval_steps,
    }


def resolve_resume_state(
    resume_adapter_dir: Path,
    n: int,
    args,
) -> tuple[dict, Path]:
    """
    Returns (state_dict, directory_that_has_optimizer_pt_or_adapter_dir).
    """
    state_path, opt_dir = find_training_state_and_optimizer_dir(resume_adapter_dir)
    if state_path is not None:
        resume_state = json.loads(state_path.read_text(encoding="utf-8"))
        print(f"Loaded training state from {state_path}")
        return resume_state, opt_dir if opt_dir is not None else resume_adapter_dir

    eval_path = resume_adapter_dir / "eval.json"
    if eval_path.is_file():
        print(
            f"Warning: no training_state.json near {resume_adapter_dir}; "
            f"reconstructing resume fields from {eval_path} (accum_micro=0, fresh optimizer "
            "unless optimizer.pt exists under output_dir)."
        )
        st = build_resume_state_from_eval_json(eval_path, n, args)
        root_guess = (
            resume_adapter_dir.parent.parent
            if (
                resume_adapter_dir.name.startswith("sample_step_")
                and resume_adapter_dir.parent.name == "checkpoints"
            )
            else resume_adapter_dir
        )
        return st, root_guess

    raise FileNotFoundError(
        f"No training_state.json and no eval.json under {resume_adapter_dir}. "
        "Cannot resume: need at least eval.json in the checkpoint folder, or "
        "training_state.json in the checkpoint or output_dir."
    )


def assert_resume_compatible(resume: dict, n: int, args) -> None:
    if int(resume["n_samples_epoch"]) != n:
        raise ValueError(
            f"Resume n_samples_epoch={resume['n_samples_epoch']} != current train rows={n}. "
            "Use the same train.csv / --train_limit as the original run."
        )
    if int(resume["train_batch_size"]) != args.train_batch_size:
        raise ValueError("Resume checkpoint requires the same --train_batch_size as the original run.")
    if int(resume["grad_accum_steps"]) != args.grad_accum_steps:
        raise ValueError("Resume checkpoint requires the same --grad_accum_steps as the original run.")
    if int(resume["seed"]) != args.seed:
        raise ValueError("Resume checkpoint requires the same --seed as the original run.")
    if abs(float(resume["learning_rate"]) - args.learning_rate) > 1e-15:
        raise ValueError("Resume checkpoint requires the same --learning_rate as the original run.")
    if int(resume["num_epochs"]) != args.num_epochs:
        raise ValueError(
            "Resume checkpoint requires the same --num_epochs as the original run "
            "(total epoch budget)."
        )
    if int(resume.get("eval_steps", 0)) != args.eval_steps:
        raise ValueError("Resume checkpoint requires the same --eval_steps as the original run.")


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
    val_df: pd.DataFrame | None,
    eval_steps: int,
    eval_val_limit: int,
    resume: dict | None,
    resume_adapter_dir: Path | None,
    resume_optimizer_dir: Path | None,
) -> Path | None:
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

    resume_epoch_idx = int(resume["epoch_idx"]) if resume else 0
    resume_skip = int(resume["samples_into_epoch"]) if resume else 0
    if resume is not None:
        while resume_skip >= n:
            resume_skip -= n
            resume_epoch_idx += 1
        if resume_epoch_idx >= num_epochs:
            raise ValueError(
                f"Checkpoint already finished all {num_epochs} epoch(s); nothing left to train."
            )

    total_steps = int(resume["total_steps"]) if resume else 0
    optimizer_step = int(resume["optimizer_step"]) if resume else 0
    losses: list[float] = []

    if resume is not None:
        if resume_adapter_dir is None:
            raise ValueError("resume_adapter_dir required when resume dict is set.")
        opt_root = resume_optimizer_dir or resume_adapter_dir
        opt_path = opt_root / "optimizer.pt"
        if opt_path.is_file():
            opt.load_state_dict(torch.load(opt_path, map_location=device))
            print(f"Loaded optimizer state from {opt_path}")
        else:
            print(
                f"Warning: no {opt_path} — optimizer re-initialized; "
                "training dynamics may differ from the interrupted run."
            )

    eff_batch = train_batch_size * grad_accum_steps
    print(
        f"train_batch_size={train_batch_size}, grad_accum_steps={grad_accum_steps} "
        f"=> effective batch ~{eff_batch} samples per optimizer step."
    )
    if resume is not None:
        print(
            f"Resuming: epoch_idx={resume_epoch_idx}, samples_into_epoch={resume_skip}, "
            f"total_steps={total_steps}, optimizer_step={optimizer_step}, "
            f"accum_micro={resume.get('accum_micro', 0)}"
        )

    out_root = Path(output_dir)
    ckpt_root = out_root / "checkpoints"
    best_adapter_dir = out_root / "adapter_best"
    do_eval = (
        eval_steps > 0
        and val_df is not None
        and len(val_df) > 0
    )
    if do_eval:
        if eval_val_limit > 0:
            val_df = val_df.iloc[: eval_val_limit].copy()
        print(
            f"Validation: {len(val_df)} rows from val.csv; "
            f"every {eval_steps} training sample step(s) "
            f"(same counter as 'global sample steps' in epoch logs) — "
            f"checkpoints under {ckpt_root}"
        )

    if resume is not None:
        _bj, best_val_acc, best_optimizer_step, best_training_samples_seen, best_epoch = (
            load_resume_and_best_metrics(out_root)
        )
        if _bj is not None:
            print(
                f"Loaded prior best val_accuracy={best_val_acc:.6f} "
                f"(samples_seen={best_training_samples_seen}) from best_checkpoint.json"
            )
        else:
            best_val_acc = -1.0
            best_optimizer_step = -1
            best_training_samples_seen = -1
            best_epoch = -1
    else:
        best_val_acc = -1.0
        best_optimizer_step = -1
        best_training_samples_seen = -1
        best_epoch = -1

    last_eval_sample_step = -1

    accum_micro = 0

    def snapshot_training_state(
        epoch_idx_for_state: int,
        extra_dirs: list[Path],
        *,
        samples_into_override: int | None = None,
    ) -> None:
        if samples_into_override is not None:
            sie = samples_into_override
        else:
            sie = total_steps - epoch_idx_for_state * n
        st = build_training_state_dict(
            total_steps=total_steps,
            optimizer_step=optimizer_step,
            accum_micro=accum_micro,
            epoch_idx=epoch_idx_for_state,
            samples_into_epoch=sie,
            n_samples_epoch=n,
            train_batch_size=train_batch_size,
            grad_accum_steps=grad_accum_steps,
            seed=seed,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            eval_steps=eval_steps,
        )
        for d in extra_dirs:
            save_training_state_payload(d, st, opt)

    def maybe_validate_and_save(epoch_idx: int, reason: str) -> None:
        nonlocal best_val_acc, best_optimizer_step, best_training_samples_seen, best_epoch, last_eval_sample_step
        if not do_eval:
            return
        acc = evaluate_val_accuracy(
            model,
            processor,
            val_df,
            data_dir,
            img_size,
            device,
            letter_token_map,
        )
        tag = f"sample_step_{total_steps:07d}"
        save_dir = ckpt_root / tag
        save_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(save_dir))
        processor.save_pretrained(str(save_dir))
        meta = {
            "training_samples_seen": total_steps,
            "optimizer_step": optimizer_step,
            "epoch": epoch_idx + 1,
            "val_accuracy": acc,
            "reason": reason,
        }
        with open(save_dir / "eval.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        snapshot_training_state(epoch_idx, [save_dir, out_root])
        print(
            f"[val] {reason} — training_samples_seen={total_steps}, "
            f"optimizer_step={optimizer_step}, epoch={epoch_idx + 1}, "
            f"val_accuracy={acc:.6f} → saved {save_dir}"
        )
        if acc > best_val_acc:
            best_val_acc = acc
            best_optimizer_step = optimizer_step
            best_training_samples_seen = total_steps
            best_epoch = epoch_idx + 1
            best_adapter_dir.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(str(best_adapter_dir))
            processor.save_pretrained(str(best_adapter_dir))
            snapshot_training_state(epoch_idx, [best_adapter_dir])
            best_info = {
                "best_training_samples_seen": best_training_samples_seen,
                "best_optimizer_step": best_optimizer_step,
                "best_epoch": best_epoch,
                "best_val_accuracy": best_val_acc,
                "adapter_path": str(best_adapter_dir.resolve()),
                "eval_tag": tag,
            }
            with open(out_root / "best_checkpoint.json", "w", encoding="utf-8") as f:
                json.dump(best_info, f, indent=2)
            print(
                f"  ★ New best val_accuracy={best_val_acc:.6f} "
                f"(training_samples_seen={best_training_samples_seen}, "
                f"optimizer_step={best_optimizer_step}, epoch={best_epoch}) "
                f"→ {best_adapter_dir}"
            )
        last_eval_sample_step = total_steps

    for epoch in range(resume_epoch_idx, num_epochs):
        order = deterministic_epoch_order(epoch, n, seed)
        skip_for_epoch = resume_skip if epoch == resume_epoch_idx else 0
        resume_start, resume_j = find_resume_microbatch_position(
            order, n, train_batch_size, skip_for_epoch
        )

        if epoch == resume_epoch_idx and skip_for_epoch > 0:
            accum_micro = int(resume.get("accum_micro", 0)) if resume else 0
        else:
            opt.zero_grad(set_to_none=True)
            accum_micro = 0

        num_micro = math.ceil(n / train_batch_size)
        pbar = tqdm(
            range(0, n, train_batch_size),
            total=num_micro,
            desc=f"Epoch {epoch+1}/{num_epochs} (K-logit CE, micro-batches)",
        )

        for start in pbar:
            if start < resume_start:
                continue
            mb_full = order[start : start + train_batch_size]
            if start == resume_start and resume_j > 0:
                mb = mb_full[resume_j:]
            else:
                mb = mb_full
            if len(mb) == 0:
                continue

            b_eff = len(mb_full)
            for idx in mb:
                row = train_df.iloc[idx]
                image = Image.open(data_dir / row["image_path"]).convert("RGB")
                image = image.resize((img_size, img_size), Image.BICUBIC)
                y = int(row["answer"])
                logits_k = option_letter_logits(
                    model,
                    processor,
                    image,
                    row,
                    device,
                    letter_token_map,
                    need_grad=True,
                )
                loss = F.cross_entropy(
                    logits_k.unsqueeze(0),
                    torch.tensor([y], device=device, dtype=torch.long),
                )
                (loss / (b_eff * grad_accum_steps)).backward()
                total_steps += 1
                losses.append(float(loss.item()))
                if do_eval and eval_steps > 0 and total_steps % eval_steps == 0:
                    maybe_validate_and_save(epoch, "periodic")
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
                optimizer_step += 1

            resume_start = 0
            resume_j = 0

        if accum_micro > 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            optimizer_step += 1

        snapshot_training_state(
            epoch + 1,
            [out_root],
            samples_into_override=0,
        )
        tail = losses[-min(500, len(losses)) :]
        print(
            f"Epoch {epoch + 1}/{num_epochs} done — "
            f"mean(loss last≤500 steps)={float(np.mean(tail)):.5f}  "
            f"(global sample steps={total_steps})"
        )

    if do_eval and total_steps != last_eval_sample_step:
        maybe_validate_and_save(num_epochs - 1, "end_of_training")

    saved = Path(output_dir) / "adapter"
    saved.parent.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(saved))
    processor.save_pretrained(str(saved))
    snapshot_training_state(
        num_epochs,
        [saved, out_root],
        samples_into_override=0,
    )
    print(f"Saved final LoRA, processor, training_state.json, optimizer.pt to {saved}")

    if do_eval and best_training_samples_seen >= 0:
        print(
            "\n=== Best checkpoint (used for test inference below) ===\n"
            f"  best_val_accuracy: {best_val_acc:.6f}\n"
            f"  best_training_samples_seen: {best_training_samples_seen}\n"
            f"  best_optimizer_step (at that eval): {best_optimizer_step}\n"
            f"  best_epoch: {best_epoch}\n"
            f"  adapter: {best_adapter_dir.resolve()}\n"
            f"  metadata: {out_root / 'best_checkpoint.json'}\n"
            "========================================================\n"
        )
        return best_adapter_dir
    return None


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
            "QLoRA + K-way logit CE on option letters (letter-token MCQ, klogits variant)."
        )
    )
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--model_id", type=str, default=MODEL_ID)
    p.add_argument(
        "--output_dir",
        type=str,
        default="outputs/qlora_lettertoken_klogits",
    )
    p.add_argument(
        "--submission_file",
        type=str,
        default="my_submission_lettertoken_klogits.csv",
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
        help="Samples per micro-batch; gradients scaled so effective batch "
        "≈ train_batch_size * grad_accum_steps (same LR semantics as batch=1).",
    )
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument(
        "--eval_steps",
        type=int,
        default=0,
        help=(
            "If >0, run val.csv accuracy every N training sample steps "
            "(one backward per sample; same as epoch log 'global sample steps'), "
            "not per optimizer step — independent of --grad_accum_steps. "
            "Saves under output_dir/checkpoints/, tracks best to adapter_best. "
            "Requires data_dir/val.csv."
        ),
    )
    p.add_argument(
        "--eval_val_limit",
        type=int,
        default=0,
        help="If >0, only first N rows of val.csv for validation (debug).",
    )
    p.add_argument("--train_limit", type=int, default=0)
    p.add_argument("--val_limit", type=int, default=0, help="If >0, only first N test rows.")
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
        help="Path to saved PEFT adapter (e.g. outputs/qlora_lettertoken_klogits/adapter).",
    )
    p.add_argument(
        "--resume_from",
        type=str,
        default="",
        help=(
            "Directory with LoRA weights + training_state.json (+ optimizer.pt). "
            "Usually output_dir, checkpoints/sample_step_*, or adapter_best. "
            "Requires the same --num_epochs / --seed / batch settings as the original run."
        ),
    )
    args = p.parse_args()
    if args.train_batch_size < 1:
        raise ValueError("--train_batch_size must be >= 1")
    if args.test_only and not (args.checkpoint and str(args.checkpoint).strip()):
        raise ValueError("--test_only requires a non-empty --checkpoint path to the adapter.")
    if args.resume_from and args.test_only:
        raise ValueError("Use either --resume_from (train) or --test_only, not both.")
    if args.resume_from and str(args.checkpoint).strip():
        raise ValueError("--resume_from already loads weights; do not pass --checkpoint.")

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
    elif args.resume_from:
        rp = Path(args.resume_from)
        if not rp.is_dir():
            raise FileNotFoundError(f"--resume_from not a directory: {rp}")
        try:
            processor = AutoProcessor.from_pretrained(str(rp))
        except (OSError, ValueError) as e:
            print(
                f"Failed to load processor from {rp}, falling back to --model_id: {e!r}"
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
    else:
        train_df = pd.read_csv(data_dir / "train.csv")
        test_df = pd.read_csv(data_dir / "test.csv")
        train_df["choices"] = train_df["choices"].apply(json.loads)
        test_df["choices"] = test_df["choices"].apply(json.loads)
        if args.train_limit > 0:
            train_df = train_df.iloc[: args.train_limit].copy()
        if args.val_limit > 0:
            test_df = test_df.iloc[: args.val_limit].copy()

        val_df: pd.DataFrame | None = None
        if args.eval_steps > 0:
            val_path = data_dir / "val.csv"
            if not val_path.is_file():
                raise FileNotFoundError(
                    f"--eval_steps={args.eval_steps} requires val split at {val_path}"
                )
            val_df = pd.read_csv(val_path)
            val_df["choices"] = val_df["choices"].apply(json.loads)

    model, _ = maybe_build_qlora_model(args.model_id)
    if not torch.cuda.is_available():
        model = model.to(torch.device("cpu"))

    if args.test_only:
        print(f"Loading LoRA from {args.checkpoint}")
        model = PeftModel.from_pretrained(model, str(Path(args.checkpoint)))
        model.eval()
    else:
        resume_state: dict | None = None
        resume_adapter_dir: Path | None = None
        resume_optimizer_dir: Path | None = None
        if args.resume_from:
            resume_adapter_dir = Path(args.resume_from).resolve()
            resume_state, resume_optimizer_dir = resolve_resume_state(
                resume_adapter_dir, len(train_df), args
            )
            if int(resume_state.get("format_version", 0)) != TRAINING_STATE_VERSION:
                print(
                    f"Warning: training_state format_version "
                    f"{resume_state.get('format_version')} != {TRAINING_STATE_VERSION}"
                )
            assert_resume_compatible(resume_state, len(train_df), args)

        target_modules = pick_lora_target_modules(model)
        print(f"LoRA target modules: {target_modules}")
        if resume_state is not None:
            print(f"Loading LoRA adapter from {resume_adapter_dir} (resume training)")
            model = PeftModel.from_pretrained(model, str(resume_adapter_dir))
        else:
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

        best_adapter_path = run_train(
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
            val_df=val_df if args.eval_steps > 0 else None,
            eval_steps=args.eval_steps,
            eval_val_limit=args.eval_val_limit,
            resume=resume_state,
            resume_adapter_dir=resume_adapter_dir,
            resume_optimizer_dir=resume_optimizer_dir,
        )

        if best_adapter_path is not None:
            print(
                f"\nReloading best adapter from {best_adapter_path} for test inference "
                "(see best_checkpoint.json / training log for best step).\n"
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            model, _ = maybe_build_qlora_model(args.model_id)
            if not torch.cuda.is_available():
                model = model.to(torch.device("cpu"))
            model = PeftModel.from_pretrained(model, str(best_adapter_path))
        model.eval()

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
