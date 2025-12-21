#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fast SmolVLM LoRA trainer:
- Fixes Half/Float mismatch in inputs_merger
- Uses TF32 + (optional) FlashAttention 2
- Moves image I/O to the Dataset (parallelizable)
- Shrinks default image token budget (configurable)
- Adds device placement checks + optional torch.compile
"""

import os, json, argparse, ast, re, platform
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,
    BitsAndBytesConfig,
    PreTrainedModel,
    SmolVLMForConditionalGeneration,
)

from peft import LoraConfig, get_peft_model, TaskType

# Optional 4-bit (QLoRA)
try:
    import bitsandbytes as bnb  # noqa: F401
    _BNB = True
except Exception:
    _BNB = False

# ---- Speed-friendly CUDA flags ----
if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# -----------------------
# Utilities
# -----------------------
def ensure_dir(p): os.makedirs(p, exist_ok=True)

def parse_label_list(v):
    if v is None:
        return []
    if isinstance(v, list):
        return [str(x).strip() for x in v if str(x).strip()]
    s = str(v).strip()
    if not s:
        return []
    try:
        maybe = ast.literal_eval(s)
        if isinstance(maybe, (list, tuple)):
            return [str(x).strip() for x in maybe if str(x).strip()]
    except Exception:
        pass
    return [t.strip() for t in s.split(",") if t.strip()]

def canonicalize_labels(raw_labels: List[str], allowed: List[str]) -> List[str]:
    allowed_set = set(allowed)
    selected = [x for x in raw_labels if x in allowed_set]
    uniq = []
    for a in allowed:
        if a in selected and a not in uniq:
            uniq.append(a)
    return uniq

def to_json_label_string(labels: List[str]) -> str:
    return json.dumps(labels, ensure_ascii=False)

def robust_json_extract(text: str) -> Optional[List[str]]:
    try:
        obj = json.loads(text.strip())
        if isinstance(obj, list):
            return obj
    except Exception:
        pass
    m = re.search(r"\[.*?\]", text, flags=re.S)
    if m:
        frag = m.group(0)
        try:
            obj = json.loads(frag)
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
    return None


# -----------------------
# Prompts
# -----------------------
SYSTEM_PROMPT_TPL = (
    "You are a strict multi-label safety classifier. "
    "You MUST respond with a JSON array (no prose) containing zero or more labels selected ONLY from this list:\n"
    "{allowed}\n"
    "Return [] if none apply. Do not include any text outside JSON."
)

def build_messages(system_prompt: str, text: str, num_images: int = 1, target_json: Optional[str] = None):
    user_content = []
    for _ in range(max(0, num_images)):
        user_content.append({"type": "image"})
    if text:
        user_content.append({"type": "text", "text": text})

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if target_json is None:
        return prompt, None
    full = prompt + [{"role": "assistant", "content": [{"type": "text", "text": target_json}]}]
    return prompt, full


# -----------------------
# Dataset (now opens images here, not in collator)
# -----------------------
class VLMJsonRawDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        class_names: List[str],
        system_prompt: str,
        drop_if_missing: bool = True,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.class_names = class_names
        self.system_prompt = system_prompt
        self.items = []

        for _, r in self.df.iterrows():
            text = str(r.get("text", "") or "")
            img_rel = str(r.get("image_path", "") or "")
            img_path = img_rel if os.path.isabs(img_rel) or not image_root else os.path.join(image_root, img_rel)

            if not os.path.exists(img_path):
                if drop_if_missing:
                    continue

            if "labels" in r:
                truth = canonicalize_labels(parse_label_list(r["labels"]), self.class_names)
            else:
                y = int(r.get("label", 0))
                truth = ["harmful"] if y == 1 else []

            target_json = to_json_label_string(truth)
            self.items.append({"text": text, "image_path": img_path, "target_json": target_json})

        if len(self.items) == 0:
            raise RuntimeError("No usable rows found. Check CSV paths and --image_root.")

    def _open_image(self, path: str) -> Image.Image:
        try:
            # Pillow-SIMD (if installed) makes this much faster
            return Image.open(path).convert("RGB")
        except Exception:
            return Image.new("RGB", (32, 32), (255, 255, 255))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        return {
            "text": item["text"],
            "image": self._open_image(item["image_path"]),
            "target_json": item["target_json"],
        }


# -----------------------
# Collator (batch-wise tokenization & image packing)
# -----------------------
class VLMChatCollatorWithProcessor:
    def __init__(self, processor: AutoProcessor, system_prompt: str, max_new_tokens: int = 64,
                 debug_shapes: bool = False):
        self.processor = processor
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.debug_shapes = debug_shapes

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [b["text"] for b in batch]
        images = [b["image"] for b in batch]
        targets = [b["target_json"] for b in batch]

        prompts_txt, fulls_txt = [], []
        for text, target in zip(texts, targets):
            msg_prompt, msg_full = build_messages(self.system_prompt, text, num_images=1, target_json=target)
            p_txt = self.processor.apply_chat_template(msg_prompt, tokenize=False, add_generation_prompt=False)
            f_txt = self.processor.apply_chat_template(msg_full, tokenize=False, add_generation_prompt=False)
            prompts_txt.append(p_txt)
            fulls_txt.append(f_txt)

        full_inputs = self.processor(text=fulls_txt, images=images, return_tensors="pt", padding=True)

        # Compute prompt lengths per-sample (tokenize prompts individually with their image)
        prompt_lens = []
        for p_txt, img in zip(prompts_txt, images):
            p_inputs = self.processor(text=[p_txt], images=[img], return_tensors="pt", padding=False)
            prompt_lens.append(p_inputs["input_ids"].shape[1])

        labels = full_inputs["input_ids"].clone()
        for i, L in enumerate(prompt_lens):
            labels[i, :L] = -100

        batch_out = dict(full_inputs)
        batch_out["labels"] = labels

        if self.debug_shapes:
            pv = batch_out.get("pixel_values", None)
            if pv is not None:
                print(f"[DEBUG] pixel_values shape: {tuple(pv.shape)} (expect 5D: [B, N_img, 3, H, W])")

        return batch_out


# -----------------------
# Evaluation (generation -> JSON -> F1)
# -----------------------
@torch.no_grad()
def evaluate_vlm(
    model: PreTrainedModel,
    processor: AutoProcessor,
    csv_path: str,
    image_root: str,
    class_names: List[str],
    system_prompt: str,
    device: str,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 4,
) -> Dict[str, float]:
    df = pd.read_csv(csv_path)
    allowed = class_names
    y_true_all: List[List[int]] = []
    y_pred_all: List[List[int]] = []

    def load_img(p):
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return Image.new("RGB", (32, 32), (255, 255, 255))

    rows = []
    for _, r in df.iterrows():
        img_rel = str(r.get("image_path", "") or "")
        img_path = img_rel if os.path.isabs(img_rel) or not image_root else os.path.join(image_root, img_rel)
        text = str(r.get("text", "") or "")
        if "labels" in r:
            raw = parse_label_list(r["labels"])
            truth = canonicalize_labels(raw, allowed)
        else:
            truth = ["harmful"] if int(r.get("label", 0)) == 1 else []
        rows.append((text, img_path, truth))

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    model.eval()
    for chunk in chunks(rows, batch_size):
        images = [load_img(p) for _, p, _ in chunk]
        prompts = []
        for text, _, _ in chunk:
            msg_prompt, _ = build_messages(system_prompt, text, num_images=1, target_json=None)
            prompts.append(processor.apply_chat_template(msg_prompt, tokenize=False, add_generation_prompt=True))

        inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=(temperature > 0.0),
        )
        input_len = inputs["input_ids"].shape[1]
        tails = gen[:, input_len:]
        texts = processor.batch_decode(tails, skip_special_tokens=True)

        for (_, _, truth), pred_text in zip(chunk, texts):
            parsed = robust_json_extract(pred_text) or []
            pred = canonicalize_labels([str(x) for x in parsed], allowed)
            y_true_all.append([1 if c in truth else 0 for c in allowed])
            y_pred_all.append([1 if c in pred else 0 for c in allowed])

    y_true = np.array(y_true_all, dtype=np.int32)
    y_pred = np.array(y_pred_all, dtype=np.int32)

    def f1_micro(y_true, y_pred):
        tp = (y_true & y_pred).sum()
        fp = ((1 - y_true) & y_pred).sum()
        fn = (y_true & (1 - y_pred)).sum()
        if tp == 0 and fp == 0 and fn == 0: return 1.0
        prec = tp / max(tp + fp, 1e-12)
        rec = tp / max(tp + fn, 1e-12)
        if (prec + rec) == 0: return 0.0
        return 2 * prec * rec / (prec + rec)

    def f1_macro(y_true, y_pred):
        f1s = []
        for j in range(y_true.shape[1]):
            tp = ((y_true[:, j] == 1) & (y_pred[:, j] == 1)).sum()
            fp = ((y_true[:, j] == 0) & (y_pred[:, j] == 1)).sum()
            fn = ((y_true[:, j] == 1) & (y_pred[:, j] == 0)).sum()
            if tp == 0 and fp == 0 and fn == 0:
                f1s.append(1.0)
                continue
            prec = tp / max(tp + fp, 1e-12)
            rec = tp / max(tp + fn, 1e-12)
            f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
        return float(np.mean(f1s))

    return {
        "f1_micro": float(f1_micro(y_true, y_pred)),
        "f1_macro": float(f1_macro(y_true, y_pred)),
    }


# -----------------------
# SmolVLM dtype patch (fixes Half vs Float index_put in inputs_merger)
# -----------------------
def patch_smolvlm_inputs_merger_dtype(model: SmolVLMForConditionalGeneration) -> bool:
    base = getattr(model, "model", None)
    if base is None or not hasattr(base, "inputs_merger"):
        return False

    orig_fn = base.inputs_merger

    def wrapped_inputs_merger(*args, **kwargs):
        if "image_hidden_states" in kwargs and "inputs_embeds" in kwargs:
            ihs = kwargs["image_hidden_states"]
            ie = kwargs["inputs_embeds"]
            try:
                if ihs is not None and ie is not None and ihs.dtype != ie.dtype:
                    kwargs["image_hidden_states"] = ihs.to(ie.dtype)
            except Exception:
                pass
        return orig_fn(*args, **kwargs)

    base.inputs_merger = wrapped_inputs_merger
    return True


# -----------------------
# Args & Main
# -----------------------
def build_args():
    ap = argparse.ArgumentParser("Fast LoRA on SmolVLM for JSON-label classification")

    # Data
    ap.add_argument("--train_csv")
    ap.add_argument("--val_csv")
    ap.add_argument("--test_csv", default="")
    ap.add_argument("--image_root", default="")
    ap.add_argument("--class_names",
                    help="Comma-separated labels for multi-label; if omitted in CSV, binary 'label' is used.")

    # Model
    ap.add_argument("--model_id", default="HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    ap.add_argument("--out_dir", default="runs/smolvlm_lora")
    ap.add_argument("--seed", type=int, default=42)

    # LoRA
    ap.add_argument("--use_lora", action="store_true", default=True)
    ap.add_argument("--lora_r", type=int, default=1)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Optimization
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=1)  # smaller == fewer micro-steps per "it"
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--eval_strategy", default="epoch", choices=["no", "epoch", "steps"])
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--patience", type=int, default=2)

    # Precision / memory
    ap.add_argument("--fp16", action="store_true", default=True)
    ap.add_argument("--bf16", action="store_true", default=False, help="Prefer on Ampere+ for speed/stability")
    ap.add_argument("--load_in_4bit", action="store_true", default=True, help="Enable QLoRA (requires bitsandbytes).")

    # Image token budget (smaller = faster)
    ap.add_argument("--min_image_tokens", type=int, default=32)
    ap.add_argument("--max_image_tokens", type=int, default=128)

    # Extras
    ap.add_argument("--enable_flash_attn", action="store_true", default=True)
    ap.add_argument("--compile", action="store_true", default=False)
    ap.add_argument("--debug_shapes", action="store_true")

    # Defaults for mmhs150k if not provided
    args = ap.parse_args()
    args.train_csv = args.train_csv or "data/mmhs150k/train.csv"
    args.val_csv = args.val_csv or "data/mmhs150k/val.csv"
    args.test_csv = args.test_csv or "data/mmhs150k/test.csv"
    args.image_root = args.image_root or "data/mmhs150k"
    args.class_names = args.class_names or "racist,sexist,homophobe,religion,otherhate"
    return args


def _percent_params_on_cuda(model) -> float:
    total = 0
    cuda = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.device.type == "cuda":
            cuda += n
    return float(cuda) / max(total, 1) * 100.0


def main():
    args = build_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    if not torch.cuda.is_available():
        print("[WARN] CUDA not available — training will be extremely slow. "
              "Consider using a GPU runtime or WSL2/Linux if on Windows.")

    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    system_prompt = SYSTEM_PROMPT_TPL.format(allowed=class_names)

    # Processor
    VIS_PATCH = 8  # pixel-per-token scale reference used by SmolVLM
    min_pixels = args.min_image_tokens * (VIS_PATCH * VIS_PATCH)
    max_pixels = args.max_image_tokens * (VIS_PATCH * VIS_PATCH)
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Model load (optionally 4-bit)
    quant_config = None
    if args.load_in_4bit:
        if not _BNB:
            raise RuntimeError("bitsandbytes not installed; install it or disable --load_in_4bit")
        compute_dtype = torch.bfloat16 if args.bf16 else torch.float16
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    extra = {}
    # if args.enable_flash_attn:
    #     # Works if FA2 is installed and transformers recognizes the kwarg.
    #     extra["attn_implementation"] = "flash_attention_2"

    try:
        model = SmolVLMForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            device_map="auto",
            quantization_config=quant_config,
            **extra,
        )
    except TypeError:
        # Transformers too old for attn_implementation kwarg — retry without it
        if "attn_implementation" in extra:
            extra.pop("attn_implementation", None)
        model = SmolVLMForConditionalGeneration.from_pretrained(
            args.model_id,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            device_map="auto",
            quantization_config=quant_config,
            **extra,
        )

    # Patch dtype mismatch in inputs_merger
    patched = patch_smolvlm_inputs_merger_dtype(model)
    if not patched:
        print("[WARN] Could not patch inputs_merger; if you hit dtype mismatch, update transformers or this patch.")

    # Sanity check: are we on CUDA?
    try:
        pct_cuda = _percent_params_on_cuda(model)
        print(f"[INFO] ~{pct_cuda:.1f}% of parameters are on CUDA.")
        if pct_cuda < 50.0 and args.load_in_4bit:
            print("[WARN] Most weights appear on CPU. On Windows, 4-bit bitsandbytes often falls back to CPU.\n"
                  "      Use WSL2/Linux, or run with --load_in_4bit false (needs more VRAM), or pick a smaller model.")
    except Exception:
        pass

    # Optional: torch.compile for extra speed
    if args.compile and torch.cuda.is_available():
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("[INFO] torch.compile enabled.")
        except Exception as e:
            print("[WARN] torch.compile disabled (", e, ")")

    # LoRA
    if args.use_lora:
        tmods = [t.strip() for t in args.lora_target_modules.split(",") if t.strip()]
        lconf = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=tmods,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lconf)
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    # Datasets
    train_ds = VLMJsonRawDataset(args.train_csv, args.image_root, class_names, system_prompt)
    val_ds = VLMJsonRawDataset(args.val_csv, args.image_root, class_names, system_prompt)
    test_ds = VLMJsonRawDataset(args.test_csv, args.image_root, class_names, system_prompt) if (
        args.test_csv and os.path.exists(args.test_csv)) else None

    # Collator
    collator = VLMChatCollatorWithProcessor(
        processor, system_prompt, max_new_tokens=args.max_new_tokens if hasattr(args, "max_new_tokens") else 64,
        debug_shapes=args.debug_shapes
    )

    # Windows: multi-worker often slows things down; elsewhere 4 workers helps
    is_windows = platform.system().lower().startswith("win")
    num_workers = 0 if is_windows else 4

    # Trainer
    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.eval_strategy,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=args.save_total_limit,
        fp16=bool(args.fp16 and not args.bf16),
        bf16=bool(args.bf16),
        tf32=True,  # Speed on Ampere+
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.out_dir, "logs"),
        dataloader_num_workers=num_workers,
        dataloader_persistent_workers=(num_workers > 0),
        remove_unused_columns=False,
        load_best_model_at_end=(args.eval_strategy != "no"),
        optim="paged_adamw_8bit" if args.load_in_4bit else ("adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch"),
    )

    callbacks = [EarlyStoppingCallback(early_stopping_patience=args.patience)]

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds if args.eval_strategy != "no" else None,
        data_collator=collator,
        callbacks=callbacks,
    )

    train_output = trainer.train()
    print(train_output)
    try:
        with open(os.path.join(args.out_dir, "train_metrics.json"), "w") as f:
            json.dump(train_output.metrics, f, indent=2)
    except Exception:
        pass
    print("Training:", train_output)

    # Save artifacts
    trainer.save_model(args.out_dir)
    with open(os.path.join(args.out_dir, "label_map.json"), "w") as f:
        json.dump({"class_names": class_names}, f, indent=2)

    # If PEFT, save adapter cleanly for vLLM
    lora_dir = os.path.join(args.out_dir, "lora_adapter")
    try:
        model.save_pretrained(lora_dir)
        print("Saved LoRA adapter ->", lora_dir)
    except Exception as e:
        print("Note: could not save adapter separately:", e)

    # Evaluate (validation)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.eval_strategy != "no":
        eval_metrics = evaluate_vlm(
            model=model, processor=processor,
            csv_path=args.val_csv, image_root=args.image_root,
            class_names=class_names, system_prompt=system_prompt,
            device=device,
            max_new_tokens=64, temperature=0.0, top_p=1.0,
            batch_size=args.per_device_eval_batch_size
        )
        with open(os.path.join(args.out_dir, "val_metrics.json"), "w") as f:
            json.dump(eval_metrics, f, indent=2)
        print("Validation:", eval_metrics)

    # Optional test
    if (args.eval_strategy != "no") and (test_ds is not None):
        test_metrics = evaluate_vlm(
            model=model, processor=processor,
            csv_path=args.test_csv, image_root=args.image_root,
            class_names=class_names, system_prompt=system_prompt,
            device=device,
            max_new_tokens=64, temperature=0.0, top_p=1.0,
            batch_size=args.per_device_eval_batch_size
        )
        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        print("Test:", test_metrics)


if __name__ == "__main__":
    main()
