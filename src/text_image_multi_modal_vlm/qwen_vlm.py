


#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
VLM (Qwen2-VL) + LoRA training on CSV (text, image, labels)
- Multi-label or binary classification via *generation* (model outputs JSON array of labels)
- LoRA (or QLoRA) with PEFT
- Robust label canonicalization & JSON parsing
- Validation & test metrics (micro/macro F1)
- Saves LoRA adapter for vLLM serving

IMPORTANT: We DO NOT pre-tokenize images per-sample.
All processor calls happen inside the collator on the whole batch,
so pixel_values has the expected 5D shape [B, N_img, 3, H, W] and image_grid_thw matches.
"""

import os, json, argparse, ast, re
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
    set_seed,
    EarlyStoppingCallback,PaliGemmaForConditionalGeneration
)

# ---- PEFT / LoRA ----
from peft import LoraConfig, get_peft_model, TaskType

# Optional 4-bit (QLoRA)
try:
    import bitsandbytes as bnb  # noqa: F401
    _BNB = True
except Exception:
    _BNB = False


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
    """Keep only allowed labels, preserve allowed-order."""
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
    """Best-effort: extract first JSON array from text."""
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

def build_messages(system_prompt: str, text: str, image: Image.Image, target_json: Optional[str] = None):
    """Return (messages_prompt_only, messages_full_if_target)."""
    user_content = []
    if text:
        user_content.append({"type": "text", "text": text})
    user_content.append({"type": "image", "image": image})
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    if target_json is None:
        return prompt, None
    full = prompt + [{"role": "assistant", "content": [{"type": "text", "text": target_json}]}]
    return prompt, full


# -----------------------
# Dataset (NO per-sample tokenization)
# -----------------------
class VLMJsonRawDataset(Dataset):
    """
    Returns raw items: text, image path, and target JSON.
    We leave tokenization + image packing to the collator (batch-wise).
    """
    def __init__(
        self,
        csv_path: str,
        image_root: str,
        class_names: List[str],
        drop_if_missing: bool = True,
    ):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.class_names = class_names
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

    def __len__(self): return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.items[idx]


# -----------------------
# Collator (the IMPORTANT part that fixes your error)
# -----------------------
class VLMChatCollatorWithProcessor:
    """
    - Builds chat prompts & full texts for the whole batch.
    - Calls processor(text=..., images=..., return_tensors='pt', padding=True) ONCE for full inputs.
    - Also computes prompt lengths (per-sample) by tokenizing each prompt individually
      (cheap vs forward pass), then creates labels by masking prompt tokens to -100.
    Result: pixel_values -> [B, N_img, 3, H, W], image_grid_thw matches, avoiding the split_with_sizes error.
    """
    def __init__(self, processor: AutoProcessor, system_prompt: str, max_new_tokens: int = 64, debug_shapes: bool = False):
        self.processor = processor
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.debug_shapes = debug_shapes

    def _open_image(self, path: str) -> Image.Image:
        try:
            return Image.open(path).convert("RGB")
        except Exception:
            # tiny fallback so shapes are valid
            return Image.new("RGB", (32, 32), (255, 255, 255))

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [b["text"] for b in batch]
        images = [self._open_image(b["image_path"]) for b in batch]
        targets = [b["target_json"] for b in batch]

        prompts_txt, fulls_txt = [], []
        for text, img, target in zip(texts, images, targets):
            msg_prompt, msg_full = build_messages(self.system_prompt, text, img, target)
            p_txt = self.processor.apply_chat_template(msg_prompt, tokenize=False, add_generation_prompt=False)
            f_txt = self.processor.apply_chat_template(msg_full, tokenize=False, add_generation_prompt=False)
            prompts_txt.append(p_txt)
            fulls_txt.append(f_txt)

        # Tokenize full batch ONCE (this supplies pixel_values, image_grid_thw, etc.)
        full_inputs = self.processor(text=fulls_txt, images=images, return_tensors="pt", padding=True)

        # Compute prompt lengths per-sample (tokenize prompts individually w/ images)
        prompt_lens = []
        for p_txt, img in zip(prompts_txt, images):
            p_inputs = self.processor(text=[p_txt], images=[img], return_tensors="pt", padding=False)
            prompt_lens.append(p_inputs["input_ids"].shape[1])

        labels = full_inputs["input_ids"].clone()
        attn = full_inputs.get("attention_mask", None)

        for i, L in enumerate(prompt_lens):
            if attn is not None:
                seq_len = labels.shape[1]
                real_len = int(attn[i].sum().item())
                start = seq_len - real_len  # left-pad prefix (0 if right-padded)
            else:
                start = 0
            labels[i, start:start + L] = -100

        # Ensure pads never contribute to loss
        if attn is not None:
            labels[attn == 0] = -100
        else:
            pad_id = getattr(self.processor.tokenizer, "pad_token_id", None)
            if pad_id is not None:
                labels[labels == pad_id] = -100

        batch_out = dict(full_inputs)
        batch_out["labels"] = labels

        if self.debug_shapes:
            pv = batch_out.get("pixel_values", None)
            if pv is not None:
                print(f"[DEBUG] pixel_values shape: {tuple(pv.shape)} (expect 5D: [B, N_img, 3, H, W])")
            thw = batch_out.get("image_grid_thw", None)
            if thw is not None:
                print(f"[DEBUG] image_grid_thw shape: {tuple(thw.shape)}")

        return batch_out


# -----------------------
# Evaluation (generation -> JSON -> F1)
# -----------------------
@torch.no_grad()
def evaluate_vlm(
    model: Qwen2VLForConditionalGeneration,
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
            yield lst[i:i+n]

    model.eval()
    for chunk in chunks(rows, batch_size):
        images = [load_img(p) for _, p, _ in chunk]
        prompts = []
        for text, _, _ in chunk:
            msg_prompt, _ = build_messages(system_prompt, text, images[len(prompts)], None)
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
        # Extract generated tail:
        input_len = inputs["input_ids"].shape[1]
        tails = gen[:, input_len:]
        texts = processor.batch_decode(tails, skip_special_tokens=True)

        for (_, _, truth), pred_text in zip(chunk, texts):
            parsed = robust_json_extract(pred_text) or []
            pred = canonicalize_labels([str(x) for x in parsed], allowed)
            y_true_all.append([1 if c in truth else 0 for c in allowed])
            y_pred_all.append([1 if c in pred else 0 for c in allowed])

    # Metrics
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
# Evaluation on a Dataset (for Trainer hooks)
# -----------------------
@torch.no_grad()
def evaluate_vlm_on_dataset(
    model: Qwen2VLForConditionalGeneration,
    processor: AutoProcessor,
    dataset: "VLMJsonRawDataset",
    class_names: List[str],
    system_prompt: str,
    device: torch.device,
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_p: float = 1.0,
    batch_size: int = 4,
) -> Dict[str, float]:
    allowed = class_names
    y_true_all: List[List[int]] = []
    y_pred_all: List[List[int]] = []

    def load_img(p):
        try:
            return Image.open(p).convert("RGB")
        except Exception:
            return Image.new("RGB", (32, 32), (255, 255, 255))

    # Build rows directly from dataset
    rows: List[Tuple[str, str, List[str]]] = []
    for it in dataset.items:
        text = it["text"]
        img_path = it["image_path"]
        truth_list = robust_json_extract(it["target_json"]) or []
        truth = canonicalize_labels([str(x) for x in truth_list], allowed)
        rows.append((text, img_path, truth))

    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    model.eval()
    for chunk in chunks(rows, batch_size):
        images = [load_img(p) for _, p, _ in chunk]
        prompts = []
        for text, _, _ in chunk:
            msg_prompt, _ = build_messages(system_prompt, text, images[len(prompts)], None)
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
        return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    def f1_macro(y_true, y_pred):
        f1s = []
        for j in range(y_true.shape[1]):
            tp = ((y_true[:, j] == 1) & (y_pred[:, j] == 1)).sum()
            fp = ((y_true[:, j] == 0) & (y_pred[:, j] == 1)).sum()
            fn = ((y_true[:, j] == 1) & (y_pred[:, j] == 0)).sum()
            if tp == 0 and fp == 0 and fn == 0:
                f1s.append(1.0)
            else:
                prec = tp / max(tp + fp, 1e-12)
                rec = tp / max(tp + fn, 1e-12)
                f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
        return float(np.mean(f1s))

    subset_acc = float((y_true == y_pred).all(axis=1).mean())
    hamming_loss = float((y_true != y_pred).mean())

    return {
        "f1_micro": float(f1_micro(y_true, y_pred)),
        "f1_macro": float(f1_macro(y_true, y_pred)),
        "subset_accuracy": subset_acc,
        "hamming_loss": hamming_loss,
    }

# -----------------------
# Trainer that adds generation-based classification metrics during eval
# -----------------------
# -----------------------
# ### NEW: Trainer that calls model.generate() at eval to produce predictions
# -----------------------
class GenWithGenerateTrainer(Trainer):
    def __init__(self, *args, processor: AutoProcessor, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.pad_token_id = (
            getattr(processor.tokenizer, "pad_token_id", None)
            or getattr(self.model.config, "pad_token_id", None) or 0
        )

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        """
        Returns (loss, pred_ids, label_ids), where pred_ids are generated tails
        LEFT-PAD safe for decoder-only models and FIXED-WIDTH across batches.
        """
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            labels = inputs.get("labels", None)

            # ----- forward for eval loss -----
            loss = None
            if labels is not None:
                outputs = model(**inputs)
                loss = outputs.loss.detach()

            # ----- build LEFT-PADDED prompt for generation -----
            inp_ids = inputs["input_ids"]  # [B, T]
            attn = inputs.get("attention_mask", None)  # [B, T] (1 for tokens)
            B, T = inp_ids.shape

            if labels is None:
                labels = inp_ids.new_full(inp_ids.shape, fill_value=-100)

            prompt_lens, start_idxs = [], []
            for i in range(B):
                if attn is not None:
                    real_len = int(attn[i].sum().item())
                    start = T - real_len
                    tgt_len = int(((labels[i] != -100) & (attn[i] == 1)).sum().item())
                else:
                    real_len = T
                    start = 0
                    tgt_len = int((labels[i] != -100).sum().item())
                L = real_len - tgt_len
                prompt_lens.append(L)
                start_idxs.append(start)

            maxL = max(max(prompt_lens), 1)
            pad_id = (
                    getattr(self.processor.tokenizer, "pad_token_id", None)
                    or getattr(self.model.config, "pad_token_id", None) or 0
            )
            prompt_ids = inp_ids.new_full((B, maxL), fill_value=pad_id)
            prompt_attn = attn.new_zeros((B, maxL)) if attn is not None else None
            for i, (L, start) in enumerate(zip(prompt_lens, start_idxs)):
                if L > 0:
                    src = inp_ids[i, start:start + L]
                    prompt_ids[i, maxL - L:maxL] = src
                    if prompt_attn is not None:
                        prompt_attn[i, maxL - L:maxL] = 1

            gen_inputs = {k: v for k, v in inputs.items() if k not in ("input_ids", "attention_mask", "labels")}
            gen_inputs["input_ids"] = prompt_ids
            if prompt_attn is not None:
                gen_inputs["attention_mask"] = prompt_attn

            # ----- generation hyperparams -----
            gen_kwargs = dict(
                max_new_tokens=getattr(self.args, "max_new_tokens", 64),
                temperature=getattr(self.args, "temperature", 0.0),
                top_p=getattr(self.args, "top_p", 1.0),
                do_sample=(getattr(self.args, "temperature", 0.0) > 0.0),
                use_cache=True,
            )
            generated = model.generate(**gen_inputs, **gen_kwargs)

            # ----- strip prompt (same for everyone = maxL) -----
            tails = generated[:, maxL:]  # [B, variable]

            # ----- FIXED WIDTH across batches -----
            gen_len = int(getattr(self.args, "eval_gen_max_length", getattr(self.args, "max_new_tokens", 64)))
            pred_ids = inp_ids.new_full((B, gen_len), fill_value=pad_id)
            if tails.numel() > 0:
                copy_len = min(gen_len, tails.shape[1])
                pred_ids[:, :copy_len] = tails[:, :copy_len]

        return (loss, pred_ids, labels)


# -----------------------
# ### NEW: compute_metrics for Trainer (works with our generated pred_ids)
# -----------------------
def make_compute_metrics(processor: AutoProcessor, class_names: List[str]):
    allowed = class_names
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)

    def _to_multi_hot(json_text: str) -> List[int]:
        arr = robust_json_extract(json_text) or []
        lab = canonicalize_labels([str(x) for x in arr], allowed)
        return [1 if c in lab else 0 for c in allowed]

    def _sanitize_row(row) -> List[int]:
        arr = np.asarray(row)
        if arr.dtype.kind == "f":                     # floats -> drop NaN then cast
            arr = arr[~np.isnan(arr)]
            arr = arr.astype(np.int64, copy=False)
        else:
            arr = arr.astype(np.int64, copy=False)
        ids = []
        for v in arr.ravel().tolist():
            if v < 0:                                 # drop negatives (e.g., -100 paddings)
                continue
            if pad_id is not None and v == pad_id:    # drop PAD tokens
                continue
            ids.append(int(v))
        return ids

    def _decode_rows(preds_like) -> List[str]:
        # unwrap common containers (tuple/list from HF)
        if isinstance(preds_like, (tuple, list)) and len(preds_like) > 0:
            preds_like = preds_like[0]
        preds_like = np.asarray(preds_like, dtype=object)  # allow ragged rows safely
        texts = []
        for i in range(preds_like.shape[0]):
            row = np.asarray(preds_like[i])
            ids = _sanitize_row(row)
            texts.append(processor.tokenizer.decode(ids, skip_special_tokens=True))
        return texts

    def _f1_micro(y_true, y_pred):
        tp = (y_true & y_pred).sum()
        fp = ((1 - y_true) & y_pred).sum()
        fn = (y_true & (1 - y_pred)).sum()
        if tp == 0 and fp == 0 and fn == 0: return 1.0
        prec = tp / max(tp + fp, 1e-12)
        rec  = tp / max(tp + fn, 1e-12)
        return 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    def _f1_macro(y_true, y_pred):
        f1s = []
        for j in range(y_true.shape[1]):
            tp = ((y_true[:, j] == 1) & (y_pred[:, j] == 1)).sum()
            fp = ((y_true[:, j] == 0) & (y_pred[:, j] == 1)).sum()
            fn = ((y_true[:, j] == 1) & (y_pred[:, j] == 0)).sum()
            if tp == 0 and fp == 0 and fn == 0:
                f1s.append(1.0)
            else:
                prec = tp / max(tp + fp, 1e-12)
                rec  = tp / max(tp + fn, 1e-12)
                f1s.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
        return float(np.mean(f1s))

    def compute_metrics(eval_pred):
        pred_like = eval_pred.predictions
        label_like = eval_pred.label_ids

        pred_texts = _decode_rows(pred_like)

        # Labels: keep only tokens != -100, and non-negative for safety
        label_arr = np.asarray(label_like, dtype=object)
        gold_texts = []
        for i in range(label_arr.shape[0]):
            row = np.asarray(label_arr[i])
            if row.dtype.kind == "f":
                row = row[~np.isnan(row)]
                row = row.astype(np.int64, copy=False)
            else:
                row = row.astype(np.int64, copy=False)
            ids = [int(v) for v in row.ravel().tolist() if v != -100 and v >= 0]
            gold_texts.append(processor.tokenizer.decode(ids, skip_special_tokens=True))

        y_true = np.array([_to_multi_hot(t) for t in gold_texts], dtype=np.int32)
        y_pred = np.array([_to_multi_hot(p) for p in pred_texts], dtype=np.int32)

        subset_acc = float((y_true == y_pred).all(axis=1).mean())
        hamming_loss = float((y_true != y_pred).mean())

        return {
            "f1_micro": float(_f1_micro(y_true, y_pred)),
            "f1_macro": float(_f1_macro(y_true, y_pred)),
            "subset_accuracy": subset_acc,
            "hamming_loss": hamming_loss,
        }

    return compute_metrics


# -----------------------
# Args & Main
# -----------------------
def build_args():
    ap = argparse.ArgumentParser("Train LoRA on Qwen2-VL for JSON-label classification")

    # Data
    ap.add_argument("--train_csv",)
    ap.add_argument("--val_csv", )
    ap.add_argument("--test_csv", default="")
    ap.add_argument("--image_root", default="")
    ap.add_argument("--class_names",
                    help="Comma-separated labels for multi-label; if omitted in CSV, binary 'label' is used.")

    # Model
    ap.add_argument("--model_id", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--out_dir", default="runs/qwen2vl_lora")
    ap.add_argument("--seed", type=int, default=42)

    # LoRA
    ap.add_argument("--use_lora", action="store_true", default=True)
    ap.add_argument("--lora_r", type=int, default=4)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_target_modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")

    # Optimization
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--num_train_epochs", type=int, default=1)
    ##16k steps for 4 batch size
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)  # VLMs are big; keep small
    ap.add_argument("--per_device_eval_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--eval_strategy", default="steps", choices=["epoch", "steps"])
    ap.add_argument("--save_total_limit", type=int, default=2)
    ap.add_argument("--patience", type=int, default=2)

    # Precision / memory
    ap.add_argument("--fp16", action="store_true", default=False)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--load_in_4bit", action="store_true", default=True,
                    help="Enable QLoRA (requires bitsandbytes).")

    # Generation (for eval)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--debug_shapes", action="store_true", help="Print shapes of pixel_values and image_grid_thw")
    args = ap.parse_args()

    # Demo defaults (used when flags are omitted). This keeps the repo runnable out-of-the-box
    # while allowing callers (e.g., SageMaker) to pass real paths.
    if not args.train_csv:
        args.train_csv = "data/mmhs150k/train.csv"
    if not args.val_csv:
        args.val_csv = "data/mmhs150k/val.csv"
    if not args.test_csv:
        args.test_csv = "data/mmhs150k/test.csv"
    if args.image_root is None:
        args.image_root = ""
    if args.image_root == "":
        args.image_root = "data/mmhs150k"
    if not args.class_names:
        args.class_names = "racist,sexist,homophobe,religion,otherhate"
    return args

def quick_eval_sanity(model, processor, dataset, class_names, system_prompt, device, n=2):
    model.eval()
    n = min(n, len(dataset))
    samples = [dataset[i] for i in np.linspace(0, len(dataset) - 1, n, dtype=int)]
    images, prompts, gold = [], [], []
    for it in samples:
        img = Image.open(it["image_path"]).convert("RGB") if os.path.exists(it["image_path"]) else Image.new("RGB", (32, 32), (255, 255, 255))
        msg_prompt, _ = build_messages(system_prompt, it["text"], img, None)
        ptxt = processor.apply_chat_template(msg_prompt, tokenize=False, add_generation_prompt=True)
        images.append(img)
        prompts.append(ptxt)
        gold.append(it["target_json"])

    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=32, temperature=0.1, top_p=1.0)
    tails = gen[:, inputs["input_ids"].shape[1]:]
    preds = processor.batch_decode(tails, skip_special_tokens=True)
    print("\n[Sanity check]")
    for i, (p, g) in enumerate(zip(preds, gold)):
        print(f"  ex#{i}  pred={p}   gold={g}")




def main():
    args = build_args()
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    system_prompt = SYSTEM_PROMPT_TPL.format(allowed=class_names)
    # Limit image resolution → fewer visual tokens → much lower memory
    VIS_PATCH = 28  # Qwen2-VL docs use 28*28 as the pixel-per-token scale
    min_pixels = 64 * (VIS_PATCH * VIS_PATCH)  # 64 tokens / image
    max_pixels = 256 * (VIS_PATCH * VIS_PATCH)  # 256 tokens / image
    # Processor
    processor = AutoProcessor.from_pretrained(
        args.model_id,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # Ensure PAD token exists
    if getattr(processor.tokenizer, "pad_token_id", None) is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # Prefer LEFT padding for decoder-only generation
    processor.tokenizer.padding_side = "left"

    # Model load (optionally 4-bit)
    quant_kwargs = {}
    if args.load_in_4bit:
        if not _BNB:
            raise RuntimeError("bitsandbytes not installed; install it or disable --load_in_4bit")
        quant_kwargs = dict(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
        device_map="auto",
        **quant_kwargs,
    )
    # model = PaliGemmaForConditionalGeneration.from_pretrained(        args.model_id,
    #     torch_dtype=torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32),
    #     device_map="auto",
    #     **quant_kwargs,)

    model.config.use_cache = False

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
    train_ds = VLMJsonRawDataset(args.train_csv, args.image_root, class_names)
    val_ds   = VLMJsonRawDataset(args.val_csv,   args.image_root, class_names)
    test_ds  = VLMJsonRawDataset(args.test_csv,  args.image_root, class_names) if (args.test_csv and os.path.exists(args.test_csv)) else None

    # Collator — all tokenization & image packing happens here
    collator = VLMChatCollatorWithProcessor(processor, system_prompt, max_new_tokens=args.max_new_tokens, debug_shapes=args.debug_shapes)

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
        save_steps=8000,
        eval_steps=8000,
        save_strategy=args.eval_strategy,
        metric_for_best_model="eval_f1_macro",   # ### NEW: pick best by validation F1-macro
        greater_is_better=True,                  # ### NEW: higher F1 is better
        save_total_limit=args.save_total_limit,
        fp16=args.fp16,
        bf16=args.bf16,
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.out_dir, "logs"),
        dataloader_num_workers=4,
        remove_unused_columns=False,
        load_best_model_at_end=True,
        # gradient_checkpointing=True,
        optim="paged_adamw_8bit"
    )

    setattr(training_args, "max_new_tokens", args.max_new_tokens)
    setattr(training_args, "temperature", args.temperature)
    setattr(training_args, "top_p", args.top_p)

    trainer = GenWithGenerateTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
        processor=processor,  # ### NEW
        compute_metrics=make_compute_metrics(processor, class_names),  # ### NEW
    )

    # --- call it before trainer.train()
    device = model.device
    quick_eval_sanity(model, processor, val_ds, class_names, system_prompt, device, n=2)
    trainer.train()

    # Save artifacts
    trainer.save_model(args.out_dir)  # if PEFT: saves adapter refs; otherwise full
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
    # eval_metrics = evaluate_vlm(
    #     model=model, processor=processor,
    #     csv_path=args.val_csv, image_root=args.image_root,
    #     class_names=class_names, system_prompt=system_prompt,
    #     device=device, max_new_tokens=args.max_new_tokens,
    #     temperature=args.temperature, top_p=args.top_p,
    #     batch_size=args.per_device_eval_batch_size
    # )
    val_metrics = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="eval")

    with open(os.path.join(args.out_dir, "val_metrics.json"), "w") as f:
        json.dump(val_metrics, f, indent=2)
    print("Validation:", val_metrics)

    # Optional test
    if test_ds is not None:
        test_metrics = evaluate_vlm(
            model=model, processor=processor,
            csv_path=args.test_csv, image_root=args.image_root,
            class_names=class_names, system_prompt=system_prompt,
            device=device, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            batch_size=args.per_device_eval_batch_size
        )
        with open(os.path.join(args.out_dir, "test_metrics.json"), "w") as f:
            json.dump(test_metrics, f, indent=2)
        print("Test:", test_metrics)


if __name__ == "__main__":
    main()
