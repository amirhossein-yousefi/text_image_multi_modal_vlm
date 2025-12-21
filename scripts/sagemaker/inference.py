#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""SageMaker inference handler for this repo's VLM + LoRA adapters.

This module is intended to be used with `sagemaker.huggingface.HuggingFaceModel` (or
as `inference.py` inside a custom container). It implements the SageMaker Python
inference toolkit hooks: model_fn / input_fn / predict_fn / output_fn.

Request format (application/json):
  {
    "text": "...",
    "image_base64": "<base64-encoded image bytes>"
  }

Response format (application/json):
  {
    "labels": ["racist", "sexist"],
    "raw_generation": "[...]"
  }

Configuration via environment variables:
  - VLM_MODEL_ID: base model id (e.g., Qwen/Qwen2-VL-2B-Instruct)
  - VLM_ADAPTER_SUBDIR: where the LoRA adapter is inside model_dir (default: lora_adapter)
  - VLM_MAX_NEW_TOKENS: generation length (default: 64)

The model artifact folder (model_dir) is expected to contain `label_map.json` written
by training, plus a LoRA adapter folder (by default `lora_adapter/`).
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Tuple

from PIL import Image


@dataclass
class _Artifacts:
    processor: Any
    model: Any
    class_names: list[str]
    system_prompt: str
    is_qwen_chat: bool


def _env_int(name: str, default: int) -> int:
    v = os.environ.get(name)
    if not v:
        return default
    try:
        return int(v)
    except Exception:
        return default


def _load_label_map(model_dir: Path) -> list[str]:
    p = model_dir / "label_map.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing label map: {p}")
    with open(p, "r", encoding="utf-8") as f:
        obj = json.load(f)
    labels = obj.get("class_names")
    if not isinstance(labels, list) or not all(isinstance(x, str) for x in labels):
        raise ValueError("label_map.json must contain {'class_names': [..]} ")
    return labels


def _build_system_prompt(class_names: list[str]) -> str:
    allowed = class_names
    return (
        "You are a strict multi-label safety classifier. "
        "You MUST respond with a JSON array (no prose) containing zero or more labels selected ONLY from this list:\n"
        + "\n".join(allowed)
        + "\nReturn [] if none apply. Do not include any text outside JSON."
    )


def _decode_image_from_base64(b64: str) -> Image.Image:
    raw = base64.b64decode(b64)
    img = Image.open(BytesIO(raw)).convert("RGB")
    return img


def model_fn(model_dir: str) -> _Artifacts:
    # Imports are inside the function so local tooling can import this file without GPU deps.
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, PaliGemmaForConditionalGeneration

    model_path = Path(model_dir)

    class_names = _load_label_map(model_path)
    system_prompt = _build_system_prompt(class_names)

    base_id = os.environ.get("VLM_MODEL_ID", "").strip()
    if not base_id:
        # Reasonable default matching the repo.
        base_id = "Qwen/Qwen2-VL-2B-Instruct"

    adapter_subdir = os.environ.get("VLM_ADAPTER_SUBDIR", "lora_adapter").strip() or "lora_adapter"
    adapter_dir = model_path / adapter_subdir
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Missing adapter directory: {adapter_dir}")

    # Decide backend by base model id. (Keeps this handler single-file.)
    is_qwen_chat = "qwen" in base_id.lower()

    processor = AutoProcessor.from_pretrained(base_id)

    if is_qwen_chat:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        # For Qwen chat-template flows, left padding makes stripping generation easier.
        if getattr(processor.tokenizer, "pad_token_id", None) is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"
    else:
        model = PaliGemmaForConditionalGeneration.from_pretrained(
            base_id,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        if getattr(processor.tokenizer, "pad_token_id", None) is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        processor.tokenizer.padding_side = "left"

    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()

    return _Artifacts(
        processor=processor,
        model=model,
        class_names=class_names,
        system_prompt=system_prompt,
        is_qwen_chat=is_qwen_chat,
    )


def input_fn(request_body: str, content_type: str) -> Dict[str, Any]:
    if content_type not in ("application/json", "application/json; charset=utf-8"):
        raise ValueError(f"Unsupported content_type: {content_type}")

    obj = json.loads(request_body)
    if not isinstance(obj, dict):
        raise ValueError("Request body must be a JSON object")

    text = str(obj.get("text", "") or "")
    image_b64 = obj.get("image_base64")
    if not image_b64:
        raise ValueError("Request must include 'image_base64'")

    return {"text": text, "image_base64": image_b64}


def _build_qwen_prompt(art: _Artifacts, text: str, image: Image.Image) -> str:
    messages = [
        {"role": "system", "content": art.system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image", "image": image},
            ],
        },
    ]
    return art.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_paligemma_prompt(art: _Artifacts, text: str) -> str:
    # Re-implement minimal PaliGemma prompt so inference doesn't need to import the trainer module.
    image_token = "<image>"
    parts = [image_token, art.system_prompt.strip()]
    if text.strip():
        parts.append("User:\n" + text.strip())
    parts.append("Assistant:")
    return "\n\n".join(parts)


def predict_fn(inputs: Dict[str, Any], model_artifacts: _Artifacts) -> Dict[str, Any]:
    import torch

    text = inputs["text"]
    image = _decode_image_from_base64(inputs["image_base64"])

    max_new_tokens = _env_int("VLM_MAX_NEW_TOKENS", 64)

    if model_artifacts.is_qwen_chat:
        prompt = _build_qwen_prompt(model_artifacts, text, image)
        enc = model_artifacts.processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
        enc = {k: v.to(model_artifacts.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model_artifacts.model.generate(**enc, max_new_tokens=max_new_tokens)
        decoded = model_artifacts.processor.decode(out[0], skip_special_tokens=True)
    else:
        prompt = _build_paligemma_prompt(model_artifacts, text)
        enc = model_artifacts.processor(images=[image], text=[prompt], return_tensors="pt", padding=True)
        enc = {k: v.to(model_artifacts.model.device) for k, v in enc.items()}
        with torch.no_grad():
            out = model_artifacts.model.generate(**enc, max_new_tokens=max_new_tokens)
        decoded = model_artifacts.processor.tokenizer.decode(out[0], skip_special_tokens=True)

    # Best-effort parse: first JSON array in the output.
    labels = []
    try:
        obj = json.loads(decoded.strip())
        if isinstance(obj, list):
            labels = [str(x) for x in obj if isinstance(x, (str, int, float))]
    except Exception:
        # cheap bracket extraction
        import re

        m = re.search(r"\[.*?\]", decoded, flags=re.S)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, list):
                    labels = [str(x) for x in obj if isinstance(x, (str, int, float))]
            except Exception:
                labels = []

    # Keep only allowed labels, preserve order
    allowed = set(model_artifacts.class_names)
    filtered = []
    for a in model_artifacts.class_names:
        if a in allowed and a in labels and a not in filtered:
            filtered.append(a)

    return {"labels": filtered, "raw_generation": decoded}


def output_fn(prediction: Dict[str, Any], accept: str) -> Tuple[str, str]:
    if accept not in ("application/json", "application/json; charset=utf-8", "*/*"):
        raise ValueError(f"Unsupported accept type: {accept}")
    return json.dumps(prediction, ensure_ascii=False), "application/json"
