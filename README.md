# Multi-Modal Vision-Language Models for Hateful Content Classification

Fine-tuning Vision–Language Models (VLMs) with LoRA/QLoRA for **multi-label hateful content detection** on paired **text + image** data.

This project uses **generative classification**: instead of training a dedicated classification head, the model generates a strict JSON array of labels (e.g., `["racist", "sexist"]`).

---

## Table of Contents

- [Overview](#overview)
- [Models](#models)
- [Hardware Used](#hardware-used)
- [Repository Layout](#repository-layout)
- [Setup](#setup)
- [Dataset](#dataset)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Evaluation & Results](#evaluation--results)
- [Inference](#inference)
- [SageMaker](#sagemaker)
- [Artifacts](#artifacts)
- [License](#license)

---

## Overview

Given an image and its associated text, the model outputs a JSON array containing zero or more labels from a fixed label set.

High-level flow:

1. Build a strict “return JSON only” prompt listing allowed labels.
2. Feed (text + image) to the VLM.
3. Generate a short response.
4. Parse the first JSON array found (best-effort JSON extraction).
5. Convert labels into multi-hot predictions and compute multi-label metrics.

---

## Models

Implemented training + evaluation scripts:

- **Qwen2-VL (2B)**: chat-template prompts; batch-wise processor call to produce correct multi-image tensor shapes.
- **PaliGemma 2 (3B)**: plain-string prompts; uses `suffix=...` for label construction.
- **SmolVLM**: lightweight baseline.

Entry points:

- Package modules: `python -m text_image_multi_modal_vlm.qwen_vlm` (or `.paligemma`, `.smol_vlm`)
- Root wrappers (thin shims): `python qwen_vlm.py` (or `paligemma.py`, `smol_vlm.py`)

---

## Hardware Used

The runs stored under `runs/` were trained on the following hardware:

| Model | Device | Platform | Notes |
|------|--------|----------|------|
| **PaliGemma 2 + LoRA** | NVIDIA A100 | Google Colab | Mixed precision used (`--bf16`). |
| **Qwen2-VL + LoRA/QLoRA** | NVIDIA GeForce RTX 3080 Laptop GPU (16GB) | Local Windows | NVIDIA driver 581.57, CUDA 13.0 (per `nvidia-smi`). |

---

## Repository Layout

The main implementations live in `src/` (root-level scripts are wrappers that import from `src`).

```
text_image_multi_modal_vlm/
├── paligemma.py
├── qwen_vlm.py
├── smol_vlm.py
├── src/
│   └── text_image_multi_modal_vlm/
│       ├── paligemma.py
│       ├── qwen_vlm.py
│       └── smol_vlm.py
├── scripts/
│   └── train.py
├── data/
│   ├── _raw_mmhs/
│   └── mmhs150k/
└── runs/
```

---

## Setup

### Requirements

- Python 3.8+
- A CUDA-capable GPU is strongly recommended for training

Install dependencies:

```bash
pip install -r requirements.txt
```

If you want imports like `text_image_multi_modal_vlm.qwen_vlm` to work cleanly:

```bash
pip install -e .
```

---

## Dataset

This project is set up for **MMHS150K** (Multi-Modal Hate Speech).

Expected processed layout:

```
data/mmhs150k/
  train.csv
  val.csv
  test.csv
  class_names.txt
  images/
```

CSV format (required columns):

- `text`: the tweet/post text
- `image_path`: path to an image file (absolute or relative)
- `labels`: comma-separated labels or a Python-list-like string

`--image_root` is joined with `image_path` when `image_path` is not absolute. Set it so that:

`os.path.join(image_root, image_path)` points to a real image.

Examples:

- If `image_path` looks like `images/123.jpg`, set `--image_root data/mmhs150k`.
- If `image_path` looks like `123.jpg`, set `--image_root data/mmhs150k/images`.

---

## Data Preparation

This repo includes a downloader/prep script that can create `train.csv` / `val.csv` / `test.csv` + `images/` in a format compatible with the training code.

Script: `scripts/prepare_data.py`

### MMHS150K (recommended / used for the reported results)

```bash
python scripts/prepare_data.py --dataset mmhs150k --out_root data
```

This will write:

```
data/mmhs150k/
  images/
  train.csv
  val.csv
  test.csv
  class_names.txt
```

### Hateful Memes (binary)

```bash
python scripts/prepare_data.py --dataset hateful_memes --out_root data
```

Notes:

- MMHS150K download uses `gdown` and can be large.
- Hateful Memes download uses `huggingface_hub` snapshot download.

---

## Training

### Important notes about current script defaults

In `src/text_image_multi_modal_vlm/qwen_vlm.py` and `src/text_image_multi_modal_vlm/paligemma.py`, the `build_args()` functions currently include “demo defaults” that override several CLI values after parsing (paths + label list). This makes it easy to reproduce the provided MMHS150K runs, but it also means:

- The safest way to reproduce the stored runs is to keep the dataset in `data/mmhs150k/`.
- If you want to use custom paths/labels, remove or modify the hardcoded assignments in those `build_args()` functions.

Also note: `--class_names` expects a **comma-separated label string** (not a file path).

### Qwen2-VL

Reproduce the stored MMHS150K run:

```bash
python qwen_vlm.py
```

### PaliGemma 2

```bash
python paligemma.py
```

### SmolVLM

SmolVLM keeps CLI-provided values when present:

```bash
python -m text_image_multi_modal_vlm.smol_vlm \
  --train_csv data/mmhs150k/train.csv \
  --val_csv data/mmhs150k/val.csv \
  --test_csv data/mmhs150k/test.csv \
  --image_root data/mmhs150k \
  --class_names "racist,sexist,homophobe,religion,otherhate" \
  --out_dir runs/smolvlm_lora
```

### Dispatch wrapper

Use `scripts/train.py` to route to any model and forward arguments:

```bash
python scripts/train.py --model qwen -- --out_dir runs/qwen2vl_lora
```

---

## Evaluation & Results

Metrics are computed as multi-label scores derived from generated JSON labels:

- Micro F1 / Macro F1
- Subset accuracy
- Hamming loss

### Reported metrics (from `runs/*/*_metrics.json`)

| Model | Split | F1 Micro | F1 Macro | Subset Acc | Hamming Loss |
|------|-------|----------|----------|------------|--------------|
| **Qwen2-VL + LoRA** | Validation | 0.6172 | 0.5077 | 0.4366 | 0.14276 |
| **PaliGemma 2 + LoRA** | Validation | 0.5378 | 0.5000 | 0.4338 | 0.14220 |
| **Qwen2-VL + LoRA** | Test | 0.6110 | 0.4992 | – | – |
| **PaliGemma 2 + LoRA** | Test | 0.5404 | 0.4896 | – | – |

### Training & validation time (from run artifacts)

Times below are taken directly from saved logs where available:

| Model | Training time | Train throughput | Validation time | Validation throughput |
|------|---------------|------------------|-----------------|-----------------------|
| **Qwen2-VL + LoRA** | 49,067.23s (13:37:47) | 2.748 samples/s | 3,529.15s (0:58:49) | 1.417 samples/s |
| **PaliGemma 2 + LoRA** | *(not recorded in artifacts)* | – | 458.13s (0:07:38) | 10.914 samples/s |

Sources:

- Qwen train summary: `runs/qwen2vl_lora/training_logs`
- Qwen validation runtime: `runs/qwen2vl_lora/val_metrics.json`
- PaliGemma validation runtime: `runs/paligemma_lora/val_metrics.json`

---

## Inference

Minimal Qwen2-VL + LoRA loading example:

```python
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

base_id = "Qwen/Qwen2-VL-2B-Instruct"
adapter_dir = "runs/qwen2vl_lora/lora_adapter"

processor = AutoProcessor.from_pretrained(base_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter_dir)

image = Image.open("path/to/image.jpg").convert("RGB")
text = "Some text to analyze"

messages = [
    {"role": "system", "content": "Return JSON only."},
    {"role": "user", "content": [
        {"type": "text", "text": text},
        {"type": "image", "image": image},
    ]},
]

prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True).to(model.device)
out = model.generate(**inputs, max_new_tokens=64)
print(processor.decode(out[0], skip_special_tokens=True))
```

---

## SageMaker

This repo includes a minimal AWS SageMaker integration for:

- training via a SageMaker Hugging Face Estimator
- inference via a SageMaker endpoint using a custom `inference.py` handler

### Install (client-side)

To launch jobs from your local machine (or a notebook) install the optional dependency:

```bash
pip install -e ".[aws]"
```

### Training (inside SageMaker)

Entry point used by SageMaker training containers:

- `scripts/sagemaker/train_entrypoint.py`

It expects the training channel directory (usually `SM_CHANNEL_TRAINING`) to contain:

```
train.csv
val.csv
test.csv           (optional)
class_names.txt    (optional)
images/
```

Notes:

- The adapter writes model artifacts into `SM_MODEL_DIR` (so SageMaker uploads them as `model.tar.gz`).
- `class_names.txt` should be one label per line; if missing, pass `--class_names "a,b,c"`.

### Inference (endpoint)

Inference handler:

- `scripts/sagemaker/inference.py`

Request body (JSON):

```json
{"text":"...","image_base64":"..."}
```

Environment variables supported by the handler:

- `VLM_MODEL_ID` (default: `Qwen/Qwen2-VL-2B-Instruct`)
- `VLM_ADAPTER_SUBDIR` (default: `lora_adapter`)
- `VLM_MAX_NEW_TOKENS` (default: `64`)

### Launch example (from Python)

The helper module is importable as `text_image_multi_modal_vlm.sagemaker`:

```python
from text_image_multi_modal_vlm.sagemaker import (
  SageMakerTrainingSpec,
  SageMakerDeploySpec,
  create_hf_estimator,
  create_hf_model,
)

train_spec = SageMakerTrainingSpec(role_arn="arn:aws:iam::<acct>:role/<role>")

estimator = create_hf_estimator(
  spec=train_spec,
  source_dir=".",
  hyperparameters={
    "model": "qwen",
    "model_id": "Qwen/Qwen2-VL-2B-Instruct",
    "class_names": "racist,sexist,homophobe,religion,otherhate",
    "num_train_epochs": 1,
  },
)

# Provide your S3 channels. The training channel should contain train/val CSV + images.
estimator.fit({"training": "s3://<bucket>/<prefix>/mmhs150k/"})

deploy_spec = SageMakerDeploySpec(role_arn="arn:aws:iam::<acct>:role/<role>")
hf_model = create_hf_model(
  spec=deploy_spec,
  model_data=estimator.model_data,
  source_dir=".",
  env={"VLM_MODEL_ID": "Qwen/Qwen2-VL-2B-Instruct"},
)

predictor = hf_model.deploy(initial_instance_count=1, instance_type=deploy_spec.instance_type)
```

If you prefer a notebook-only workflow, you can paste the snippet above into a SageMaker Studio notebook.

---

## Artifacts

Training outputs are written to `runs/<experiment>/`:

- `lora_adapter/`: exported LoRA adapter (load via PEFT)
- `checkpoint-*/`: intermediate checkpoints
- `val_metrics.json`, `test_metrics.json`: saved metrics
- `logs/`: TensorBoard event files

---

## License

MIT License. See `LICENSE`.
