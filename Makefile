PYTHON ?= python
PACKAGE := text_image_multi_modal_vlm

.PHONY: help install install-dev lint format test train-qwen train-pali train-smol clean

help:
	@echo "Targets:"
	@echo "  install       Install package and runtime dependencies"
	@echo "  install-dev   Install package with dev tooling"
	@echo "  lint          Run Ruff lint checks"
	@echo "  format        Run Black formatter"
	@echo "  test          Run pytest suite"
	@echo "  train-qwen    Example Qwen2-VL training command"
	@echo "  train-pali    Example PaliGemma training command"
	@echo "  train-smol    Example SmolVLM training command"

install:
	$(PYTHON) -m pip install -e .

install-dev: install
	$(PYTHON) -m pip install -e .[dev]

lint:
	ruff check src tests

format:
	black src tests

test:
	pytest

train-qwen:
	$(PYTHON) -m $(PACKAGE).qwen_vlm --train_csv data/mmhs150k/train.csv --val_csv data/mmhs150k/val.csv --image_root data/mmhs150k/images --class_names data/mmhs150k/class_names.txt --out_dir runs/qwen2vl_lora

train-pali:
	$(PYTHON) -m $(PACKAGE).paligemma --train_csv data/mmhs150k/train.csv --val_csv data/mmhs150k/val.csv --image_root data/mmhs150k/images --class_names data/mmhs150k/class_names.txt --out_dir runs/paligemma_lora

train-smol:
	$(PYTHON) -m $(PACKAGE).smol_vlm --train_csv data/mmhs150k/train.csv --val_csv data/mmhs150k/val.csv --image_root data/mmhs150k/images --class_names data/mmhs150k/class_names.txt --out_dir runs/smolvlm_lora

clean:
	@find . -name "__pycache__" -type d -prune -exec rm -rf {} +
	@find . -name ".pytest_cache" -type d -prune -exec rm -rf {} +
