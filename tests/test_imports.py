import importlib


def test_import_qwen():
    mod = importlib.import_module("text_image_multi_modal_vlm.qwen_vlm")
    assert hasattr(mod, "main")


def test_import_paligemma():
    mod = importlib.import_module("text_image_multi_modal_vlm.paligemma")
    assert hasattr(mod, "main")


def test_import_smol():
    mod = importlib.import_module("text_image_multi_modal_vlm.smol_vlm")
    assert hasattr(mod, "main")


def test_import_sagemaker_helpers():
    mod = importlib.import_module("text_image_multi_modal_vlm.sagemaker")
    assert hasattr(mod, "create_hf_estimator")
