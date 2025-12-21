"""AWS SageMaker helpers for training and deployment.

This module is deliberately import-safe even if the `sagemaker` SDK is not installed.
Install with:
  pip install -e .[aws]

It provides thin helpers for:
  - launching training jobs using SageMaker HuggingFace Estimator
  - deploying an HTTPS endpoint with a custom inference handler

You can use the scripts in `scripts/sagemaker/` as runnable examples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class SageMakerTrainingSpec:
    role_arn: str
    instance_type: str = "ml.g5.2xlarge"
    instance_count: int = 1
    volume_size_gb: int = 200
    max_run_seconds: int = 24 * 60 * 60

    # Hugging Face DLC versions
    transformers_version: str = "4.41"
    pytorch_version: str = "2.1"
    py_version: str = "py310"


@dataclass(frozen=True)
class SageMakerDeploySpec:
    role_arn: str
    instance_type: str = "ml.g5.2xlarge"
    initial_instance_count: int = 1

    # Hugging Face DLC versions
    transformers_version: str = "4.41"
    pytorch_version: str = "2.1"
    py_version: str = "py310"


def _require_sagemaker() -> Any:
    try:
        import sagemaker  # type: ignore

        return sagemaker
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SageMaker SDK not installed. Install with: pip install -e .[aws] (or pip install sagemaker)"
        ) from e


def create_hf_estimator(
    *,
    spec: SageMakerTrainingSpec,
    source_dir: str,
    entry_point: str = "scripts/sagemaker/train_entrypoint.py",
    base_job_name: str = "text-image-multi-modal-vlm",
    hyperparameters: Optional[Dict[str, Any]] = None,
    environment: Optional[Dict[str, str]] = None,
    dependencies: Optional[list[str]] = None,
) -> Any:
    """Create a SageMaker HuggingFace Estimator for training.

    Parameters
    - source_dir: local path sent to SageMaker (should include `src/` and `scripts/`)
    - entry_point: training adapter script inside source_dir
    - hyperparameters: passed to entry_point as CLI args

    Returns a `sagemaker.huggingface.HuggingFace` Estimator.
    """

    sagemaker = _require_sagemaker()
    from sagemaker.huggingface import HuggingFace  # type: ignore

    return HuggingFace(
        entry_point=entry_point,
        source_dir=source_dir,
        dependencies=dependencies,
        role=spec.role_arn,
        instance_type=spec.instance_type,
        instance_count=spec.instance_count,
        volume_size=spec.volume_size_gb,
        max_run=spec.max_run_seconds,
        transformers_version=spec.transformers_version,
        pytorch_version=spec.pytorch_version,
        py_version=spec.py_version,
        base_job_name=base_job_name,
        hyperparameters=hyperparameters or {},
        environment=environment or {},
    )


def create_hf_model(
    *,
    spec: SageMakerDeploySpec,
    model_data: str,
    source_dir: str,
    entry_point: str = "scripts/sagemaker/inference.py",
    env: Optional[Dict[str, str]] = None,
) -> Any:
    """Create a deployable SageMaker HuggingFaceModel.

    Parameters
    - model_data: S3 URI to model.tar.gz produced by training
    - source_dir / entry_point: inference code uploaded alongside the model
    - env: environment variables for inference container (e.g., VLM_MODEL_ID)

    Returns a `sagemaker.huggingface.HuggingFaceModel`.
    """

    _require_sagemaker()
    from sagemaker.huggingface import HuggingFaceModel  # type: ignore

    return HuggingFaceModel(
        model_data=model_data,
        role=spec.role_arn,
        entry_point=entry_point,
        source_dir=source_dir,
        transformers_version=spec.transformers_version,
        pytorch_version=spec.pytorch_version,
        py_version=spec.py_version,
        env=env or {},
    )
