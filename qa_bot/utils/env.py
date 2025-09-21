"""Runtime environment probing for selecting the retrieval pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(slots=True)
class EnvironmentInfo:
    gpu: bool
    memorag: bool
    faiss_gpu: bool
    torch_version: Optional[str]
    cuda_version: Optional[str]


def probe_env() -> EnvironmentInfo:
    info: Dict[str, Optional[bool | str]] = {
        "gpu": False,
        "memorag": False,
        "faiss_gpu": False,
        "torch_version": None,
        "cuda_version": None,
    }
    try:  # pragma: no cover - depends on optional torch install
        import torch  # type: ignore

        info["torch_version"] = getattr(torch, "__version__", None)
        info["gpu"] = bool(getattr(torch.cuda, "is_available", lambda: False)())
        if info["gpu"]:
            info["cuda_version"] = getattr(getattr(torch, "version", object()), "cuda", None)
    except Exception:
        info["torch_version"] = None

    try:  # pragma: no cover - optional memoRAG availability
        import memorag  # noqa: F401

        info["memorag"] = True
    except Exception:
        info["memorag"] = False

    try:  # pragma: no cover - depends on faiss installation
        import faiss  # type: ignore

        has_gpu_attr = hasattr(faiss, "get_num_gpus")
        gpu_count = faiss.get_num_gpus() if has_gpu_attr else 0
        info["faiss_gpu"] = bool(gpu_count and gpu_count >= 1)
    except Exception:
        info["faiss_gpu"] = False

    return EnvironmentInfo(
        gpu=bool(info["gpu"]),
        memorag=bool(info["memorag"]),
        faiss_gpu=bool(info["faiss_gpu"]),
        torch_version=info.get("torch_version"),
        cuda_version=info.get("cuda_version"),
    )


def select_pipeline(info: EnvironmentInfo, force: Optional[str] = None) -> str:
    """Return the pipeline name based on environment and overrides."""

    if force:
        force_lower = force.lower()
        if force_lower in {"memorag", "rag"}:
            return force_lower
    if info.gpu and info.memorag and info.faiss_gpu:
        return "memorag"
    return "rag"

