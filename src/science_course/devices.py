from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RuntimeDevice:
    """Resolved training runtime, ordered CUDA -> MPS -> CPU."""

    device: torch.device
    dtype: torch.dtype
    backend: str
    trainer_bf16: bool
    trainer_fp16: bool
    use_cpu: bool

    def as_dict(self) -> dict[str, str | bool]:
        return {
            "backend": self.backend,
            "device": str(self.device),
            "dtype": str(self.dtype).removeprefix("torch."),
            "trainer_bf16": self.trainer_bf16,
            "trainer_fp16": self.trainer_fp16,
            "use_cpu": self.use_cpu,
        }


def detect_runtime() -> RuntimeDevice:
    """Select CUDA first, then Apple MPS, then CPU."""

    if torch.cuda.is_available():
        use_bf16 = bool(torch.cuda.is_bf16_supported())
        return RuntimeDevice(
            device=torch.device("cuda"),
            dtype=torch.bfloat16 if use_bf16 else torch.float16,
            backend="cuda",
            trainer_bf16=use_bf16,
            trainer_fp16=not use_bf16,
            use_cpu=False,
        )

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return RuntimeDevice(
            device=torch.device("mps"),
            dtype=torch.float16,
            backend="mps",
            trainer_bf16=False,
            trainer_fp16=False,
            use_cpu=False,
        )

    return RuntimeDevice(
        device=torch.device("cpu"),
        dtype=torch.float32,
        backend="cpu",
        trainer_bf16=False,
        trainer_fp16=False,
        use_cpu=True,
    )


def clear_device_cache(runtime: RuntimeDevice) -> None:
    if runtime.backend == "cuda":
        torch.cuda.empty_cache()
    elif runtime.backend == "mps":
        torch.mps.empty_cache()
