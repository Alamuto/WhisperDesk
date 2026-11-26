"""Monitoramento de GPU sem torch: CTranslate2 para disponibilidade, NVML para VRAM/nome."""
from __future__ import annotations

import threading
from dataclasses import dataclass

_lock = threading.Lock()
_nvml_state: bool | None = None  # None = não tentado
_handle = None
_cuda_ok: bool | None = None


@dataclass
class GPUInfo:
    used_mb: float
    total_mb: float
    name: str = "GPU"

    @property
    def free_mb(self) -> float:
        return max(0.0, self.total_mb - self.used_mb)

    @property
    def used_gb(self) -> float:
        return self.used_mb / 1024

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024

    @property
    def usage_pct(self) -> float:
        return (self.used_mb / self.total_mb) * 100 if self.total_mb > 0 else 0.0

    @property
    def display_str(self) -> str:
        return f"{self.used_gb:.1f}/{self.total_gb:.1f} GB"


def _nvml():
    global _nvml_state, _handle
    with _lock:
        if _nvml_state is None:
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    import pynvml
                pynvml.nvmlInit()
                _handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                _nvml_state = True
            except Exception:
                _nvml_state = False
                _handle = None
        if not _nvml_state:
            return None, None
        import pynvml

        return pynvml, _handle


def is_cuda_available() -> bool:
    """True se o CTranslate2 enxerga ao menos uma GPU CUDA (driver presente)."""
    global _cuda_ok
    if _cuda_ok is None:
        try:
            from app import runtime

            runtime.prepare()
            import ctranslate2

            _cuda_ok = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            _cuda_ok = False
    return _cuda_ok


def get_gpu_name() -> str:
    nv, h = _nvml()
    if nv is None:
        return "CUDA GPU" if is_cuda_available() else "—"
    try:
        name = nv.nvmlDeviceGetName(h)
        return name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        return "CUDA GPU"


def get_gpu_info() -> GPUInfo | None:
    nv, h = _nvml()
    if nv is None:
        return None
    try:
        mem = nv.nvmlDeviceGetMemoryInfo(h)
        return GPUInfo(used_mb=mem.used / 1048576, total_mb=mem.total / 1048576, name=get_gpu_name())
    except Exception:
        return None


def get_total_vram_mb() -> float:
    info = get_gpu_info()
    return info.total_mb if info else 0.0


def shutdown() -> None:
    global _nvml_state, _handle
    with _lock:
        if _nvml_state:
            try:
                import pynvml

                pynvml.nvmlShutdown()
            except Exception:
                pass
        _nvml_state = None
        _handle = None
