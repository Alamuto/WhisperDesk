"""Monitoramento de GPU sem torch: CTranslate2 para disponibilidade, NVML para VRAM/nome."""
from __future__ import annotations

import threading
from dataclasses import dataclass

_lock = threading.Lock()
_nvml_state: bool | None = None  # None = não tentado
_handle = None
_cuda_ok: bool | None = None
_gpu_ok: bool | None = None


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


def has_nvidia_gpu() -> bool:
    """True se há GPU CUDA com driver (mesmo que as bibliotecas cuBLAS/cuDNN ainda não estejam instaladas)."""
    global _gpu_ok
    if _gpu_ok is None:
        try:
            from app import runtime

            runtime.prepare()
            import ctranslate2

            _gpu_ok = ctranslate2.get_cuda_device_count() > 0
        except Exception:
            _gpu_ok = False
        if not _gpu_ok:
            nv, _h = _nvml()
            _gpu_ok = nv is not None
    return _gpu_ok


def is_cuda_available() -> bool:
    """True se dá para transcrever em CUDA agora: GPU com driver e bibliotecas CUDA resolvíveis."""
    global _cuda_ok
    if _cuda_ok is None:
        try:
            from app import runtime

            _cuda_ok = has_nvidia_gpu() and runtime.cuda_libs_ready()
        except Exception:
            _cuda_ok = False
    return _cuda_ok


def gpu_pack_needed() -> bool:
    """GPU NVIDIA presente, mas sem as bibliotecas CUDA: o pacote GPU pode ser baixado."""
    from app import gpu_pack

    return gpu_pack.platform_key() is not None and has_nvidia_gpu() and not is_cuda_available()


def refresh_cuda() -> bool:
    """Reavalia depois de instalar o pacote GPU."""
    global _cuda_ok
    from app import runtime

    runtime.activate_cuda_dirs()
    _cuda_ok = None
    return is_cuda_available()


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
