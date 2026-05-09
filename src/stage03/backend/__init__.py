from .base import Backend
from .torch_cpu import TorchCPUBackend
from .torch_npu import TorchNPUBackend


def get_backend(name: str) -> Backend:
    normalized = name.lower()

    if normalized == "cpu":
        return TorchCPUBackend()

    if normalized == "npu":
        return TorchNPUBackend(device_id=0)

    if normalized.startswith("npu:"):
        device_id = int(normalized.split(":", 1)[1])
        return TorchNPUBackend(device_id=device_id)

    raise ValueError(f"unsupported backend: {name}")