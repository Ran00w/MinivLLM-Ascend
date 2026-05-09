import torch

from .base import Backend

class TorchNPUBackend(Backend):
    name = "npu"

    def __init__(self, device_id = 0):
        super().__init__()
        self._import_error: Exception | None = None

        self.device_id = device_id
        try:
            import torch_npu
        except Exception as exc:
            self._import_error = exc
        else:
            self.device = torch.device(f"npu:{device_id}")
        # self.device = torch.device(f"npu:{device_id}")

    def is_available(self):
        return hasattr(torch, "npu") and torch.npu.is_available()
    
    def synchronize(self):
        if self.is_available():
            torch.npu.synchronize()

    def empty_cache(self):
        if self.is_available():
            torch.npu.empty_cache()

    def memory_info(self):
        if not self.is_available():
            return {"allocated": None, "reserved": None}
        return {
            "allocated": torch.npu.memory_allocated(self.device), 
            "reserved": torch.npu.memory_reserved(self.device)
        }

    