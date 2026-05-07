from abc import ABC, abstractmethod
import torch


class Backend(ABC):
    name: str
    device: torch.device

    @abstractmethod
    def is_available(self):
        raise NotImplementedError
    
    def tensor(self, data, dtype):
        return torch.tensor(data, dtype=dtype, device=self.device)
    
    def to_device(self, tensor):
        return tensor.to(self.device)
    
    def module_to_device(self, module):
        return module.to(self.device)
    
    def synchronize(self):
        return None
    
    def empty_cache(self):
        return None
    
    def memory_info(self) -> dict[str, int | None]:
        return {"allocated": None, "reserved": None}


    

    
    
