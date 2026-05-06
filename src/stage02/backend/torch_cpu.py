import torch
from .base import Backend

class TorchCPUbackend(Backend):
    name = 'cpu'
    def __init__(self):
        self.device = torch.device('cpu')

    def is_available(self):
        return True