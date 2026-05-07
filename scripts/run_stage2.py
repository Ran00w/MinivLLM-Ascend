from stage02.engine.engine import Engine
from stage02.layers.model import TinyTransformerConfig, transformers
from stage02.layers.sampler import Sampler
from stage02.sampling_params import SamplingParams
from stage02.backend import TorchNPUBackend
from stage02.backend import TorchCPUBackend
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=['cpu', 'npu'], default='cpu')
    return parser.parse_args()

if __name__ == '__main__':
    torch.manual_seed(42)
    args = parse_args()
    if args.device == 'npu':
        backend = TorchNPUBackend()
    else:
        backend = TorchCPUBackend()
    config = TinyTransformerConfig(vocab_size=10000, hidden_size=64, 
                                   num_layers=12, num_heads=8, 
                                   intermediate_size=512, 
                                   max_position_embeddings=1024)
    model = transformers()
    backend.module_to_device(model)
    sampler = Sampler()
    SamplingParams = SamplingParams()
    engine = Engine(model = model, sampler = sampler)
    input_tokens = [1,4,9,10,34,23,3,56]
    ans = engine.generate(prompt_token_ids=input_tokens, sampling_params=SamplingParams)