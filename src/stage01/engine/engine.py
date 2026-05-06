from .sequence import Sequence
from ..sampling_params import SamplingParams
from ..layers.sampler import Sampler
from ..layers.model import transformers
import torch
import torch.nn as nn

device = "cuda:0"

class Engine:
    def __init__(self, model, sampler):
        self.model = model
        self.sampler = sampler

    @torch.no_grad
    def generate(
        self, 
        prompt_token_ids: list[int],
        sampling_params
    ):
        seq = Sequence(prompt_token_ids=prompt_token_ids,
                       generated_token_ids=[],
                       SamplingParams=sampling_params,finished=False)
        
        self.model.eval()
        step = 0
        while not seq.finished:
            input_ids = torch.tensor(
                [seq.prompt_token_ids + seq.generated_token_ids],
                dtype=torch.long,
                device=device
            )

            logits = self.model(input_ids)
            last_token = logits[:, -1, :]#.squeeze(0)

            next_token_ids = self.sampler.sample(last_token)

            next_token_id = int(next_token_ids[0].item())

            if step == 0:
                phase = 'prefill'
            else:
                phase = 'decode'
            seq.append_token(next_token_id)
            print(f"step={step}, phase={phase}, token_id={next_token_id}")

            step += 1

        return seq
    
if __name__ == '__main__':
    model = transformers()
    sampler = Sampler()
    SamplingParams = SamplingParams()
    engine = Engine(model=model, sampler=sampler)
    engine.generate(prompt_token_ids=[0,1,2], sampling_params=SamplingParams)
    print("finished")