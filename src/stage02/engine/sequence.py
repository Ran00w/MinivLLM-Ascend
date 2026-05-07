from itertools import count

class Sequence:
    counter = count()
    
    def __init__(self, prompt_token_ids, generated_token_ids, SamplingParams, finished):

        self.seq_id = next(Sequence.counter)
        self.prompt_token_ids = prompt_token_ids
        self.generated_token_ids = generated_token_ids
        self.sampling_params = SamplingParams
        self.finished = finished

    def append_token(self, token_id):

        if self.finished:
            raise RuntimeError("cannot append token to a finished sequence")
        
        self.generated_token_ids.append(token_id)
        
        if token_id == self.sampling_params.eos_token_id or len(self.generated_token_ids) >= self.sampling_params.max_tokens:
            self.finished = True

    
        
