import torch
from models.text2text.gpt2 import GPT2
from models.text2text.llama3 import Llama3
from models.text2text.gemma3 import Gemma3
from models.text2text.qwen3 import Qwen3


class LLM:

    def __init__(self, tokenizer, model):

        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()
        self.device = next(model.parameters()).device

        # internal kv-cache state (per-LLM state)
        self._past_key_values = None
        self._cached_ids = []

        self.max_context = self.model.max_length

        # generate configs are here for now:

        self.top_k: int = 100
        self.T: float = 0.8
        self.use_cache = True

    def _reset_cache(self):
        self._past_key_values = None
        self._cached_ids = []
    
    def _infer_once(self,
                    prompt_ids,
                    *,
                    top_k: int = 1,
                    T: float = 1.0,
                    use_cache: bool = False
                ):

        if use_cache: # even if were to use cache but is not prefix then cant really use it right?

            is_prefix = (self._cached_ids == prompt_ids[:len(self._cached_ids)])
            # if cached ids are prefix of current prompt ids

            if not is_prefix:
                self._reset_cache()
                new_token_ids = prompt_ids
            else:
                new_token_ids = prompt_ids[len(self._cached_ids):]

            assert len(new_token_ids) > 0, "No new tokens fed to the model while use_cache=True."

            input_ids = torch.tensor([new_token_ids], device=self.device, dtype=torch.long)

            with torch.no_grad():
                out = self.model(input_ids, past_key_values=self._past_key_values, use_cache=use_cache)
                logits, self._past_key_values = out['logits'], out['past_key_values']
                self._cached_ids = prompt_ids.copy()
                next_logits = logits[:, -1, :]

        else:

            input_ids = torch.tensor([prompt_ids], device=self.device)
            with torch.no_grad():
                out = self.model(input_ids, use_cache=use_cache)
                logits = out["logits"]
                next_logits = logits[:, -1, :]

        # Handle TopK (replace < cutoff with -inf so they are softmaxx to 0):
        topk_logits, _ = torch.topk(next_logits, min(top_k, next_logits.shape[-1])) # [1, TOPK]
        cutoff = topk_logits[..., -1].unsqueeze(-1) # [1, TOPK]
        next_logits = torch.where(next_logits < cutoff, torch.full_like(next_logits, -float("inf")), next_logits)

        # Sampling with Temp:
        probs = torch.softmax(next_logits / T, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        # [1, 1]

        return next_id[0].item()
    
    def generate(self, prompt: str):

        prompt_ids = self.tokenizer.encode(prompt)
        print(self.tokenizer.decode(prompt_ids), end='')

        total_toks = 0
        try:
            while True:
                total_toks += 1
                new_id = self._infer_once(prompt_ids, top_k=self.top_k, T=self.T, use_cache=self.use_cache)
                prompt_ids.append(new_id)

                # for showoff:
                new_tok = self.tokenizer.decode([new_id])
                print(new_tok, end="", flush=True)
                # ---

                if (total_toks % 256) == 0:
                    occ = total_toks / self.max_context
                    used_pct = int(occ * 100)
                    print(
                        f"\n\n[CTX] {used_pct}% used | {total_toks}/{self.max_context}\n",
                        flush=True
                    )
        except Exception as e:
            raise e

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Run LLM with specific model and type")
    parser.add_argument("--model", type=str, required=True, choices=["gpt2", "llama3", "gemma3", "qwen3"], help="Model name")
    parser.add_argument("--type", type=str, required=True, choices=["base", "instruct"], help="Model type")

    args = parser.parse_args()

    model = args.model
    type = args.type

    dtype = torch.float16 if model == "gpt2" else torch.bfloat16
    device = "cpu" if model == "gpt2" else "cuda"

    # dispatch table for models
    model_classes = {
        "gpt2": GPT2,
        "llama3": Llama3,
        "qwen3": Qwen3,
        "gemma3": Gemma3,
    }

    if model not in model_classes:
        raise ValueError(f"Unknown model: {model}")
    
    cls = model_classes[model]
    tokenizer, model = cls.from_pretrained(type.upper(), torch_dtype=dtype, device=device)
    llm = LLM(tokenizer, model)

    prompts = {
        "base": "Life is beautiful because",
        "instruct": "Why is life beautiful?"
    }
    
    llm.generate(prompts[type])
