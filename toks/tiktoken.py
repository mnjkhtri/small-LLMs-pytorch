import tiktoken
from tiktoken.load import load_tiktoken_bpe
from pathlib import Path
import requests
import os


class Tiktoken:
    def __init__(self, model_path, pat_str):

        mergeable_ranks = load_tiktoken_bpe(model_path)
        special_tokens = [
            "<|begin_of_text|>",                # BOS (beginning of sequence) – always added at start
            "<|end_of_text|>",                  # EOS (end of sequence) – marks end of document/output
            "<|reserved_special_token_0|>",     # placeholder, never trained (future expansion)
            "<|reserved_special_token_1|>",     # placeholder
            "<|reserved_special_token_2|>",     # placeholder
            "<|reserved_special_token_3|>",     # placeholder
            "<|start_header_id|>",              # used in instruct/chat models: marks start of role header (e.g., user/assistant/system)
            "<|end_header_id|>",                # used in instruct/chat models: marks end of role header
            "<|reserved_special_token_4|>",     # another placeholder, unused
            "<|eot_id|>",                       # end-of-turn marker (chat fine-tuning); every message ends with this
        ] + [
            # Remaining reserved slots: unused placeholders filling up the reserved 256-token block.
            f"<|reserved_special_token_{i}|>"
            for i in range(5, 256 - 5)
        ]
        self.special_tokens = {token: len(mergeable_ranks) + i for i, token in enumerate(special_tokens)}

        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )
    
    def encode(self, text, allow_special=False):
        return [self.special_tokens['<|begin_of_text|>']] + self.model.encode(text, allowed_special="all" if allow_special else set(), disallowed_special=set())

    def encode_instruct(self, system: str, user: str):

        st = self.special_tokens
        enc = lambda s: self.model.encode(
            s,
            allowed_special=set(),
            disallowed_special=set()
        )

        toks = [st['<|begin_of_text|>']]

        # System
        toks += [st['<|start_header_id|>']] + enc("system") + [st['<|end_header_id|>']]
        toks += enc("\n\n" + system) + [st['<|eot_id|>']]

        # User
        toks += [st['<|start_header_id|>']] + enc("user") + [st['<|end_header_id|>']]
        toks += enc("\n\n" + user) + [st['<|eot_id|>']]

        # Assistant (generation will continue from here)
        toks += [st['<|start_header_id|>']] + enc("assistant") + [st['<|end_header_id|>']]
        toks += enc("\n\n")

        return toks

    def decode(self, ids):
        return self.model.decode(ids)

    @classmethod
    def llama3(cls):
        TOKENIZER_URL = "https://huggingface.co/meta-llama/llama-3.2-1b/resolve/main/original/tokenizer.model"
        save_path = Path("weights_cache") / "llama-3.2-1b"
        save_path.mkdir(parents=True, exist_ok=True)
        model_file = save_path / "tokenizer.model"

        HF_TOKEN = os.getenv('HF_TOKEN')
        if not model_file.exists():
            headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
            r = requests.get(TOKENIZER_URL, headers=headers, stream=True, timeout=60)
            r.raise_for_status()
            with open(model_file, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)

        pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
        return cls(str(model_file), pat_str)
    