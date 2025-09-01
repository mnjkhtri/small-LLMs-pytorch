import pandas as pd
from tokenizer.gpt2 import BPE


def play_tokenizer_gpt2():

    from utils import Gutenberg

    corpus = Gutenberg.fetch_dataset('kant_critique')
    corpus = corpus[:250_000]

    text = "Let's test GPT-2 tokenizer: leading space, punctuation, contractions (don't), numbers 12345, decimals 12.34, emoji 🤖🔥, accents café, CJK 漢字, URL https://example.com, hashtag #AI, and loooooong."

    # 1. gpt pretrained:
    gpt2_pretrained = BPE.gpt2_bpe(pretrained=True)

    # 2. gpt style:
    gpt2_trained = BPE.gpt2_bpe(pretrained=False)
    gpt2_trained.train(corpus, vocab_size=1000, verbose=False)

    # 3. custom style:
    enc = {a: chr(a) for a in range(256)}
    dec = {v: k for k, v in enc.items()}
    init_standins = [enc[b] for b in range(256)]
    custom_bpe_with_regex = BPE(enc, dec, init_standins, (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""))
    custom_bpe_with_regex.train(corpus, vocab_size=1000, verbose=False)

    # 4. another custom:
    custom_bpe_without_regex = BPE(enc, dec, init_standins, None)
    custom_bpe_without_regex.train(corpus, vocab_size=1000, verbose=False)
 
    # the variants
    variants = [
        ("GPT-2 (pretrained @ 50256)",              gpt2_pretrained),
        ("GPT-2 (@ 1k)",                            gpt2_trained),
        ("Custom with regex (@ 1k)",                custom_bpe_with_regex),
        ("Custom without regex (@ 1k)",             custom_bpe_without_regex),
    ]

    rows = []
    for name, tok in variants:
        ids = tok.encode(text, verbose=False)
        chunks = tok._chunks(text)
        row = {
            "variant": name,
            "regex?": "yes" if tok.regex_pat is not None else "no",
            "vocab":  len(tok.tok2id),
            "chunks": len(chunks),
            "tokens": len(ids),
            "chars/tok": f"{len(text) / len(ids):.2f}",
        }
        rows.append(row)

    print(pd.DataFrame(rows))

if __name__ == "__main__":

    play_tokenizer_gpt2()