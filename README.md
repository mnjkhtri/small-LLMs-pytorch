## gpt2 bpe tokenizer:

- take text → UTF-8 bytes → map each byte to a unique Unicode stand-in (so BPE can work on them losslessly)
- optionally pre-split with the GPT-2 regex (so BPE merges are confined to chunks and leading spaces, punctuation behave right)
- init vocab with 256 stand-ins (one per byte), each mapped to a token id
- train by counting most-frequent adjacent pairs (inside regex-split chunks), merge the top pair into a new token
- keep merging until target vocab size; at encode, apply merges greedily by rank; decode by inverting stand-ins → bytes → UTF-8