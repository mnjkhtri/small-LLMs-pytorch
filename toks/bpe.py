import json
import regex as re
from typing import Dict, List, Tuple
import pandas as pd


class BPE:
    """
    byte_to_unicode:
    unicode_to_byte:
    base_toks:
    regex_str:
    """

    def __init__(self, byte_to_unicode, unicode_to_byte, base_toks, regex_str):
        self.byte_to_unicode = byte_to_unicode
        self.unicode_to_byte = unicode_to_byte

        self.tok2id: Dict[str, int] = {}
        self.id2tok: Dict[int, str] = {}
        for ch in base_toks:
            i = len(self.tok2id)
            self.tok2id[ch] = i
            self.id2tok[i] = ch

        self.regex_pat = re.compile(regex_str) if regex_str is not None else None
        self.bpe_ranks: Dict[Tuple[str, str], int] = {}

        self.sos_tok = None
        self.eos_tok = None
        # they should init by training process (or preloading) and be part of tok2id, 

    def _chunks(self, text: str) -> List[str]:
        """
        normal str -> chunkify.
        """
        if self.regex_pat is None:
            return [text]
        return [m.group(0) for m in self.regex_pat.finditer(text)]

    def _visible(self, text: str) -> str:
        """
        raw text -> utf-8 bytes -> visible stand-ins.
        """
        return "".join(self.byte_to_unicode[b] for b in text.encode("utf-8"))

    def train(self, text, vocab_size, verbose=False):
        """
        Uses the configured pre-splitter:
        regex_pat is None: stream mode (one big chunk; merges can cross anything)
        regex_pat: regex-chunked (merges confined within chunks; GPT-style)
        Training mutates tok2id/id2tok and bpe_ranks.
        """

        assert vocab_size >= len(
            self.tok2id
        ), "vocab_size must be >= 256 (base alphabet + any pre-added specials)."

        new_merges = vocab_size - len(self.tok2id)

        # precreate a freq dictonary of chunks so instead of iterating over all chunks we iterate on this
        chunks = self._chunks(text)
        from collections import Counter
        chunk_freq = Counter()
        for ch in chunks:
            vis = self._visible(ch)
            chunk_freq[tuple(vis)] += 1

        # iteratively pick the most frequent adjacent pair, assign next rank, update words
        for step in range(new_merges):
            pair_freq = {}
            for syms, f in chunk_freq.items(): # (tuple, frequency)
                if len(syms) < 2:
                    continue
                for i in range(len(syms) - 1):
                    p = (syms[i], syms[i + 1])
                    pair_freq[p] = pair_freq.get(p, 0) + f
            if not pair_freq:
                print(f"[train] stop: no pairs at step {step}")
                break

            best = max(pair_freq.items(), key=lambda kv: kv[1])[0]  # (a,b) to merge
            rank = len(self.bpe_ranks)
            self.bpe_ranks[best] = rank

            merged_tok = best[0] + best[1]
            
            assert merged_tok not in self.tok2id
            nid = len(self.tok2id)
            self.tok2id[merged_tok] = nid
            self.id2tok[nid] = merged_tok

            a, b = best

            def merge_once(syms):
                """Replace every (a,b) with 'a+b' inside one tuple of unicodes."""
                out: List[str] = []
                i = 0
                while i < len(syms):
                    if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                        out.append(merged_tok)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                return tuple(out)

            new_freq = {}
            for syms, f in chunk_freq.items():
                ms = merge_once(syms)
                new_freq[ms] = new_freq.get(ms, 0) + f

            chunk_freq = new_freq

            if verbose:
                print(f"merge {step+1}/{new_merges}: {best} -> '{merged_tok}' | occurence: {pair_freq[best]} | rank={rank}")

        # specials:
        nid = len(self.tok2id)
        self.tok2id['<sos>'] = nid
        self.id2tok[nid] = '<sos>'
        self.sos_tok = '<sos>'

        nid = len(self.tok2id)
        self.tok2id['<eos>'] = nid
        self.id2tok[nid] = '<eos>'
        self.eos_tok = '<eos>'

    def _bpe(self, vis_chunk: str) -> List[str]:
        """
        Apply BPE merges greedily (by rank) within ONE visible chunk.
        Returns a list of merged token strings (each 1+ stand-in chars).
        """
        syms: List[str] = list(vis_chunk)
        while True:
            ps = {(syms[i], syms[i + 1]) for i in range(len(syms) - 1)}
            if not ps:
                break
            # choose the present pair with the smallest rank number
            best = min(ps, key=lambda p: self.bpe_ranks.get(p, float("inf")))
            # if best comes from inf then need to check if such merge is possible
            if best not in self.bpe_ranks:
                break
            a, b = best
            # merge all (a,b) occurrences in a single pass
            out: List[str] = []
            i = 0
            while i < len(syms):
                if i + 1 < len(syms) and syms[i] == a and syms[i + 1] == b:
                    out.append(a + b)
                    i += 2
                else:
                    out.append(syms[i])
                    i += 1
            syms = out
            if len(syms) == 1:
                break
        return syms

    def encode(self, text: str, add_specials=False, verbose=False) -> List[int]:
        """
        Convert text: str -> token IDs: List
        """

        ids: List[int] = []

        for i, chunk in enumerate(self._chunks(text)):
            vis = self._visible(chunk)
            toks = self._bpe(vis)
            tok_ids = [self.tok2id[tok] for tok in toks]
            if verbose:
                print(f"chunk {i}: {chunk!r} | visible: {vis} | toks: {toks} | ids: {tok_ids}")
            ids.extend(tok_ids)

        if add_specials:
            if self.sos_tok:
                ids = [self.tok2id[self.sos_tok]] + ids
            if self.eos_tok:
                ids.append(self.tok2id[self.eos_tok])

        return ids

    def decode(self, ids) -> str:
        """
        Convert token IDs: List -> text: str. Fully lossless (reverses the byte map).
        """
        s: str = "".join(self.id2tok[i] for i in ids)
        return bytes(self.unicode_to_byte[ch] for ch in s).decode("utf-8")

    @classmethod
    def gpt2(cls, pretrained=False):
        
        # nice bytes kept as their own codepoints
        bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        cs = bs[:]  # copy
        # remaining bytes remapped to U+0100, U+0101, ...
        n = 0
        for b in range(256):
            if b not in bs:
                bs.append(b)
                cs.append(256 + n)
                n += 1
        enc = {b: chr(c) for b, c in zip(bs, cs)}
        dec = {v: k for k, v in enc.items()}
        init_standins = [chr(c) for c in cs]

        init_bpe = cls(enc, dec, init_standins, (r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""))

        if not pretrained:
            return init_bpe
        
        # instead of training load the weights?
                
        from huggingface_hub import hf_hub_download
        vocab_path  = hf_hub_download(repo_id="openai-community/gpt2", filename="vocab.json")
        merges_path = hf_hub_download(repo_id="openai-community/gpt2", filename="merges.txt")

        with open(vocab_path, "r", encoding="utf-8") as f:
            enc = json.load(f)

        init_bpe.tok2id = {k: int(v) for k, v in enc.items()}
        init_bpe.id2tok = {int(v): k for k, v in enc.items()}

        with open(merges_path, "r", encoding="utf-8") as f:
            merges = [
                tuple(line.strip().split())
                for line in f
                if line and not line.startswith("#")
            ]

        init_bpe.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        init_bpe.eos_tok = '<|endoftext|>'

        return init_bpe