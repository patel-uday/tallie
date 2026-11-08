"""Tokenization + label alignment for Tallie."""
import json
import torch
from torch.utils.data import Dataset

from .model import BIO_O, BIO_B, BIO_I, TYPES

TYPE_ID = {t: i for i, t in enumerate(TYPES)}

QUERY_TAG, TEXT_TAG = "Query: ", "\nText: "


def build_input(tokenizer, query, text, max_len):
    prefix = QUERY_TAG + query + TEXT_TAG
    enc = tokenizer(prefix + text, return_offsets_mapping=True, truncation=True,
                    max_length=max_len, add_special_tokens=False)
    offs = enc["offset_mapping"]
    q_lo, q_hi = len(QUERY_TAG), len(QUERY_TAG) + len(query)
    base = len(prefix)
    query_mask = [int(s < q_hi and e > q_lo) for s, e in offs]
    text_mask = [int(e > base) for s, e in offs]
    return enc["input_ids"], offs, query_mask, text_mask, base


def char_span_to_tokens(offs, base, s, e):
    s, e = s + base, e + base
    toks = [i for i, (ts, te) in enumerate(offs) if te > s and ts < e]
    return (toks[0], toks[-1]) if toks else None


class TallieDataset(Dataset):
    def __init__(self, path, tokenizer, max_len=384, limit=0):
        self.rows = [json.loads(l) for l in open(path, encoding="utf-8")]
        if limit:
            self.rows = self.rows[:limit]
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        ids, offs, qmask, tmask, base = build_input(self.tok, r["query"], r["text"], self.max_len)
        bio = [BIO_O if t else -100 for t in tmask]
        bio = [b if q == 0 else BIO_O for b, q in zip(bio, qmask)]
        spans, rel, pol, typ = [], [], [], []
        for sp in r["spans"]:
            te = char_span_to_tokens(offs, base, sp["start"], sp["end"])
            if te is None:
                continue  # truncated away
            a, b = te
            bio[a] = BIO_B
            for k in range(a + 1, b + 1):
                bio[k] = BIO_I
            spans.append([a, b]); rel.append(sp["relevant"]); pol.append(sp["polarity"])
            typ.append(TYPE_ID.get(sp.get("type"), -100))
        return {"input_ids": ids, "query_mask": qmask, "bio": bio,
                "spans": spans, "rel": rel, "pol": pol, "typ": typ}


def collate(batch, pad_id):
    L = max(len(b["input_ids"]) for b in batch)
    S = max(1, max(len(b["spans"]) for b in batch))
    B = len(batch)
    out = {
        "input_ids": torch.full((B, L), pad_id, dtype=torch.long),
        "attention_mask": torch.zeros(B, L, dtype=torch.long),
        "query_mask": torch.zeros(B, L, dtype=torch.long),
        "bio_labels": torch.full((B, L), -100, dtype=torch.long),
        "span_idx": torch.zeros(B, S, 2, dtype=torch.long),
        "span_valid": torch.zeros(B, S, dtype=torch.long),
        "rel_labels": torch.zeros(B, S, dtype=torch.long),
        "pol_labels": torch.zeros(B, S, dtype=torch.long),
        "type_labels": torch.full((B, S), -100, dtype=torch.long),
    }
    for i, b in enumerate(batch):
        n = len(b["input_ids"])
        out["input_ids"][i, :n] = torch.tensor(b["input_ids"])
        out["attention_mask"][i, :n] = 1
        out["query_mask"][i, :n] = torch.tensor(b["query_mask"])
        out["bio_labels"][i, :n] = torch.tensor(b["bio"])
        k = len(b["spans"])
        if k:
            out["span_idx"][i, :k] = torch.tensor(b["spans"])
            out["span_valid"][i, :k] = 1
            out["rel_labels"][i, :k] = torch.tensor(b["rel"])
            out["pol_labels"][i, :k] = torch.tensor(b["pol"])
            if b.get("typ"):
                out["type_labels"][i, :k] = torch.tensor(b["typ"])
    return out
