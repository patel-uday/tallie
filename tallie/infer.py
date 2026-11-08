"""Inference, ranking (the counting part), save/load and the slim export."""
import json, re
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .data import build_input, collate
from .model import TallieModel, POLARITY, TYPES, amp_dtype
from .filters import is_valid_item, clean_span


# ---------------------------------------------------------------------------- inference
def chunk(text, max_chars=1500):
    if len(text) <= max_chars:
        return [(0, text)]
    out, i = [], 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        if j < len(text):
            k = text.rfind(". ", i + 200, j)
            j = k + 1 if k > 0 else j
        out.append((i, text[i:j]))
        i = j
    return out


@torch.no_grad()
def extract(model, tok, query, texts, batch_size=16, max_len=384, threshold=0.5):
    """Returns, for each text, the list of items that satisfy `query`."""
    model.eval()
    dev = next(model.parameters()).device
    pieces = [(ti, off, c) for ti, t in enumerate(texts) for off, c in chunk(t)]
    pieces.sort(key=lambda p: len(p[2]))  # similar lengths per batch = less padding, ~2x faster
    results = [[] for _ in texts]
    for b0 in range(0, len(pieces), batch_size):
        batch = pieces[b0:b0 + batch_size]
        feats, metas = [], []
        for ti, off, c in batch:
            ids, offs, qm, tm, base = build_input(tok, query, c, max_len)
            feats.append({"input_ids": ids, "query_mask": qm, "bio": [0] * len(ids),
                          "spans": [], "rel": [], "pol": []})
            metas.append((ti, off, c, offs, tm, base))
        x = collate(feats, tok.pad_token_id)
        x = {k: v.to(dev) for k, v in x.items()}
        with torch.autocast("cuda", dtype=amp_dtype(), enabled=dev.type == "cuda"):
            h = model.encode(x["input_ids"], x["attention_mask"])
        tags = model.mention_head(h).argmax(-1).tolist()
        span_lists = [model.decode_bio(tg[:len(f["input_ids"])], m[4])
                      for tg, f, m in zip(tags, feats, metas)]
        S = max(1, max(len(s) for s in span_lists))
        idx = torch.zeros(len(batch), S, 2, dtype=torch.long, device=dev)
        for i, sl in enumerate(span_lists):
            if sl:
                idx[i, :len(sl)] = torch.tensor(sl, device=dev)
        rel, pol, _ = model.heads(h, x["query_mask"], idx)
        rel, pol = torch.sigmoid(rel).tolist(), pol.softmax(-1).tolist()
        for i, sl in enumerate(span_lists):
            ti, off, c, offs, tm, base = metas[i]
            for j, (a, b) in enumerate(sl):
                fixed = clean_span(c, max(offs[a][0] - base, 0), offs[b][1] - base)
                if fixed is None:
                    continue
                cs, ce = fixed
                surface = c[cs:ce]
                if rel[i][j] >= threshold:
                    p = max(range(4), key=lambda k: pol[i][j][k])
                    results[ti].append({"text": surface, "start": off + cs, "end": off + ce,
                                        "score": round(rel[i][j], 3), "polarity": POLARITY[p]})
    return results


def normalize(name):
    n = name.lower().strip().strip("\"'“”‘’*_()[]:;,.!?")
    n = re.sub(r"^(the|a|an)\s+", "", n)
    return re.sub(r"\s+", " ", n)


def rank(results, top_k=50):
    """The counting step: plain code, exact and free."""
    groups = defaultdict(list)
    for per_text in results:
        for m in per_text:
            if is_valid_item(m["text"]):
                groups[normalize(m["text"])].append(m)
    table = []
    for key, ms in groups.items():
        pol = Counter(m["polarity"] for m in ms)
        table.append({"item": Counter(m["text"] for m in ms).most_common(1)[0][0],
                      "mentions": len(ms),
                      "avg_score": round(sum(m["score"] for m in ms) / len(ms), 3),
                      "polarity": dict(pol)})
    table.sort(key=lambda r: (-r["mentions"], -r["avg_score"]))
    return table[:top_k]


# ---------------------------------------------------------------------------- save / load
def save(model, tok, outdir, extra=None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    bb = model.backbone
    if hasattr(bb, "merge_and_unload"):
        bb = bb.merge_and_unload()
    bb.save_pretrained(out / "backbone", safe_serialization=True)
    tok.save_pretrained(out / "backbone")
    heads = {k: v for k, v in model.state_dict().items() if not k.startswith("backbone.")}
    torch.save(heads, out / "heads.pt")
    old = {}
    for name in ("tallie_config.json",):
        if (out / name).exists():
            old = json.loads((out / name).read_text()); break
    cfg = dict(model.cfg, lora_r=0, n_layers_original=model.n_layers_original,
               threshold=getattr(model, "threshold", old.get("threshold", 0.5)))
    cfg.update(extra or {})
    (out / "tallie_config.json").write_text(json.dumps(cfg, indent=2))


def load(outdir, device="cuda", dtype=None):
    out = Path(outdir)
    if dtype is None:
        dtype = amp_dtype() if str(device).startswith("cuda") else torch.float32
    cfg_file = out / "tallie_config.json"
    cfg = json.loads(cfg_file.read_text())
    bb = AutoModel.from_pretrained(out / "backbone", torch_dtype=dtype, attn_implementation="sdpa")
    m = TallieModel(backbone=bb, keep_layers=None, bidirectional=cfg["bidirectional"], lora_r=0,
                     gradient_checkpointing=False, span_dim=cfg.get("span_dim", 512))
    m.n_layers_original = cfg.get("n_layers_original", m.n_layers_original)
    m.threshold = cfg.get("threshold", 0.5)
    heads = torch.load(out / "heads.pt", map_location="cpu")
    if heads.get("vocab_map") is not None:
        m.vocab_map = heads["vocab_map"]
    m.load_state_dict(heads, strict=False)
    tok = AutoTokenizer.from_pretrained(out / "backbone")
    return m.to(device), tok


# ---------------------------------------------------------------------------- slim export
def trim_vocab(model, tok, corpus_texts, extra_ids=()):
    """Shrink the embedding table for English use.
    Keeps EVERY token that decodes to plain ASCII text (all of English, code, numbers, punctuation)
    plus every token seen in `corpus_texts`. Drops tokens for other scripts (Chinese, Arabic, ...).
    v1.1 kept only corpus tokens, which broke questions containing unseen words; this fixes that.
    Anything not kept maps to one shared 'unknown' row."""
    base = model.backbone.get_base_model() if hasattr(model.backbone, "get_base_model") else model.backbone
    n_vocab = base.embed_tokens.weight.shape[0]
    used = set(extra_ids) | set(tok.all_special_ids)
    for i in range(min(n_vocab, len(tok))):
        piece = tok.decode([i])
        if piece and piece.isascii():
            used.add(i)
    for t in corpus_texts:
        used.update(tok(t, add_special_tokens=False)["input_ids"])
    used.update(range(256))
    if tok.pad_token_id is not None:
        used.add(tok.pad_token_id)
    keep = sorted(i for i in used if i < n_vocab)
    W = base.embed_tokens.weight.data
    newW = torch.cat([W[keep], W[keep].mean(0, keepdim=True)], 0)
    emb = nn.Embedding(newW.shape[0], newW.shape[1], dtype=W.dtype, device=W.device)
    emb.weight.data.copy_(newW)
    base.embed_tokens = emb
    base.config.vocab_size = newW.shape[0]
    vm = torch.full((W.shape[0],), len(keep), dtype=torch.long, device=W.device)
    vm[torch.tensor(keep, device=W.device)] = torch.arange(len(keep), device=W.device)
    model.vocab_map = vm
    for k in ("bos_token_id", "eos_token_id"):  # ids refer to the old vocabulary
        setattr(base.config, k, None)
    return len(keep) + 1


@torch.no_grad()
def extract_all(model, tok, texts, batch_size=32, max_len=384, query="all named items mentioned"):
    """One pass over all texts: every named item with the commenter's opinion and a context window.
    Questions are then answered from this list (see ranking.py) without reading the texts again."""
    model.eval()
    dev = next(model.parameters()).device
    pieces = [(ti, off, c) for ti, t in enumerate(texts) for off, c in chunk(t)]
    pieces.sort(key=lambda p: len(p[2]))
    out = []
    for b0 in range(0, len(pieces), batch_size):
        batch = pieces[b0:b0 + batch_size]
        feats, metas = [], []
        for ti, off, c in batch:
            ids, offs, qm, tm, base = build_input(tok, query, c, max_len)
            feats.append({"input_ids": ids, "query_mask": qm, "bio": [0] * len(ids),
                          "spans": [], "rel": [], "pol": []})
            metas.append((ti, c, offs, tm, base))
        x = {k: v.to(dev) for k, v in collate(feats, tok.pad_token_id).items()}
        with torch.autocast("cuda", dtype=amp_dtype(), enabled=dev.type == "cuda"):
            h = model.encode(x["input_ids"], x["attention_mask"])
        tags = model.mention_head(h).argmax(-1).tolist()
        span_lists = [model.decode_bio(tg[:len(f["input_ids"])], m[3]) for tg, f, m in zip(tags, feats, metas)]
        S = max(1, max(len(s) for s in span_lists))
        idx = torch.zeros(len(batch), S, 2, dtype=torch.long, device=dev)
        for i, sl in enumerate(span_lists):
            if sl:
                idx[i, :len(sl)] = torch.tensor(sl, device=dev)
        _, pol, typ = model.heads(h, x["query_mask"], idx)
        pol = pol.argmax(-1).tolist()
        tprob = typ.softmax(-1)
        tconf, tid = tprob.max(-1)
        tid, tconf = tid.tolist(), tconf.tolist()
        for i, sl in enumerate(span_lists):
            ti, c, offs, tm, base = metas[i]
            for j, (a, b) in enumerate(sl):
                fx = clean_span(c, max(offs[a][0] - base, 0), offs[b][1] - base)
                if fx is None:
                    continue
                s, e = fx
                name = c[s:e].strip("\"'“”‘’")
                if not is_valid_item(name):
                    continue
                out.append({"text": name, "polarity": POLARITY[pol[i][j]], "doc": ti,
                            "type": TYPES[tid[i][j]], "type_conf": round(tconf[i][j], 3),
                            "context": c[max(0, s - 150):e + 150]})
    return out
