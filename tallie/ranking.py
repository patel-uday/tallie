"""
Turning per-comment extractions into ranked answers.

Tallie's model reads each comment once and returns every named item with its type and the
commenter's opinion. A question such as "podcasts people recommend" is answered by
  1. parsing it into a kind of thing (podcast) and an opinion (positive),
  2. keeping the items the model typed as that kind, with that opinion,
  3. tallying across all comments (one vote per comment) and ranking.
Extra topic words ("for beginners", "about space") are checked by the model's relevance head.
"""
import re
from collections import Counter, defaultdict

QUERY_TYPES = [  # words in a question -> wanted type
    ("tv_show", r"\b(tv shows?|television|series|sitcoms?|anime)\b"),
    ("board_game", r"\b(board games?|card games?|tabletop( games?)?)\b"),
    ("video_game", r"\b(video games?|games?)\b"),
    ("podcast", r"\bpodcasts?\b"),
    ("movie", r"\b(movies?|films?|documentar(y|ies))\b"),
    ("book", r"\b(books?|novels?|reads?|reading)\b"),
    ("music", r"\b(albums?|songs?|music|bands?)\b"),
    ("software", r"\b(software|tools?|apps?|programs?|editors?|browsers?|languages?|libraries|frameworks?)\b"),
    ("person", r"\b(authors?|writers?|directors?|people to follow)\b"),
]
TYPE_FAMILY = {"board_game": {"board_game"}, "video_game": {"video_game"},
               "software": {"software", "product"}, "person": {"person"}}
POS_WORDS = r"\b(lik(e|ed)|lov(e|ed)|favou?rites?|best|great|recommend(ed)?|praised?|enjoy(ed)?|good|top)\b"
NEG_WORDS = r"\b(hat(e|ed)|dislik(e|ed)|worst|regret(ted)?|overrated|criticiz(e|ed)|bad|awful|disappointing|complain(ed)?)\b"
STOP = set("""a an the of for about to in on with and or by from that which who people commenters
comments liked loved like love hate hated disliked best worst most favorite favourite recommended
recommend regret regretted listening watching reading read watch play played good great bad
things stuff related everyone anyone some any all mentioned named discussed here
found find think thought say said consider considered really very much more less ones one
would should could get got make made also just""".split())


def _norm(t):
    t = t.lower().strip().strip("\"'“”‘’*_()[]:;,.!?")
    t = re.sub(r"^(the|a|an)\s+", "", t)
    return re.sub(r"[\s\-–:]+", " ", t)


def parse_query(q):
    ql = q.lower()
    want_type = next((t for t, pat in QUERY_TYPES if re.search(pat, ql)), None)
    sentiment = "negative" if re.search(NEG_WORDS, ql) else "positive" if re.search(POS_WORDS, ql) else None
    rest = ql
    for _, pat in QUERY_TYPES:
        rest = re.sub(pat, " ", rest)
    rest = re.sub(POS_WORDS, " ", rest)
    rest = re.sub(NEG_WORDS, " ", rest)
    topics = [w for w in re.findall(r"[a-z0-9\-]+", rest) if w not in STOP and len(w) > 2]
    return {"type": want_type, "sentiment": sentiment, "topics": topics}


def build_index(mentions):
    """mentions: output of Tallie.extract() over many texts. One vote per comment per item."""
    groups, seen = defaultdict(list), set()
    for m in mentions:
        k = _norm(m["text"])
        if (k, m.get("doc")) in seen and m.get("doc") is not None:
            continue
        seen.add((k, m.get("doc")))
        groups[k].append(m)
    index = {}
    for k, ms in groups.items():
        types = Counter(m["type"] for m in ms if m.get("type"))
        t = types.most_common(1)
        index[k] = {"count": len(ms), "polarity": Counter(m["polarity"] for m in ms),
                    "surface": Counter(m["text"] for m in ms),
                    "type": t[0][0] if t else None,
                    "contexts": [(m["text"], m.get("context", "")) for m in ms[:6]]}
    for e in index.values():
        e["name"] = e["surface"].most_common(1)[0][0]
    return index


def answer(query, index, top=15, min_count=1, judge=None, judge_threshold=0.5):
    pq = parse_query(query)
    cands = []
    for e in index.values():
        if e["count"] < min_count:
            continue
        surf = [x for x, _ in e["surface"].most_common(3)]
        if all(len(x.split()) == 1 and x.islower() for x in surf) and e["count"] < 5:
            continue  # stray lower-case words ("aren't", "way")
        if pq["type"] and e["type"] not in TYPE_FAMILY.get(pq["type"], {pq["type"]}):
            continue
        pol = e["polarity"]; n = sum(pol.values()) or 1
        if pq["sentiment"]:
            k = pol.get(pq["sentiment"], 0)
            if k == 0 or k / n < (0.3 if pq["sentiment"] == "negative" else 0.4):
                continue  # 7 negative out of 109 mentions is not "hated"
            score = round(k * k / n, 2)
        else:
            score = e["count"]
        cands.append((e, score))
    if pq["topics"]:
        probs = judge(query, [e for e, _ in cands]) if judge and cands else {}
        cands = [(e, s) for e, s in cands if probs.get(id(e), 0) >= judge_threshold]
    rows = [{"item": e["name"], "type": e["type"], "score": s, "mentions": e["count"],
             "polarity": dict(e["polarity"]), "also_written_as": [x for x, _ in e["surface"].most_common(4)]}
            for e, s in cands]
    rows.sort(key=lambda r: (-r["score"], -r["mentions"]))
    return {"query": query, "parsed": pq, "results": rows[:top]}


def make_judge(model, tok, max_contexts=3, batch_size=32, max_len=256):
    """Model relevance for extra topic words: run the question over a few of the item's comment
    snippets and average the relevance score of that item's span."""
    import torch
    from .data import build_input, collate
    from .model import amp_dtype

    @torch.no_grad()
    def judge(query, entries):
        dev = next(model.parameters()).device
        jobs = [(id(e), name, ctx) for e in entries for name, ctx in e["contexts"][:max_contexts]]
        scores = defaultdict(list)
        for b0 in range(0, len(jobs), batch_size):
            batch = jobs[b0:b0 + batch_size]
            feats, metas = [], []
            for eid, name, ctx in batch:
                ids, offs, qm, tm, base = build_input(tok, query, ctx, max_len)
                feats.append({"input_ids": ids, "query_mask": qm, "bio": [0] * len(ids),
                              "spans": [], "rel": [], "pol": []})
                metas.append((eid, name, ctx, offs, base))
            x = {k: v.to(dev) for k, v in collate(feats, tok.pad_token_id).items()}
            idx = torch.zeros(len(batch), 1, 2, dtype=torch.long, device=dev)
            found = []
            for j, (eid, name, ctx, offs, base) in enumerate(metas):
                i = ctx.find(name)
                toks = [t for t, (s0, e0) in enumerate(offs) if i >= 0 and e0 > i + base and s0 < i + base + len(name)]
                if toks:
                    idx[j, 0, 0], idx[j, 0, 1] = toks[0], toks[-1]
                found.append(bool(toks))
            with torch.autocast("cuda", dtype=amp_dtype(), enabled=dev.type == "cuda"):
                h = model.encode(x["input_ids"], x["attention_mask"])
            rel, _, _ = model.heads(h, x["query_mask"], idx)
            for j, (eid, *_r) in enumerate(metas):
                if found[j]:
                    scores[eid].append(torch.sigmoid(rel[j, 0]).item())
        return {k: sum(v) / len(v) for k, v in scores.items()}
    return judge
