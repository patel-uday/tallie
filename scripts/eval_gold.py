"""
Score models (and the teacher) against the HAND-LABELED test set (data/gold_test.jsonl).
This is the trustworthy number; the old test split is scored against noisy teacher labels.

  python eval_gold.py --gold data/gold_test.jsonl --ckpt runs/tallie \
                      --teacher data/labeled/labels.jsonl
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from tallie.data import build_input, collate
from tallie.filters import clean_span, is_valid_item
from tallie.infer import load, normalize
from tallie.model import amp_dtype, POLARITY, TYPES


def norm(t):
    return normalize(t).replace("’", "'")


def f1(tp, fp, fn):
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(2 * p * r / max(1e-9, p + r), 3)}


@torch.no_grad()
def predict(model, tok, query, text, max_len=384):
    """All predicted mentions with (relevance score, polarity) for one (query, text)."""
    dev = next(model.parameters()).device
    ids, offs, qm, tm, base = build_input(tok, query, text, max_len)
    x = collate([{"input_ids": ids, "query_mask": qm, "bio": [0] * len(ids), "spans": [], "rel": [], "pol": []}],
                tok.pad_token_id)
    x = {k: v.to(dev) for k, v in x.items()}
    with torch.autocast("cuda", dtype=amp_dtype(), enabled=dev.type == "cuda"):
        h = model.encode(x["input_ids"], x["attention_mask"])
    tags = model.mention_head(h).argmax(-1)[0].tolist()
    spans = model.decode_bio(tags, tm)
    if not spans:
        return []
    rel, pol, typ = model.heads(h, x["query_mask"], torch.tensor([spans], device=dev))
    out = []
    for (a, b), r, p, t in zip(spans, torch.sigmoid(rel[0]).tolist(), pol[0].argmax(-1).tolist(),
                               typ[0].argmax(-1).tolist()):
        fx = clean_span(text, max(offs[a][0] - base, 0), offs[b][1] - base)
        if fx and is_valid_item(text[fx[0]:fx[1]].strip("\"'“”‘’")):
            out.append({"text": text[fx[0]:fx[1]].strip("\"'“”‘’"), "score": r, "polarity": POLARITY[p],
                        "type": TYPES[t]})
    return out


def score_model(model, tok, gold, threshold):
    S = defaultdict(lambda: defaultdict(int))
    examples = []
    for g in gold:
        dom = "games (held out)" if g["domain"] == "games" else "seen domains"
        gm = {norm(m["text"]): m for m in g["mentions"]}
        preds_any = None
        for q in g["queries"]:
            preds = predict(model, tok, q["query"], g["text"])
            if preds_any is None:  # names found (query-independent part), scored once per comment
                preds_any = preds
                pm = {norm(p["text"]): p for p in preds}
                for k in ("all", dom):
                    S[k]["m_tp"] += len(set(pm) & set(gm)); S[k]["m_fp"] += len(set(pm) - set(gm))
                    S[k]["m_fn"] += len(set(gm) - set(pm))
                    for n in set(pm) & set(gm):
                        S[k]["pol_n"] += 1; S[k]["pol_ok"] += int(pm[n]["polarity"] == gm[n]["sentiment"])
                        S[k]["typ_n"] += 1; S[k]["typ_ok"] += int(pm[n].get("type") == gm[n].get("type"))
            pr = {norm(p["text"]) for p in preds if p["score"] >= threshold}
            gr = {norm(t) for t in q["relevant"]}
            for k in ("all", dom):
                S[k]["e_tp"] += len(pr & gr); S[k]["e_fp"] += len(pr - gr); S[k]["e_fn"] += len(gr - pr)
            if len(examples) < 40 and (pr != gr):
                examples.append({"query": q["query"], "gold": sorted(gr), "model": sorted(pr),
                                 "text": g["text"][:300]})
    rep = {}
    for k, s in S.items():
        rep[k] = {"names_found": f1(s["m_tp"], s["m_fp"], s["m_fn"]),
                  "end_to_end": f1(s["e_tp"], s["e_fp"], s["e_fn"]),
                  "opinion_accuracy": round(s["pol_ok"] / max(1, s["pol_n"]), 3),
                  "type_accuracy": round(s["typ_ok"] / max(1, s["typ_n"]), 3)}
    return rep, examples


def score_teacher(gold, labels_path):
    lab = {}
    for l in open(labels_path, encoding="utf-8"):
        r = json.loads(l); lab[r["chunk_id"]] = r
    S = defaultdict(lambda: defaultdict(int)); n = 0
    for g in gold:
        t = lab.get(g["chunk_id"])
        if t is None:
            continue
        n += 1
        dom = "games (held out)" if g["domain"] == "games" else "seen domains"
        gm = {norm(m["text"]): m for m in g["mentions"]}
        tm = {norm(m["text"]): m for m in t["mentions"]}
        for k in ("all", dom):
            S[k]["tp"] += len(set(tm) & set(gm)); S[k]["fp"] += len(set(tm) - set(gm)); S[k]["fn"] += len(set(gm) - set(tm))
            for x in set(tm) & set(gm):
                S[k]["pn"] += 1; S[k]["pok"] += int(tm[x]["sentiment"] == gm[x]["sentiment"])
    return {"comments_covered": n,
            **{k: {"names_found": f1(s["tp"], s["fp"], s["fn"]),
                   "opinion_accuracy": round(s["pok"] / max(1, s["pn"]), 3)} for k, s in S.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", default="data/gold_test.jsonl")
    ap.add_argument("--ckpt", nargs="*", default=[])
    ap.add_argument("--teacher", nargs="*", default=[], help="labels.jsonl files to score as teachers")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default="gold_report.json")
    a = ap.parse_args()
    gold = [json.loads(l) for l in open(a.gold, encoding="utf-8")]
    report = {}
    for t in a.teacher:
        report[f"teacher: {t}"] = score_teacher(gold, t)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for c in a.ckpt:
        if not Path(c).exists():
            print("skip (not found):", c); continue
        model, tok = load(c, device=dev); model.eval()
        rep, ex = score_model(model, tok, gold, a.threshold)
        report[f"model: {c}"] = rep
        report[f"model: {c} | example mistakes"] = ex
        del model; torch.cuda.empty_cache() if dev == "cuda" else None
    json.dump(report, open(a.out, "w"), indent=1, ensure_ascii=False)
    for k, v in report.items():
        if "example mistakes" in k:
            continue
        print("\n###", k)
        print(json.dumps(v, indent=1))
    print("\nfull report ->", a.out)




if __name__ == "__main__":
    main()
