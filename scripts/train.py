"""
Train Tallie.

  python train.py --data data/processed --backbone Qwen/Qwen3-1.7B --keep-layers 20 --out runs/tallie
"""
import argparse, json, math, sys, time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from tallie.model import TallieModel, amp_dtype
from tallie.data import TallieDataset, collate
from tallie.infer import save


@torch.no_grad()
def evaluate(model, loader, device, max_batches=0, thr=0.5):
    model.eval()
    import math
    logit_thr = math.log(thr / (1 - thr))
    typ_ok = typ_n = 0
    tp = fp = fn = 0            # mention spans
    e2e_tp = e2e_fp = e2e_fn = 0  # end-to-end: predicted span AND judged relevant
    rel_ok = rel_n = pol_ok = pol_n = 0
    for bi, x in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        x = {k: v.to(device) for k, v in x.items()}
        with torch.autocast("cuda", dtype=amp_dtype(), enabled=device.type == "cuda"):
            out = model(**{k: x[k] for k in ("input_ids", "attention_mask", "query_mask",
                                              "span_idx", "span_valid")})
        v = x["span_valid"].bool()
        rel_pred = out["rel_logits"] > 0
        rel_ok += (rel_pred[v] == x["rel_labels"][v].bool()).sum().item(); rel_n += v.sum().item()
        pol_ok += (out["pol_logits"].argmax(-1)[v] == x["pol_labels"][v]).sum().item(); pol_n += v.sum().item()
        tv = v & (x["type_labels"] >= 0)
        typ_ok += (out["type_logits"].argmax(-1)[tv] == x["type_labels"][tv]).sum().item(); typ_n += tv.sum().item()

        tags = out["bio_logits"].argmax(-1).tolist()
        text_mask = (x["bio_labels"] != -100) & (x["query_mask"] == 0)
        with torch.autocast("cuda", dtype=amp_dtype(), enabled=device.type == "cuda"):
            h = model.encode(x["input_ids"], x["attention_mask"])
        for i in range(len(tags)):
            gold = {tuple(s) for s, ok in zip(x["span_idx"][i].tolist(), v[i].tolist()) if ok}
            gold_rel = {tuple(s) for s, ok, r in zip(x["span_idx"][i].tolist(), v[i].tolist(),
                                                      x["rel_labels"][i].tolist()) if ok and r}
            pred = set(model.decode_bio(tags[i], text_mask[i].tolist()))
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
            if pred:
                pl = sorted(pred)
                idx = torch.tensor([pl], device=device)
                r, _, _ = model.heads(h[i:i + 1], x["query_mask"][i:i + 1], idx)
                pred_rel = {s for s, sc in zip(pl, r[0].tolist()) if sc > logit_thr}
            else:
                pred_rel = set()
            e2e_tp += len(pred_rel & gold_rel); e2e_fp += len(pred_rel - gold_rel); e2e_fn += len(gold_rel - pred_rel)
    f1 = lambda a, b, c: 2 * a / max(1, 2 * a + b + c)
    model.train()
    return {"mention_f1": round(f1(tp, fp, fn), 4),
            "relevance_acc": round(rel_ok / max(1, rel_n), 4),
            "polarity_acc": round(pol_ok / max(1, pol_n), 4),
            "type_acc": round(typ_ok / max(1, typ_n), 4),
            "end_to_end_f1": round(f1(e2e_tp, e2e_fp, e2e_fn), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--backbone", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--keep-layers", type=int, default=20)
    ap.add_argument("--causal", action="store_true", help="ablation: keep causal attention")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--bio-weight", type=float, default=3.0, help="extra weight on name tokens")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--out", default="runs/tallie")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.backbone)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    # balanced opinion training: inverse-frequency weights (labels lean heavily positive)
    from collections import Counter
    pc = Counter(sp["polarity"] for l in open(Path(args.data) / "train.jsonl", encoding="utf-8")
                 for sp in json.loads(l)["spans"])
    tot = sum(pc.values()) or 1
    pol_w = [min(4.0, (tot / (4 * max(1, pc.get(i, 0)))) ** 0.5) for i in range(4)]
    print("polarity counts", dict(pc), "-> loss weights", [round(w, 2) for w in pol_w])
    model = TallieModel(args.backbone, keep_layers=args.keep_layers,
                         bidirectional=not args.causal, lora_r=args.lora_r,
                         bio_weight=args.bio_weight, pol_class_weights=pol_w).to(device)
    print(json.dumps(model.param_report(), indent=2))

    D = Path(args.data)
    train = TallieDataset(D / "train.jsonl", tok, args.max_len, args.limit)
    val = TallieDataset(D / "val.jsonl", tok, args.max_len)
    col = partial(collate, pad_id=tok.pad_token_id)
    vl = DataLoader(val, batch_size=args.batch_size, collate_fn=col)

    lora = [p for n, p in model.named_parameters() if n.startswith("backbone.") and p.requires_grad]
    opt = torch.optim.AdamW([{"params": lora, "lr": args.lr},
                             {"params": model.head_params(), "lr": args.head_lr}], weight_decay=0.01)
    batches_per_epoch = math.ceil(len(train) / args.batch_size)
    steps = math.ceil(batches_per_epoch / args.grad_accum) * args.epochs
    sched = get_cosine_schedule_with_warmup(opt, int(0.05 * steps), steps)
    adt = amp_dtype()
    scaler = torch.amp.GradScaler("cuda", enabled=adt == torch.float16)
    print("mixed precision:", adt)

    trainable = lambda: {k: v for k, v in model.state_dict().items()
                         if "lora_" in k or not k.startswith("backbone.")}
    best, step, start_ep, skip, t0 = -1, 0, 0, 0, time.time()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    last_path = Path(args.out + "_last.pt")
    if last_path.exists():  # resume after a Colab disconnect
        ck = torch.load(last_path, map_location=device)
        model.load_state_dict(ck["model"], strict=False)
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, step = ck["best"], ck["step"]
        start_ep, skip = divmod(step * args.grad_accum, batches_per_epoch)
        print(f"resumed at step {step} (epoch {start_ep}, batch {skip})")
    log = open(args.out + ".log", "a")
    model.train()
    for ep in range(start_ep, args.epochs):
        g = torch.Generator().manual_seed(1000 + ep)  # same order on resume
        tl = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=g,
                        collate_fn=col, num_workers=2)
        for bi, x in enumerate(tl):
            if ep == start_ep and bi < skip:
                continue
            x = {k: v.to(device) for k, v in x.items()}
            with torch.autocast("cuda", dtype=adt, enabled=device.type == "cuda"):
                out = model(**x)
            scaler.scale(out["loss"] / args.grad_accum).backward()
            if (bi + 1) % args.grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad(set_to_none=True); step += 1
            if step % 50 == 0:
                msg = (f"ep {ep} step {step}/{steps} loss {out['loss'].item():.3f} "
                       f"(bio {out['l_bio']:.3f} rel {out['l_rel']:.3f} pol {out['l_pol']:.3f}) "
                       f"{(time.time() - t0) / 60:.1f} min")
                print(msg); log.write(msg + "\n"); log.flush()
            if step % args.eval_every == 0 or step == steps:
                m = evaluate(model, vl, device, max_batches=100)
                print("VAL", m); log.write(f"VAL step {step} {json.dumps(m)}\n"); log.flush()
                if m["end_to_end_f1"] > best:
                    best = m["end_to_end_f1"]
                    torch.save(trainable(), Path(args.out + "_best_trainable.pt"))
                torch.save({"model": trainable(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "best": best, "step": step}, last_path)
    # load best trainable weights back, then merge LoRA and save the final model
    best_path = Path(args.out + "_best_trainable.pt")
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device), strict=False)
    # pick the relevance cutoff that maximizes end-to-end F1 on the validation set
    best_thr, final = 0.5, None
    for thr in (0.5, 0.3, 0.35, 0.4, 0.45, 0.55, 0.6, 0.65):  # 0.5 first: ties keep the default
        m = evaluate(model, vl, device, thr=thr)
        print(f"threshold {thr:.2f}: {m}")
        if final is None or m["end_to_end_f1"] > final["end_to_end_f1"]:
            best_thr, final = thr, m
    final["threshold"] = best_thr
    print("FINAL VAL", final)
    save(model, tok, args.out, extra={"threshold": best_thr})
    (Path(args.out) / "val_metrics.json").write_text(json.dumps(final, indent=2))
    print("saved ->", args.out)


if __name__ == "__main__":
    main()
