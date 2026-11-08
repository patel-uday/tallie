"""
Evaluate a saved Tallie model on test splits and measure speed.

  python evaluate.py --ckpt runs/tallie --data data/processed
"""
import argparse, json, sys, time
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from tallie.data import TallieDataset, collate
from tallie.infer import load, extract
from train import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/tallie")
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--max-len", type=int, default=384)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok = load(args.ckpt, device=dev)
    report = {"params": model.param_report()}
    for split in ("test", "test_heldout"):
        p = Path(args.data) / f"{split}.jsonl"
        if p.exists():
            dl = DataLoader(TallieDataset(p, tok, args.max_len), batch_size=16,
                            collate_fn=partial(collate, pad_id=tok.pad_token_id))
            report[split] = evaluate(model, dl, dev)
            print(split, report[split])

    # throughput: comments per second for one query
    rows = [json.loads(l) for l in open(Path(args.data) / "test.jsonl", encoding="utf-8")][:512]
    texts = [r["text"] for r in rows]
    extract(model, tok, "books people liked", texts[:32])  # warm-up
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    extract(model, tok, "books people liked", texts, batch_size=32)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    report["comments_per_second"] = round(len(texts) / (time.time() - t), 1)
    print("throughput:", report["comments_per_second"], "comments/s")
    (Path(args.ckpt) / "test_report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
