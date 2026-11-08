"""CPU smoke test with a tiny random Qwen3 + a locally trained tokenizer (no downloads)."""
import json, random, subprocess, sys, tempfile
from functools import partial
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from tallie.model import TallieModel
from tallie.data import TallieDataset, collate
from tallie.infer import extract, rank, save, load, trim_vocab

tmp = Path(tempfile.mkdtemp())
random.seed(0); torch.manual_seed(0)

# ---- fake teacher labels -------------------------------------------------------------
BOOKS = [("Dune", ["space"], ["mind-blowing"]), ("The Once and Future King", ["King Arthur"], ["sad"]),
         ("SICP", ["programming"], ["hard"]), ("Le Morte d'Arthur", ["King Arthur"], ["boring"])]
MOVIES = [("Airplane!", ["comedy"], ["funny"]), ("Tenet", ["time travel"], ["confusing"]),
          ("Monty Python and the Holy Grail", ["King Arthur"], ["funny"])]
rows = []
for i in range(240):
    items = random.sample(BOOKS + MOVIES, 2)
    sents, ments = [], []
    for name, topics, attrs in items:
        pos = random.random() < 0.5
        sents.append(f"{name} was {'amazing, loved it' if pos else 'awful, total waste of time'}.")
        ments.append({"text": name, "type": "book" if (name, topics, attrs) in BOOKS else "movie",
                      "sentiment": "positive" if pos else "negative", "recommended": pos,
                      "attributes": attrs, "topics": topics})
    rows.append({"chunk_id": f"{i}_0", "thread_id": i % 30, "thread_title": "t",
                 "domain": "games" if i % 30 == 29 else "books", "text": " ".join(sents),
                 "mentions": ments,
                 "queries": [{"query": "things people loved", "relevant": [j for j, m in enumerate(ments) if m["sentiment"] == "positive"]}]})
(tmp / "labels.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
subprocess.run([sys.executable, str(ROOT / "scripts/build_dataset.py"), "--inp", str(tmp / "labels.jsonl"),
                "--outdir", str(tmp / "proc")], check=True)

# ---- tokenizer + tiny model -----------------------------------------------------------
tk = Tokenizer(models.BPE())
tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tk.decoder = decoders.ByteLevel()
corpus = [r["text"] for r in rows] + ["Query: books people liked about King Arthur funny movies hated"] * 5
tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=600, special_tokens=["<pad>"],
                                                     initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>")
cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=6,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=512)
bb = Qwen3Model(cfg)
bb.config._attn_implementation = "sdpa"
model = TallieModel(backbone=bb, keep_layers=4, bidirectional=True, lora_r=4, span_dim=64,
                     gradient_checkpointing=False, dtype=torch.float32)
print("params:", model.param_report())
assert model.param_report()["layers_kept"] == 4

# ---- bidirectionality check: changing the LAST token must change the FIRST token's state
ids = torch.tensor([tok("Dune was amazing, loved it.")["input_ids"]])
am = torch.ones_like(ids)
h1 = model.encode(ids, am)
ids2 = ids.clone(); ids2[0, -1] = (ids2[0, -1] + 7) % len(tok)
h2 = model.encode(ids2, am)
d = (h1[0, 0] - h2[0, 0]).abs().max().item()
print("bidirectional first-token change:", d); assert d > 1e-6, "attention is still causal!"
causal = TallieModel(backbone=Qwen3Model(cfg), keep_layers=4, bidirectional=False, lora_r=0,
                      span_dim=64, gradient_checkpointing=False)
c1, c2 = causal.encode(ids, am), causal.encode(ids2, am)
print("causal first-token change:", (c1[0, 0] - c2[0, 0]).abs().max().item())

# ---- padding check: a padded sequence gives the same states as unpadded -------------
pad = torch.cat([ids, torch.full((1, 5), tok.pad_token_id)], 1)
pam = torch.cat([am, torch.zeros(1, 5, dtype=torch.long)], 1)
hp = model.encode(pad, pam)[:, :ids.shape[1]]
print("padding diff:", (hp - h1).abs().max().item()); assert (hp - h1).abs().max() < 1e-4

# ---- train a few hundred steps --------------------------------------------------------
ds = TallieDataset(tmp / "proc/train.jsonl", tok, 128)
dl = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=partial(collate, pad_id=tok.pad_token_id))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-3)
model.train()
for ep in range(8):
    for x in dl:
        out = model(**x); out["loss"].backward(); opt.step(); opt.zero_grad()
    print(f"epoch {ep} loss {out['loss'].item():.3f} bio {out['l_bio']:.3f} rel {out['l_rel']:.3f} pol {out['l_pol']:.3f}")

sys.argv = ["x"]
from train import evaluate
vl = DataLoader(TallieDataset(tmp / "proc/val.jsonl", tok, 128), batch_size=8,
                collate_fn=partial(collate, pad_id=tok.pad_token_id))
print("val metrics:", evaluate(model, vl, torch.device("cpu")))

texts = ["Airplane! was amazing, loved it. Tenet was awful, total waste of time.",
         "Monty Python and the Holy Grail was amazing, loved it.",
         "Dune was awful, total waste of time."]
res = extract(model, tok, "things people loved", texts, max_len=128)
print("extract:", res)
print("rank:", rank(res))

# ---- save -> load round trip, then slim export ---------------------------------------
save(model, tok, tmp / "ckpt")
m2, tok2 = load(tmp / "ckpt", device="cpu", dtype=torch.float32)
r2 = extract(m2, tok2, "things people loved", texts, max_len=128)
assert [[x["text"] for x in a] for a in r2] == [[x["text"] for x in a] for a in res], "reload mismatch"
n = trim_vocab(m2, tok2, [r["text"] for r in rows] + texts + ["Query: things people loved\nText: "])
r3 = extract(m2, tok2, "things people loved", texts, max_len=128)
print("trimmed vocab ->", n, "rows; same output:",
      [[x["text"] for x in a] for a in r3] == [[x["text"] for x in a] for a in res])
save(m2, tok2, tmp / "slim"); m3, _ = load(tmp / "slim", device="cpu", dtype=torch.float32)
print("slim reload ok:", [[x["text"] for x in a] for a in extract(m3, tok2, "things people loved", texts, max_len=128)])
print("ALL SMOKE TESTS PASSED")

# ---- run the real train.py loop (with tiny model patched in) and test resume ----------
import train as T
T.AutoTokenizer.from_pretrained = staticmethod(lambda *a, **k: tok)
T.TallieModel = lambda *a, **k: TallieModel(backbone=Qwen3Model(cfg), keep_layers=4, lora_r=4,
                                              span_dim=64, gradient_checkpointing=False,
                                              dtype=torch.float32)
T.save = lambda *a, **k: Path(tmp / "run").mkdir(exist_ok=True)
base = ["train.py", "--data", str(tmp / "proc"), "--batch-size", "16", "--eval-every", "20",
        "--max-len", "128", "--out", str(tmp / "run")]
sys.argv = base + ["--epochs", "1"]; T.main()
ck = torch.load(str(tmp / "run") + "_last.pt"); print("checkpoint step:", ck["step"])
sys.argv = base + ["--epochs", "2"]; T.main()   # should print "resumed at step ..."
print("RESUME TEST PASSED")
