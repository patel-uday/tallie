"""Builds notebooks/train_tallie.ipynb: the steps and settings that trained Tallie 1.0 (run from the repo root)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()})

md(r"""
# Train Tallie 1.0

The steps and settings used to train the released model:
Qwen3-1.7B → 20 of 28 layers, bidirectional attention, mention / relevance / opinion / type heads, LoRA (r=16), 3 epochs, relevance cutoff tuned on validation.

**Needs, in Google Drive under `MyDrive/Tallie/`:**
- `data/labeled/labels_clean.jsonl`: labeled comments (format produced by `scripts/label_teacher.py`)
- `data/gold_test.jsonl`: the hand-labeled test set (`scripts/fetch_gold_text.py` builds it)

**GPU:** L4 or A100 (Runtime → Change runtime type). Keep at least 8 GB free in Drive.
""")
code(r"""
from google.colab import drive
drive.mount('/content/drive')
import os, json, sys
os.chdir("/content/drive/MyDrive/Tallie"); print("working in", os.getcwd())
!nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
need = ["data/labeled/labels_clean.jsonl", "data/gold_test.jsonl"]
missing = [p for p in need if not os.path.exists(p)]
print("missing files:", missing or "none")
assert not missing
!pip -q install -U "transformers>=4.55" "peft>=0.15" accelerate
!pip -q uninstall -y torchao
!mkdir -p /content/repo/tallie /content/repo/scripts
""")
md("## Code")
files = sorted((ROOT / "tallie").glob("*.py")) + [ROOT / "scripts" / n for n in
        ("build_dataset.py", "train.py", "evaluate.py", "eval_gold.py")]
for f in files:
    code(f"%%writefile /content/repo/{f.relative_to(ROOT).as_posix()}\n" + f.read_text(encoding="utf-8"))
code(r"""
REPO = "/content/repo"; sys.path.insert(0, REPO); sys.path.insert(0, f"{REPO}/scripts")
import torch
mem = torch.cuda.get_device_properties(0).total_memory / 2**30
BATCH, ACCUM, MAX_LEN = (16, 1, 384) if mem > 38 else (8, 2, 384) if mem > 20 else (8, 2, 256)
EPOCHS, LAYERS = 3, 20
RUN = "runs/tallie"
print(f"{torch.cuda.get_device_name(0)} ({mem:.0f} GB): batch {BATCH}x{ACCUM}, max_len {MAX_LEN}, {EPOCHS} epochs -> {RUN}")
""")
md("## 1. Build the training set")
code(r"""
!python {REPO}/scripts/build_dataset.py --inp data/labeled/labels_clean.jsonl --outdir data/processed
import collections
rows = [json.loads(l) for l in open("data/processed/train.jsonl")]
print("empty-answer share:", round(sum(1 for r in rows if not any(s["relevant"] for s in r["spans"])) / len(rows), 2))
print("types:", collections.Counter(s["type"] for r in rows for s in r["spans"]).most_common())
""")
md("## 2. Train (~40 min on an L4; re-run the cell to resume after a disconnect)")
code(r"""
!python {REPO}/scripts/train.py --data data/processed --backbone Qwen/Qwen3-1.7B --keep-layers {LAYERS} --epochs {EPOCHS} --batch-size {BATCH} --grad-accum {ACCUM} --max-len {MAX_LEN} --eval-every 300 --out {RUN}
!ls -la {RUN}/backbone   # model.safetensors should be ~2.5 GB
""")
md("## 3. Test split, held-out games topic, speed")
code(r"""
!python {REPO}/scripts/evaluate.py --ckpt {RUN} --data data/processed --max-len {MAX_LEN}
""")
md("## 4. Hand-labeled test set")
code(r"""
!python {REPO}/scripts/eval_gold.py --gold data/gold_test.jsonl --ckpt {RUN}
""")
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"gpuType": "L4"},
      "kernelspec": {"display_name": "Python 3", "name": "python3"}, "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}
for c in nb["cells"]:
    c["source"] = c["source"].splitlines(keepends=True)
out = ROOT / "notebooks" / "train_tallie.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote", out, len(cells), "cells")
