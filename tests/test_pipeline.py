"""End-to-end pipeline with a tiny model (code path check)."""
import sys, tempfile, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3Model
from tallie.model import TallieModel
from tallie.infer import save
from tallie import Tallie
texts = ["I read Dune and loved it. Airplane! was hilarious.", "Firefox is fine but Chrome eats RAM."] * 6
tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
tk.train_from_iterator(texts * 3 + ["Query: all named items mentioned funny movies\nText: "], trainers.BpeTrainer(vocab_size=300, special_tokens=["<pad>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
tok = PreTrainedTokenizerFast(tokenizer_object=tk, pad_token="<pad>")
cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2, head_dim=16)
m = TallieModel(backbone=Qwen3Model(cfg), keep_layers=2, lora_r=0, span_dim=32, gradient_checkpointing=False, dtype=torch.float32)
with torch.no_grad(): m.mention_head[-1].bias.copy_(torch.tensor([0.0, 2.0, 1.0]))
d = Path(tempfile.mkdtemp()); save(m, tok, d / "ck", extra={"threshold": 0.4})
v = Tallie.load(d / "ck", device="cpu")
assert v.threshold == 0.4
v.index(texts)
r = v.ask("funny movies"); print(r["parsed"], len(r["results"]))
r = v.ask("things people recommend"); print(len(r["results"]))
print("PIPELINE TEST PASSED")
