"""
Tallie: a non-generative "query -> list" model built by surgery on a pretrained LLM.

  1. Load Qwen3 WITHOUT its LM head (AutoModel gives the bare transformer).
  2. Prune the top layers (keep the first `keep_layers`).
  3. Switch causal attention to bidirectional, so every token sees the whole input.
  4. Add four new heads that read the final hidden states once:
       mention head   : BIO tag per token  -> where are the named items?
       relevance head : per span, given the query -> does it satisfy the request? (0..1)
       polarity head  : per span -> positive / negative / neutral / mixed
       type head      : per span -> book / movie / tv_show / video_game / ... (v1.0)
No decoding loop: one forward pass produces the whole answer.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

BIO_O, BIO_B, BIO_I = 0, 1, 2
POLARITY = ["positive", "negative", "neutral", "mixed"]
TYPES = ["book", "movie", "tv_show", "video_game", "board_game", "podcast", "music", "software",
         "product", "person", "other"]


def amp_dtype():
    """bf16 on Ampere+ (A100, L4), fp16 on older GPUs (T4), fp32 on CPU."""
    if not torch.cuda.is_available():
        return torch.float32
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


def mlp(i, h, o, p=0.1):
    return nn.Sequential(nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class TallieModel(nn.Module):
    def __init__(self, backbone_name="Qwen/Qwen3-1.7B", keep_layers=20, bidirectional=True,
                 lora_r=16, gradient_checkpointing=True, dtype=None, backbone=None,
                 span_dim=512, bio_weight=3.0, pol_class_weights=None):
        self_bio_weight = bio_weight
        if dtype is None:
            dtype = amp_dtype()
        super().__init__()
        self.bio_weight = self_bio_weight
        self.cfg = dict(backbone_name=backbone_name, keep_layers=keep_layers,
                        bidirectional=bidirectional, lora_r=lora_r, span_dim=span_dim)
        if backbone is None:
            backbone = AutoModel.from_pretrained(backbone_name, torch_dtype=dtype,
                                                 attn_implementation="sdpa")
        self.n_layers_original = len(backbone.layers)

        # --- surgery 1: prune top layers -------------------------------------------------
        if keep_layers and keep_layers < len(backbone.layers):
            backbone.layers = nn.ModuleList(list(backbone.layers)[:keep_layers])
            backbone.config.num_hidden_layers = keep_layers
            if getattr(backbone.config, "layer_types", None):
                backbone.config.layer_types = backbone.config.layer_types[:keep_layers]

        # --- surgery 2: bidirectional attention ------------------------------------------
        self.bidirectional = bidirectional
        if bidirectional:
            for layer in backbone.layers:
                layer.self_attn.is_causal = False

        if gradient_checkpointing:
            backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            backbone.enable_input_require_grads()

        if lora_r:
            from peft import LoraConfig, get_peft_model
            backbone = get_peft_model(backbone, LoraConfig(
                r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"]))
            for p in backbone.parameters():  # trainable LoRA weights in fp32 (stable with fp16 AMP)
                if p.requires_grad:
                    p.data = p.data.float()
        self.backbone = backbone

        H = backbone.config.hidden_size
        D = span_dim
        # --- new heads (kept in fp32) ----------------------------------------------------
        self.mention_head = mlp(H, H // 2, 3)
        self.span_proj = mlp(3 * H, D, D)
        self.query_proj = nn.Linear(H, D)
        self.relevance_head = mlp(3 * D, D, 1)
        self.polarity_head = mlp(D, D // 2, len(POLARITY))
        self.type_head = mlp(D, D // 2, len(TYPES))
        # inverse-frequency weights so the polarity head isn't biased toward "positive"
        self.register_buffer("pol_w", torch.tensor(pol_class_weights or [1.0] * len(POLARITY)), persistent=False)
        self.register_buffer("vocab_map", None)  # optional: set by trim_vocab() for the slim export

    # ------------------------------------------------------------------------------------
    def _attn_mask(self, attention_mask, dtype):
        if not self.bidirectional:
            return attention_mask  # model builds the usual causal mask
        # 4D additive mask: every position may attend to every non-pad position.
        B, L = attention_mask.shape
        m = torch.zeros(B, 1, L, L, dtype=dtype, device=attention_mask.device)
        return m.masked_fill(attention_mask[:, None, None, :] == 0, torch.finfo(dtype).min)

    def encode(self, input_ids, attention_mask):
        if self.vocab_map is not None:
            input_ids = self.vocab_map[input_ids]
        emb = self.backbone.get_input_embeddings()
        dtype = emb.weight.dtype
        out = self.backbone(input_ids=input_ids,
                            attention_mask=self._attn_mask(attention_mask, dtype),
                            use_cache=False)
        return out.last_hidden_state.float()

    @staticmethod
    def span_reps(h, span_idx):
        """h: (B,L,H); span_idx: (B,S,2) inclusive token [start,end]. Returns (B,S,3H)."""
        B, L, H = h.shape
        s = span_idx[..., 0].clamp(0, L - 1)
        e = span_idx[..., 1].clamp(0, L - 1)
        gather = lambda idx: torch.gather(h, 1, idx.unsqueeze(-1).expand(-1, -1, H))
        cs = torch.cat([h.new_zeros(B, 1, H), h.cumsum(1)], 1)  # prefix sums
        tot = torch.gather(cs, 1, (e + 1).unsqueeze(-1).expand(-1, -1, H)) - \
              torch.gather(cs, 1, s.unsqueeze(-1).expand(-1, -1, H))
        mean = tot / (e - s + 1).clamp(min=1).unsqueeze(-1).float()
        return torch.cat([gather(s), gather(e), mean], -1)

    def heads(self, h, query_mask, span_idx):
        qm = query_mask.unsqueeze(-1).float()
        q = self.query_proj((h * qm).sum(1) / qm.sum(1).clamp(min=1))         # (B,D)
        sp = self.span_proj(self.span_reps(h, span_idx))                       # (B,S,D)
        qe = q.unsqueeze(1).expand_as(sp)
        rel = self.relevance_head(torch.cat([sp, qe, sp * qe], -1)).squeeze(-1)  # (B,S)
        pol = self.polarity_head(sp)                                             # (B,S,4)
        typ = self.type_head(sp)                                                 # (B,S,11)
        return rel, pol, typ

    def forward(self, input_ids, attention_mask, query_mask, span_idx, span_valid,
                bio_labels=None, rel_labels=None, pol_labels=None, type_labels=None,
                pol_weight=0.5, type_weight=0.5):
        h = self.encode(input_ids, attention_mask)
        bio_logits = self.mention_head(h)
        rel, pol, typ = self.heads(h, query_mask, span_idx)
        out = {"bio_logits": bio_logits, "rel_logits": rel, "pol_logits": pol, "type_logits": typ}
        if bio_labels is not None:
            # names are rare tokens: up-weight B/I so the model doesn't learn "everything is O"
            w = bio_logits.new_tensor([1.0, self.bio_weight, self.bio_weight])
            l_bio = F.cross_entropy(bio_logits.view(-1, 3), bio_labels.view(-1), weight=w,
                                    ignore_index=-100)
            v = span_valid.bool()
            if v.any():
                l_rel = F.binary_cross_entropy_with_logits(rel[v], rel_labels[v].float())
                l_pol = F.cross_entropy(pol[v], pol_labels[v], weight=self.pol_w.to(pol.dtype))
                if type_labels is not None and (type_labels[v] >= 0).any():
                    l_typ = F.cross_entropy(typ[v], type_labels[v], ignore_index=-100)
                else:
                    l_typ = h.sum() * 0
            else:
                l_rel = l_pol = l_typ = h.sum() * 0
            out.update(loss=l_bio + l_rel + pol_weight * l_pol + type_weight * l_typ,
                       l_bio=l_bio.detach(), l_rel=l_rel.detach(), l_pol=l_pol.detach(),
                       l_typ=l_typ.detach())
        return out

    # ------------------------------------------------------------------------------------
    @staticmethod
    def decode_bio(tags, text_mask):
        """tags: list[int] per token. Returns list of (start_tok, end_tok) inclusive."""
        spans, start = [], None
        for i, (t, ok) in enumerate(zip(tags, text_mask)):
            if not ok:
                t = BIO_O
            if t == BIO_B or (t == BIO_I and start is None):
                if start is not None:
                    spans.append((start, i - 1))
                start = i
            elif t == BIO_O and start is not None:
                spans.append((start, i - 1))
                start = None
        if start is not None:
            spans.append((start, len(tags) - 1))
        return spans

    def head_params(self):
        return [p for n, p in self.named_parameters() if not n.startswith("backbone.")]

    def param_report(self):
        bb = self.backbone
        base = bb.get_base_model() if hasattr(bb, "get_base_model") else bb
        emb = base.embed_tokens.weight.numel()
        total_bb = sum(p.numel() for n, p in base.named_parameters() if "lora_" not in n)
        heads = sum(p.numel() for p in self.head_params())
        return {"layers_kept": len(base.layers), "layers_original": self.n_layers_original,
                "embedding_params": emb, "backbone_params": total_bb, "head_params": heads,
                "total_params": total_bb + heads}

