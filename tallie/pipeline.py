"""
    from tallie import Tallie
    t = Tallie.load("path/to/model")
    t.index(comments)                         # one pass over your texts (list of strings)
    t.ask("podcasts people recommend")        # instant after indexing
    t.extract(["I loved Dune but Snow Crash bored me."])
"""
import torch

from .infer import load as load_model, extract_all
from .ranking import build_index, answer, make_judge


class Tallie:
    def __init__(self, model, tok, threshold=0.5):
        self.model, self.tok, self.threshold = model, tok, threshold
        self.judge = make_judge(model, tok)
        self._index = None

    @classmethod
    def load(cls, path, device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, tok = load_model(path, device=device)
        return cls(model, tok, threshold=getattr(model, "threshold", 0.5))

    def index(self, texts, batch_size=16):
        """Read every text once and keep every named item with its type and opinion."""
        self._index = build_index(extract_all(self.model, self.tok, list(texts), batch_size=batch_size))
        return self

    def ask(self, query, top=15):
        if self._index is None:
            raise RuntimeError("call .index(texts) first")
        return answer(query, self._index, top=top, judge=self.judge, judge_threshold=self.threshold)

    def extract(self, texts):
        """Raw output: every named item with its type and the commenter's opinion, per text."""
        return extract_all(self.model, self.tok, list(texts))
