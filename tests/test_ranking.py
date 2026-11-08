"""Ranking logic: questions become type + opinion filters; items are tallied one vote per comment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tallie.ranking import build_index, answer, parse_query

M = lambda t, typ, pol, doc: {"text": t, "type": typ, "polarity": pol, "doc": doc, "context": t}
mentions = ([M("Planet Money", "podcast", "positive", d) for d in range(6)] +
            [M("Planet Money", "podcast", "positive", 0)] +            # same comment again: no extra vote
            [M("Radiolab", "podcast", "positive", d) for d in range(10, 13)] +
            [M("1Q84", "book", "negative", d) for d in range(20, 24)] + [M("1Q84", "book", "positive", 24)] +
            [M("Dune", "book", "positive", d) for d in range(30, 50)] + [M("Dune", "book", "negative", 51)] +
            [M("aren't", "podcast", "negative", 60)])
idx = build_index(mentions)
names = lambda q: [r["item"] for r in answer(q, idx)["results"]]
assert parse_query("most hated books") == {"type": "book", "sentiment": "negative", "topics": []}
assert parse_query("board games people recommend")["type"] == "board_game"
assert idx["planet money"]["count"] == 6
assert names("podcasts people recommend") == ["Planet Money", "Radiolab"]
assert names("most hated books") == ["1Q84"]            # Dune: 1 negative out of 21 is not "hated"
assert names("podcasts people regret listening to") == []  # stray lower-case word filtered
print("RANKING TEST PASSED")
