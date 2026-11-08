"""
Rebuild the hand-labeled test set (data/gold_test.jsonl) by downloading the comment texts from the
public Hacker News API. The repository ships only comment IDs + labels (data/gold_test_labels.jsonl),
not the comments themselves.

  python scripts/fetch_gold_text.py
"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scrape_hn import get, clean, API
from label_teacher import chunk_text

ROOT = Path(__file__).resolve().parents[1]


def main(inp=ROOT / "data/gold_test_labels.jsonl", out=ROOT / "data/gold_test.jsonl"):
    rows = [json.loads(l) for l in open(inp, encoding="utf-8")]
    ok = missing = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            cid, j = r["chunk_id"].rsplit("_", 1)
            item = get(f"{API}/items/{cid}")
            text = None
            if item and item.get("text"):
                chunks = chunk_text(clean(item["text"]))
                if int(j) < len(chunks):
                    text = chunks[int(j)]
            if text is None or not all(m["text"] in text for m in r["mentions"]):
                missing += 1  # comment deleted or edited since labeling
                continue
            f.write(json.dumps({**r, "text": text}, ensure_ascii=False) + "\n")
            ok += 1
            time.sleep(0.2)
    print(f"rebuilt {ok}/{len(rows)} test comments -> {out}  ({missing} unavailable)")


if __name__ == "__main__":
    main()
