"""
Scrape opinion-heavy Hacker News threads ("Ask HN: favorite books", "funniest movies", ...)
using the free Algolia HN API. No API key needed.

Output: data/raw/hn_comments.jsonl, one comment per line:
  {"thread_id", "thread_title", "domain", "comment_id", "text"}

Usage:
  python scrape_hn.py --out data/raw/hn_comments.jsonl --max-threads-per-query 6
"""
import argparse, html, json, re, time
from pathlib import Path

import requests

API = "https://hn.algolia.com/api/v1"

# (search query, domain label). Domains are only used to make sure the dataset is diverse
# and to hold one domain out for the generalization test. The model itself never sees them.
SEARCHES = [
    # books
    ("favorite books", "books"), ("books that changed your life", "books"),
    ("best science fiction books", "books"), ("best fantasy books", "books"),
    ("best non-fiction books", "books"), ("books you regret reading", "books"),
    ("best programming books", "books"), ("overrated books", "books"),
    ("best history books", "books"), ("what are you reading", "books"),
    # movies
    ("favorite movies", "movies"), ("funniest movies", "movies"),
    ("best documentaries", "movies"), ("best sci-fi movies", "movies"),
    ("movies about hackers", "movies"), ("worst movies", "movies"),
    ("underrated movies", "movies"), ("best horror movies", "movies"),
    # tv
    ("best tv shows", "tv"), ("favorite tv series", "tv"), ("best anime", "tv"),
    # podcasts / music
    ("favorite podcasts", "podcasts"), ("best podcasts", "podcasts"),
    ("music for programming", "music"), ("favorite albums", "music"),
    # software / products (shows the model is not only about media)
    ("favorite tools", "tools"), ("best self-hosted software", "tools"),
    ("favorite programming language", "tools"), ("text editor do you use", "tools"),
    # HELD-OUT domain: video & board games. Never used for training (see build_dataset.py).
    ("favorite video games", "games"), ("best video games", "games"),
    ("best board games", "games"), ("indie games", "games"),
]

TAG_RE = re.compile(r"<[^>]+>")


def clean(text_html: str) -> str:
    t = text_html.replace("<p>", "\n\n")
    t = re.sub(r"<pre><code>.*?</code></pre>", " ", t, flags=re.S)  # drop code blocks
    t = TAG_RE.sub("", t)
    t = html.unescape(t)
    return re.sub(r"[ \t]+", " ", t).strip()


def get(url, params=None, tries=4):
    for i in range(tries):
        try:
            r = requests.get(url, params=params, timeout=30,
                             headers={"User-Agent": "tallie-research-scraper/1.0"})
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            pass
        time.sleep(2 ** i)
    return None


def find_threads(query, n, min_comments):
    hits = []
    for tags in ("ask_hn", "story"):
        res = get(f"{API}/search", {"query": query, "tags": tags, "hitsPerPage": 50,
                                    "numericFilters": f"num_comments>{min_comments}"})
        if res:
            hits += res.get("hits", [])
        if len(hits) >= n:
            break
    hits.sort(key=lambda h: h.get("num_comments") or 0, reverse=True)
    return hits[:n]


def flatten(item, out):
    for child in item.get("children") or []:
        txt = child.get("text")
        if txt and child.get("type") == "comment":
            out.append((child["id"], clean(txt)))
        flatten(child, out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw/hn_comments.jsonl")
    ap.add_argument("--max-threads-per-query", type=int, default=6)
    ap.add_argument("--min-comments", type=int, default=60)
    ap.add_argument("--min-chars", type=int, default=40)
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen_threads, seen_comments, n_written = set(), set(), 0

    with out.open("w", encoding="utf-8") as f:
        for query, domain in SEARCHES:
            threads = find_threads(query, args.max_threads_per_query, args.min_comments)
            for h in threads:
                tid = int(h["objectID"])
                if tid in seen_threads:
                    continue
                seen_threads.add(tid)
                item = get(f"{API}/items/{tid}")
                if not item:
                    continue
                comments = []
                flatten(item, comments)
                kept = 0
                for cid, text in comments:
                    if cid in seen_comments or len(text) < args.min_chars:
                        continue
                    seen_comments.add(cid)
                    f.write(json.dumps({"thread_id": tid, "thread_title": h.get("title", ""),
                                        "domain": domain, "comment_id": cid,
                                        "text": text}, ensure_ascii=False) + "\n")
                    kept += 1
                n_written += kept
                print(f"[{domain:8s}] {kept:5d} comments  <- {h.get('title','')[:70]}")
                time.sleep(0.3)
    print(f"\nDone: {len(seen_threads)} threads, {n_written} comments -> {out}")


if __name__ == "__main__":
    main()
