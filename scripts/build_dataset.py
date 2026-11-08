"""
Turn teacher labels into training examples for Tallie.

Each example = (query, text, spans). Every mention span is labeled with:
  relevant (0/1) -> does it satisfy THIS query?
  polarity       -> positive / negative / neutral / mixed (the commenter's opinion)

Two sources of queries per chunk:
  1. the teacher's free-form queries (reasoning-heavy, diverse wording)
  2. templated queries computed from the mention labels (exact, cheap, many paraphrases),
     including "negative" queries that match nothing, so the model learns to say "none".

Splits: by THREAD (no leakage). The "games" domain is held out entirely -> test_heldout.jsonl.
"""
import argparse, json, random, re
from collections import defaultdict
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).parent))
from tallie.filters import is_valid_item, find_all

PLURAL = {"book": "books", "movie": "movies", "tv_show": "TV shows", "video_game": "video games",
          "board_game": "board games", "podcast": "podcasts", "music": "albums or songs",
          "software": "software tools", "product": "products", "person": "people (authors, directors, artists)",
          "other": "things"}
POL = {"positive": 0, "negative": 1, "neutral": 2, "mixed": 3}
# several ways people name each type, so the model learns "films" == "movies", "apps" == "software"
TYPE_WORDS = {"book": ["books", "novels", "books", "reads"], "movie": ["movies", "films", "movies"],
              "tv_show": ["TV shows", "TV series", "shows"], "video_game": ["video games", "games", "video games"],
              "board_game": ["board games", "tabletop games"], "podcast": ["podcasts"],
              "music": ["albums", "songs", "bands", "music"], "software": ["software", "tools", "apps", "programs"],
              "product": ["products", "gadgets", "devices"], "person": ["authors", "people", "creators"],
              "other": ["things"]}


def tword(t, rng):
    return rng.choice(TYPE_WORDS.get(t, ["things"]))


def singular(w):
    return w[:-1] if w.endswith("s") else w

T_LIKED = ["{t} people liked", "{t} that were praised", "most loved {t}", "{t} recommended by commenters",
           "which {t} do people enjoy?", "positive mentions of {t}", "{t} people say are great"]
T_HATED = ["{t} people disliked", "most hated {t}", "{t} that were criticized", "{t} people regret",
           "negative opinions about {t}", "which {t} do people think are bad?"]
T_ALL = ["all {t} mentioned", "list every {t_s} that comes up", "{t} discussed here", "every {t_s} named"]
T_ATTR = ["{t} people found {a}", "{a} {t}", "{t} described as {a}", "which {t} are {a}?"]
T_TOPIC = ["{t} related to {p}", "{t} about {p}", "anything about {p}", "{t} on the topic of {p}"]
T_REC = ["things people recommend", "suggestions from commenters", "what should I check out?"]
T_REC_TYPE = ["{t} people recommend", "{t} worth trying according to commenters", "recommended {t}",
              "any good {t}?", "what {t} do people recommend?", "{t} worth checking out"]
T_TOPIC_POS = ["{t} about {p} that people liked", "well-liked {t} related to {p}", "good {t} on {p}"]
T_TOPIC_NEG = ["{t} about {p} that people criticized", "{t} related to {p} people disliked"]

SINGULAR = {"books": "book", "movies": "movie", "TV shows": "TV show", "video games": "video game",
            "board games": "board game", "podcasts": "podcast", "albums or songs": "album or song",
            "software tools": "software tool", "products": "product",
            "people (authors, directors, artists)": "person", "things": "thing"}


NON_LIST_Q = re.compile(r"^(how|why|when|where|is|are there|does|do|did|can|could|should|would|will|who is)\b|steps|requirements|explain", re.I)
STRICT = False  # --strict-names: names must contain a capital letter or digit (for noisy teachers)


def list_query(q):
    """Keep only requests whose answer is a list of items (drop 'how do I...', 'why did...')."""
    return not NON_LIST_Q.search(q.strip())


def name_like(t):
    return is_valid_item(t) and (not STRICT or bool(re.search(r"[A-Z0-9]", t))) and len(t.split()) <= 8


def find_spans(text, mentions):
    """Locate every occurrence of each mention; resolve overlaps (longer wins)."""
    cands = []
    for mi, m in enumerate(mentions):
        if not name_like(m["text"]):   # teacher junk like "books", "it", ",", "wafer"
            continue
        for h in find_all(text, m["text"]):
            cands.append((h.start(), h.end(), mi))
    cands.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
    taken, spans = [], []
    for s, e, mi in cands:
        if any(s < te and ts < e for ts, te in taken):
            continue
        taken.append((s, e))
        spans.append((s, e, mi))
    return sorted(spans)


def make_example(query, text, spans, mentions, relevant_set):
    return {"query": query, "text": text,
            "spans": [{"start": s, "end": e, "text": text[s:e],
                       "relevant": int(mi in relevant_set),
                       "polarity": POL[mentions[mi]["sentiment"]],
                       "type": mentions[mi].get("type", "other")} for s, e, mi in spans]}


def templated(mentions, rng, global_topics, global_attrs, k):
    out = []
    types = sorted({m["type"] for m in mentions})
    all_types = list(PLURAL)
    for _ in range(k):
        kind = rng.choice(["liked", "hated", "all", "attr", "attr", "topic", "topic", "rec", "rec_type", "absent",
                           "topic_pol", "absent"])
        t = rng.choice(types) if types else "book"
        tp = tword(t, rng)
        if kind == "liked":
            q = rng.choice(T_LIKED).format(t=tp)
            rel = {i for i, m in enumerate(mentions) if m["type"] == t and m["sentiment"] == "positive"}
        elif kind == "hated":
            q = rng.choice(T_HATED).format(t=tp)
            rel = {i for i, m in enumerate(mentions) if m["type"] == t and m["sentiment"] == "negative"}
        elif kind == "all":
            q = rng.choice(T_ALL).format(t=tp, t_s=singular(tp))
            rel = {i for i, m in enumerate(mentions) if m["type"] == t}
        elif kind == "attr":
            local = [a for m in mentions if m["type"] == t for a in m["attributes"]]
            a = rng.choice(local) if local and rng.random() < 0.7 else rng.choice(global_attrs)
            q = rng.choice(T_ATTR).format(t=tp, a=a)
            rel = {i for i, m in enumerate(mentions) if m["type"] == t and a in m["attributes"]}
        elif kind == "topic":
            local = [p for m in mentions for p in m["topics"]]
            p = rng.choice(local) if local and rng.random() < 0.7 else rng.choice(global_topics)
            tmpl = rng.choice(T_TOPIC)
            q = tmpl.format(t=tp, p=p)
            anytype = tmpl.startswith("anything")
            rel = {i for i, m in enumerate(mentions)
                   if (anytype or m["type"] == t) and p.lower() in [x.lower() for x in m["topics"]]}
        elif kind == "rec":
            q = rng.choice(T_REC)
            rel = {i for i, m in enumerate(mentions) if m["recommended"]}
        elif kind == "rec_type":
            q = rng.choice(T_REC_TYPE).format(t=tp)
            rel = {i for i, m in enumerate(mentions) if m["type"] == t and m["recommended"]}
        elif kind == "topic_pol":
            local = [(p, m["sentiment"]) for m in mentions if m["type"] == t for p in m["topics"]
                     if m["sentiment"] in ("positive", "negative")]
            if not local:
                continue
            p, sent = rng.choice(local)
            q = rng.choice(T_TOPIC_POS if sent == "positive" else T_TOPIC_NEG).format(t=tp, p=p)
            rel = {i for i, m in enumerate(mentions) if m["type"] == t and m["sentiment"] == sent
                   and p.lower() in [x.lower() for x in m["topics"]]}
        else:  # a type that is not in this chunk -> nothing relevant
            absent = [x for x in all_types if x not in types and x != "other"]
            if not absent:
                continue
            w = tword(rng.choice(absent), rng)
            q = rng.choice(T_ALL + T_LIKED + T_HATED + T_REC_TYPE).format(t=w, t_s=singular(w))
            rel = set()
        if not rel and kind != "absent" and rng.random() < 0.5:
            continue  # keep empty answers mostly from deliberate "wrong type" questions
        out.append((q, rel))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/labeled/labels.jsonl")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--heldout-domain", default="games")
    ap.add_argument("--templated-per-chunk", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--strict-names", action="store_true")
    args = ap.parse_args()
    global STRICT
    STRICT = args.strict_names
    rng = random.Random(args.seed)

    rows = [json.loads(l) for l in open(args.inp, encoding="utf-8")]
    global_topics = sorted({p for r in rows for m in r["mentions"] for p in m["topics"]}) or ["space"]
    global_attrs = sorted({a for r in rows for m in r["mentions"] for a in m["attributes"]}) or ["funny"]

    threads = sorted({r["thread_id"] for r in rows if r["domain"] != args.heldout_domain})
    rng.shuffle(threads)
    n = len(threads)
    split_of = {t: "train" for t in threads[: int(0.9 * n)]}
    split_of.update({t: "val" for t in threads[int(0.9 * n): int(0.95 * n)]})
    split_of.update({t: "test" for t in threads[int(0.95 * n):]})

    buckets = defaultdict(list)
    for r in rows:
        split = "test_heldout" if r["domain"] == args.heldout_domain else split_of[r["thread_id"]]
        spans = find_spans(r["text"], r["mentions"])
        mentions = r["mentions"]
        queries = [(q["query"], set(q["relevant"])) for q in r["queries"] if list_query(q["query"])]
        if mentions:
            k = args.templated_per_chunk + (2 if not queries else 0)  # more templated when no teacher queries
            queries += templated(mentions, rng, global_topics, global_attrs, k)
        elif rng.random() < 0.3:  # some mention-free chunks so the model learns to output nothing
            queries += templated([], rng, global_topics, global_attrs, 1)
        for q, rel in queries:
            ex = make_example(q, r["text"], spans, mentions, rel)
            ex.update({"chunk_id": r["chunk_id"], "thread_id": r["thread_id"], "domain": r["domain"]})
            buckets[split].append(ex)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    for split, exs in buckets.items():
        rng.shuffle(exs)
        with open(Path(args.outdir) / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for ex in exs:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        npos = sum(s["relevant"] for ex in exs for s in ex["spans"])
        print(f"{split:13s} {len(exs):7d} examples, {npos} relevant spans")


if __name__ == "__main__":
    main()
