"""
Teacher labeling: a big LLM reads each comment ONCE and writes rich labels for every
mention (type, sentiment, attributes, topics) plus a few free-form queries.
The small Tallie model is later trained to reproduce these judgements without generating text.

Backends:
  --backend vllm       run an open model on the Colab GPU (free). Default: Qwen3-14B on A100, Qwen3-8B otherwise.
  --backend anthropic  use the Claude API (needs ANTHROPIC_API_KEY), higher quality, costs a few dollars.

Output: data/labeled/labels.jsonl, one chunk per line:
  {"chunk_id", "thread_id", "domain", "text", "mentions": [...], "queries": [...]}
"""
import argparse, json, random, re, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TYPES = ["book", "movie", "tv_show", "video_game", "board_game", "podcast", "music",
         "software", "product", "person", "other"]
SENTIMENTS = ["positive", "negative", "neutral", "mixed"]

SYSTEM = f"""You annotate internet comments for an information-extraction dataset.
Return ONLY a JSON object, no prose, with this schema:
{{
  "mentions": [
    {{"text": "<exact substring copied character-for-character from the comment>",
      "type": one of {TYPES},
      "sentiment": one of {SENTIMENTS}  (the COMMENTER'S opinion of this item),
      "recommended": true/false  (does the commenter suggest others try it?),
      "attributes": [short adjectives people would search by, e.g. "funny","scary","inspiring","boring","overrated","underrated","beginner-friendly","sad","mind-blowing","educational","relaxing","hard"],
      "topics": [1-4 short subject tags, e.g. "King Arthur","space","startups","World War II","programming"]}}
  ],
  "queries": [
    {{"query": "<a natural request someone could ask about a big collection of such comments>",
      "relevant": [indices into mentions that satisfy the request]}}
  ]
}}
Rules:
- Mention = a specific named thing: a title of a book/movie/show/game/podcast/album/song, a named software/product, or a named person (author, director, musician).
- Copy "text" EXACTLY as written in the comment (keep its casing/typos). Do not invent mentions. Skip generic words ("a book", "that movie").
- Use world knowledge for topics/attributes (e.g. "The Once and Future King" -> topics ["King Arthur"]), and read sarcasm correctly.
- Write 2-3 queries of DIFFERENT styles (opinion, attribute, topic, audience, comparison). At least one query should need reasoning beyond keywords. A query may have zero relevant mentions.
- If there are no mentions, return {{"mentions": [], "queries": []}}."""


def chunk_text(text, max_chars=1500):
    if len(text) <= max_chars:
        return [text]
    chunks, cur = [], ""
    for para in re.split(r"\n\s*\n", text):
        if len(cur) + len(para) + 2 > max_chars and cur:
            chunks.append(cur.strip())
            cur = ""
        cur += para + "\n\n"
        while len(cur) > max_chars:  # very long paragraph: hard split on sentence boundary
            cut = cur.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > 200 else max_chars
            chunks.append(cur[:cut].strip())
            cur = cur[cut:]
    if cur.strip():
        chunks.append(cur.strip())
    return chunks


def parse_json(raw):
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.S)
    a, b = raw.find("{"), raw.rfind("}")
    if a < 0 or b < 0:
        return None
    try:
        obj = json.loads(raw[a:b + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    ments = []
    for m in obj.get("mentions") or []:
        if not isinstance(m, dict) or not isinstance(m.get("text"), str) or not m["text"].strip():
            continue
        ments.append({
            "text": m["text"].strip(),
            "type": m.get("type") if m.get("type") in TYPES else "other",
            "sentiment": m.get("sentiment") if m.get("sentiment") in SENTIMENTS else "neutral",
            "recommended": bool(m.get("recommended", False)),
            "attributes": [str(x).lower().strip() for x in (m.get("attributes") or []) if str(x).strip()][:8],
            "topics": [str(x).strip() for x in (m.get("topics") or []) if str(x).strip()][:4],
        })
    qs = []
    for q in obj.get("queries") or []:
        if isinstance(q, dict) and isinstance(q.get("query"), str):
            rel = [i for i in (q.get("relevant") or []) if isinstance(i, int) and 0 <= i < len(ments)]
            qs.append({"query": q["query"].strip(), "relevant": rel})
    return {"mentions": ments, "queries": qs}


def load_chunks(path, limit):
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            for j, t in enumerate(chunk_text(c["text"])):
                chunks.append({"chunk_id": f"{c['comment_id']}_{j}", "thread_id": c["thread_id"],
                               "thread_title": c["thread_title"], "domain": c["domain"], "text": t})
    return chunks[:limit] if limit else chunks


def user_msg(c):
    return f"Thread title (context only): {c['thread_title']}\n\nComment:\n{c['text']}"


def run_vllm(chunks, llm):
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1200)
    convs = [[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_msg(c)}]
             for c in chunks]
    outs = llm.chat(convs, sp, chat_template_kwargs={"enable_thinking": False}, use_tqdm=True)
    return [o.outputs[0].text for o in outs]


def run_anthropic(chunks, model, workers=8):
    import anthropic
    client = anthropic.Anthropic()

    def one(c):
        for _ in range(3):
            try:
                r = client.messages.create(model=model, max_tokens=1200, system=SYSTEM,
                                           messages=[{"role": "user", "content": user_msg(c)}])
                return r.content[0].text
            except Exception as e:  # rate limits etc.
                err = e
        print("failed:", err, file=sys.stderr)
        return ""
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, chunks))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/raw/hn_comments.jsonl")
    ap.add_argument("--out", default="data/labeled/labels.jsonl")
    ap.add_argument("--backend", choices=["vllm", "anthropic"], default="vllm")
    ap.add_argument("--model", default=None)
    ap.add_argument("--limit", type=int, default=0, help="label a random N chunks (0 = all)")
    ap.add_argument("--save-every", type=int, default=1000)
    args = ap.parse_args()

    chunks = load_chunks(args.inp, 0)
    random.Random(0).shuffle(chunks)          # mix domains before applying --limit
    if args.limit:
        chunks = chunks[:args.limit]

    # resume: skip chunks already labeled (Colab sessions disconnect)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(l)["chunk_id"] for l in open(out, encoding="utf-8")}
    done_path = out.with_suffix(".attempted")
    if done_path.exists():
        done |= set(done_path.read_text().split())
    todo = [c for c in chunks if c["chunk_id"] not in done]
    print(f"{len(chunks)} chunks selected, {len(chunks) - len(todo)} already done, "
          f"{len(todo)} to label with {args.backend}")
    if not todo:
        return

    if args.backend == "vllm":
        import torch
        from vllm import LLM
        model = args.model
        mem = torch.cuda.get_device_properties(0).total_memory / 2**30
        if model is None:
            model = "Qwen/Qwen3-14B" if mem > 38 else "Qwen/Qwen3-8B-AWQ" if mem > 14 else "Qwen/Qwen3-4B-AWQ"
        bf16 = torch.cuda.get_device_capability()[0] >= 8
        print("teacher:", model, "| dtype:", "bfloat16" if bf16 else "float16")
        llm = LLM(model=model, max_model_len=4096, gpu_memory_utilization=0.90,
                  dtype="auto" if bf16 else "half", enforce_eager=not bf16)
        label = lambda batch: run_vllm(batch, llm)
    else:
        name = args.model or "claude-haiku-5-5"
        label = lambda batch: run_anthropic(batch, name)

    ok = bad = 0
    for b0 in range(0, len(todo), args.save_every):
        batch = todo[b0:b0 + args.save_every]
        raws = label(batch)
        with open(out, "a", encoding="utf-8") as f, open(done_path, "a") as fd:
            for c, raw in zip(batch, raws):
                fd.write(c["chunk_id"] + "\n")
                lab = parse_json(raw)
                if lab is None:
                    bad += 1
                    continue
                ok += 1
                f.write(json.dumps({**c, **lab}, ensure_ascii=False) + "\n")
        print(f"[{b0 + len(batch)}/{len(todo)}] saved. ok={ok} unparseable={bad}", flush=True)
    print(f"labeled {ok} chunks, {bad} unparseable -> {out}")


if __name__ == "__main__":
    main()
