"""
Download Tallie from Hugging Face and rank what a few comments talk about.

  pip install -r requirements.txt
  python examples/example.py
"""
import sys
from huggingface_hub import snapshot_download

path = snapshot_download("pateluday/tallie")   # ~2.7 GB, cached after the first run
sys.path.insert(0, path)
from tallie import Tallie

t = Tallie.load(path)

comments = [
    "Planet Money and EconTalk are the two podcasts I never skip.",
    "I tried 1Q84 twice and gave up, it's so slow.",
    "Honestly 99% Invisible is the best podcast out there.",
    "EconTalk changed how I think about economics.",
]
for item in t.extract(comments):
    print(f"{item['text']:20s} type={item['type']:10s} opinion={item['polarity']}")

t.index(comments)
for q in ["podcasts people recommend", "books people disliked"]:
    print("\n" + q)
    for r in t.ask(q)["results"]:
        print(f"  {r['item']:20s} {r['mentions']} mention(s)  {r['polarity']}")
