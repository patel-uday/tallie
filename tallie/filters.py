"""Rules for what counts as a *named item*. Used both when building training spans
and when ranking results, so junk like "books", "it", "The" or "," never becomes an item."""
import re

GENERIC = set("""
a an the it its this that these those one ones they them he she him her his i me my we us you your
book books novel novels story stories series trilogy volume author authors writer writers
movie movies film films show shows tv episode episodes season seasons documentary documentaries
game games video videos podcast podcasts episode album albums song songs music band bands track tracks
software tool tools app apps program programs language languages library libraries framework editor
fiction nonfiction non-fiction sci-fi scifi science fantasy horror comedy drama romance thriller history
philosophy literature poetry biography biographies memoir textbook textbooks classic classics
anime manga comic comics cartoon cartoons article articles essay essays paper papers blog blogs
something anything everything nothing thing things stuff lot lots part
""".split())

_ALNUM = re.compile(r"[A-Za-z0-9]")


def is_valid_item(text: str) -> bool:
    t = text.strip().strip("\"'“”‘’*_()[]:;,.!?").strip()
    if t in {"C", "R", "D", "J"}:   # single-letter programming languages
        return True
    if len(t) < 2 or not _ALNUM.search(t):
        return False
    low = t.lower()
    if low in GENERIC:
        return False
    words = re.findall(r"[a-z0-9]+", low)
    if words and all(w in GENERIC for w in words):  # e.g. "the books", "sci fi movies"
        return False
    return True


def find_all(text: str, needle: str):
    """Whole-word occurrences. Case-insensitive fallback only for longer names
    (so the novel "IT" never matches every "it")."""
    pat = r"(?<![\w])" + re.escape(needle) + r"(?![\w])"
    hits = list(re.finditer(pat, text))
    if not hits and len(needle) >= 4:
        hits = list(re.finditer(pat, text, flags=re.I))
    return hits


def clean_span(text: str, s: int, e: int):
    """Fix model spans that start/end mid-word ('loat', 'L' from 'Luddites') by snapping to
    whole words, and drop spans that sit inside a URL. Returns (s, e) or None."""
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    if s >= e:
        return None
    while s > 0 and text[s - 1].isalnum():
        s -= 1
    while e < len(text) and text[e].isalnum():
        e += 1
    a = text.rfind(" ", 0, s) + 1
    b = text.find(" ", e)
    run = text[a: b if b != -1 else len(text)]
    if "://" in run or "www." in run or run.count("/") >= 2:
        return None
    return s, e
