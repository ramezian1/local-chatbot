#!/usr/bin/env python3
"""
Local Chatbot with Multi‑File Q&A (TF‑IDF)
-----------------------------------------
New features:
- Load one file:   `load <file>` (checks CWD, then script folder, then ./docs)
- Load a folder:   `load folder <path>` (indexes all .txt/.md/.log)
- Ask questions:   `ask <question>` (cosine similarity over TF‑IDF chunk vectors)
- Manage:          `list docs`, `clear docs`

Notes:
- Stays fully local. No dependencies. Index is in memory.
- Chunking is paragraph‑aware, ~600 chars per chunk.
"""
from __future__ import annotations
import re
import sys
import json
import time
import math
import random
from datetime import datetime, date
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "chat_logs"
DOCS_DIR = ROOT / "docs"  # default folder for docs
MEMORY_FILE = DATA_DIR / "facts.json"
TODO_FILE = DATA_DIR / "todos.json"

for p in (DATA_DIR, LOG_DIR, DOCS_DIR):
    p.mkdir(parents=True, exist_ok=True)

if not MEMORY_FILE.exists():
    MEMORY_FILE.write_text(json.dumps({}, indent=2))
if not TODO_FILE.exists():
    TODO_FILE.write_text(json.dumps([], indent=2))

# ---------------- Utilities: storage, logging ----------------

def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default

def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)

def log_line(path: Path, who: str, text: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {who}: {text}\n")

# ---------------- Q&A: tokenizer, chunking, vectors ----------
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
ALLOWED_SUFFIX = {".txt", ".md", ".log"}


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in WORD_RE.findall(text)]


def split_chunks(text: str, max_chars: int = 600) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    for p in paras:
        if len(p) <= max_chars:
            chunks.append(p)
        else:
            sent = re.split(r"(?<=[.!?])\s+", p)
            cur = ""
            for s in sent:
                if len(cur) + len(s) + 1 <= max_chars:
                    cur = (cur + " " + s).strip()
                else:
                    if cur:
                        chunks.append(cur)
                    cur = s
            if cur:
                chunks.append(cur)
    if not chunks and text.strip():
        chunks = [text.strip()[:max_chars]]
    return chunks

class TfidfIndex:
    def __init__(self):
        self.docs: List[Dict[str, object]] = []  # {path, n_chunks}
        self.chunks: List[Dict[str, object]] = []  # {doc_path, chunk_id, text, tokens, tf, vec, norm}
        self.idf: Dict[str, float] = {}

    # ----- Loading -----
    def add_file(self, path: Path) -> int:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = split_chunks(text)
        base = len(self.chunks)
        for i, c in enumerate(chunks):
            toks = tokenize(c)
            tf: Dict[str, float] = {}
            if toks:
                inv = 1.0 / len(toks)
                for t in toks:
                    tf[t] = tf.get(t, 0.0) + inv
            self.chunks.append({
                "doc_path": str(path),
                "chunk_id": i,
                "text": c,
                "tokens": toks,
                "tf": tf,
                "vec": {},
                "norm": 0.0,
            })
        self.docs.append({"path": str(path), "n_chunks": len(chunks)})
        return len(chunks)

    def add_folder(self, folder: Path) -> Tuple[int, int]:
        files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIX]
        files.sort()
        nfiles = 0
        nchunks = 0
        for f in files:
            nchunks += self.add_file(f)
            nfiles += 1
        return nfiles, nchunks

    # ----- Vectorization -----
    def rebuild(self) -> None:
        # compute IDF across chunks
        N = max(1, len(self.chunks))
        df: Dict[str, int] = {}
        for ch in self.chunks:
            seen = set(ch["tokens"]) if ch["tokens"] else set()
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self.idf = {t: math.log((1 + N) / (1 + df_t)) + 1.0 for t, df_t in df.items()}
        # build vectors and norms
        for ch in self.chunks:
            vec: Dict[str, float] = {}
            for t, tf in ch["tf"].items():
                idf = self.idf.get(t)
                if idf is None:
                    continue
                vec[t] = tf * idf
            norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
            ch["vec"] = vec
            ch["norm"] = norm

    # ----- Query -----
    def query(self, text: str, k: int = 3) -> List[Tuple[float, Dict[str, object]]]:
        q_tokens = tokenize(text)
        if not q_tokens or not self.chunks:
            return []
        # TF for query
        tf: Dict[str, float] = {}
        inv = 1.0 / len(q_tokens)
        for t in q_tokens:
            tf[t] = tf.get(t, 0.0) + inv
        # Build query vector using index IDF
        qvec: Dict[str, float] = {}
        for t, tfv in tf.items():
            idf = self.idf.get(t)
            if idf:
                qvec[t] = tfv * idf
        if not qvec:
            return []
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        # Cosine against all chunks
        scored: List[Tuple[float, Dict[str, object]]] = []
        for ch in self.chunks:
            vec = ch["vec"]
            if not vec:
                continue
            dot = 0.0
            # iterate smaller map
            (small, large) = (qvec, vec) if len(qvec) <= len(vec) else (vec, qvec)
            for t, v in small.items():
                if t in large:
                    dot += v * large[t]
            if dot == 0.0:
                continue
            score = dot / (qnorm * ch["norm"])
            scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]

# ---------------- Bot ----------------
class Bot:
    def __init__(self, name: str = "Bobo"):
        self.name = name
        self.facts: Dict[str, str] = _read_json(MEMORY_FILE, {})
        self.todos: List[Dict[str, object]] = _read_json(TODO_FILE, [])
        self.start_ts = time.time()
        self.jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "There are 10 kinds of people: those who understand binary and those who don't.",
            "I told my computer I needed a break, and it said 'No problem, I'll go to sleep.'",
        ]
        self.index = TfidfIndex()
        self.intents: List[Tuple[re.Pattern, Callable[[re.Match, str], str]]] = [
            # Q&A first so 'ask what is ...' routes here
            (re.compile(r"^load\s+folder\s+(?P<folder>.+)$", re.I), self._intent_load_folder),
            (re.compile(r"^load\s+(?P<file>.+)$", re.I), self._intent_load_file),
            (re.compile(r"^ask\s+(?P<q>.+)$", re.I), self._intent_ask),
            (re.compile(r"^(?:list|show)\s+docs$", re.I), self._intent_doc_list),
            (re.compile(r"^clear\s+docs$", re.I), self._intent_doc_clear),

            # Core features
	    (re.compile(r"^remember (?:that )?(?P<k>[\w\s-]{1,60})\s+(?:is|=)\s+(?P<v>.+)$", re.I), self._intent_remember),
            (re.compile(r"^(?:what is|what's)\s+(?P<k>[\w\s-]{1,60})\??$", re.I), self._intent_recall),
            (re.compile(r"^(?:add|todo)\s+(?P<item>.+)$", re.I), self._intent_todo_add),
            (re.compile(r"^(?:list|show)\s+(?:todos?|tasks?)$", re.I), self._intent_todo_list),
            (re.compile(r"^(?:done|complete)\s+(?P<idx>\d+)$", re.I), self._intent_todo_done),
            (re.compile(r"^clear\s+(?:todos?|tasks?)$", re.I), self._intent_todo_clear),

            # Small talk & utilities
            (re.compile(r"^(?:hi|hello|hey)(?:\\b.*)?$", re.I), self._intent_greet),
            (re.compile(r"^(?:thanks|thank you).*", re.I), self._intent_thanks),
            (re.compile(r"^(?:bye|exit|quit)$", re.I), self._intent_bye),
            (re.compile(r"^(?:time|what time is it)\??$", re.I), self._intent_time),
            (re.compile(r"^(?:date|what(?:'s| is) the date)\??$", re.I), self._intent_date),
            (re.compile(r"^echo\s+(.+)$", re.I), self._intent_echo),
            (re.compile(r"^joke$", re.I), self._intent_joke),
            (re.compile(r"^help$", re.I), self._intent_help),
        ]

    # ----- Memory / todos -----
    def _intent_remember(self, m: re.Match, _: str) -> str:
        k = normalize_key(m.group("k"))
        v = m.group("v").strip()
        self.facts[k] = v
        _write_json(MEMORY_FILE, self.facts)
        return f"Got it. I’ll remember {k!r} = {v!r}."

    def _intent_recall(self, m: re.Match, _: str) -> str:
        k = normalize_key(m.group("k"))
        if k in self.facts:
            return f"You told me {k!r} = {self.facts[k]!r}."
        return f"I don't have anything saved for {k!r} yet."

    def _intent_todo_add(self, m: re.Match, _: str) -> str:
        item = m.group("item").strip()
        self.todos.append({"text": item, "done": False, "ts": time.time()})
        _write_json(TODO_FILE, self.todos)
        return f"Added to‑do #{len(self.todos)}: {item}"

    def _intent_todo_list(self, *_args) -> str:
        if not self.todos:
            return "No to‑dos yet. Add one with: add <task>"
        lines = ["To‑dos:"]
        for i, t in enumerate(self.todos, 1):
            mark = "[x]" if t.get("done") else "[ ]"
            lines.append(f"  {i:>2}. {mark} {t['text']}")
        return "\n".join(lines)

    def _intent_todo_done(self, m: re.Match, _: str) -> str:
        idx = int(m.group("idx")) - 1
        if 0 <= idx < len(self.todos):
            self.todos[idx]["done"] = True
            _write_json(TODO_FILE, self.todos)
            return f"Marked to‑do #{idx+1} as done."
        return "Invalid index. Try: done 1"

    def _intent_todo_clear(self, *_args) -> str:
        self.todos.clear()
        _write_json(TODO_FILE, self.todos)
        return "Cleared all to‑dos."

    # ----- Q&A intents -----
    def _resolve_for_load(self, raw: str) -> Optional[Path]:
        raw = raw.strip().strip('"')
        candidates: List[Path] = []
        p = Path(raw).expanduser()
        candidates.append(p)
        candidates.append((ROOT / raw))
        candidates.append((DOCS_DIR / raw))
        for c in candidates:
            if c.exists():
                return c.resolve()
        return None

    def _intent_load_file(self, m: re.Match, _: str) -> str:
        target = self._resolve_for_load(m.group("file"))
        if not target or not target.is_file() or target.suffix.lower() not in ALLOWED_SUFFIX:
            return "File not found or unsupported. Use .txt/.md/.log"
        n = self.index.add_file(target)
        self.index.rebuild()
        return f"Loaded {target.name} with {n} chunks. Ask with: ask <question>"

    def _intent_load_folder(self, m: re.Match, _: str) -> str:
        folder = self._resolve_for_load(m.group("folder"))
        if not folder or not folder.is_dir():
            return "Folder not found."
        nfiles, nchunks = self.index.add_folder(folder)
        if nfiles == 0:
            return "No .txt/.md/.log files found in that folder."
        self.index.rebuild()
        return f"Indexed {nfiles} files, {nchunks} chunks. Ask with: ask <question>"

    def _intent_doc_list(self, *_args) -> str:
        if not self.index.docs:
            return "No docs loaded. Use: load <file> or load folder <path>"
        lines = ["Docs:"]
        for i, d in enumerate(self.index.docs, 1):
            lines.append(f"  {i:>2}. {Path(d['path']).name}  ({d['n_chunks']} chunks)")
        return "\n".join(lines)

    def _intent_doc_clear(self, *_args) -> str:
        self.index = TfidfIndex()
        return "Cleared all docs."

    def _intent_ask(self, m: re.Match, _: str) -> str:
        q = m.group("q").strip()
        results = self.index.query(q, k=3)
        if not results:
            return "No good matches found. Try rephrasing or loading more files."
        lines = [f"Top matches for: {q}"]
        for score, ch in results:
            name = Path(ch["doc_path"]).name
            snippet = ch["text"].strip().replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:237] + "..."
            lines.append(f"- [{name}] ({score:.3f}) {snippet}")
        return "\n".join(lines)

    # ----- Small talk & utils -----
    def _intent_greet(self, *_args) -> str:
        return random_choice(["Hey! What's up?", "Hello! Hit 'help' to see commands."])

    def _intent_thanks(self, *_args) -> str:
        return random_choice(["Anytime!", "You got it.", "Happy to help."])

    def _intent_bye(self, *_args) -> str:
        return "Bye! (psst: your data is saved in ./data)"

    def _intent_time(self, *_args) -> str:
        return datetime.now().strftime("%I:%M %p").lstrip('0')

    def _intent_date(self, *_args) -> str:
        return date.today().strftime("%A, %B %d, %Y")

    def _intent_echo(self, m: re.Match, _: str) -> str:
        return m.group(1)

    def _intent_joke(self, *_args) -> str:
        return random.choice(self.jokes)

    def _intent_help(self, *_args) -> str:
        return (
            "Commands:\n"
            "  load <file>                  – add a .txt/.md/.log (./, script dir, or ./docs)\n"
            "  load folder <path>           – add all .txt/.md/.log in a folder\n"
            "  list docs                    – show loaded files\n"
            "  clear docs                   – remove all indexed files\n"
            "  ask <question>               – search with TF‑IDF + cosine\n"
            "\n"
            "  remember X is Y              – save a fact\n"
            "  what's X?                    – recall a fact\n"
            "  add <task>                   – add a to‑do\n"
            "  list todos                   – list to‑dos\n"
            "  done <n>                     – mark to‑do done\n"
            "  clear todos                  – remove all\n"
            "  time | date                  – local time/date\n"
            "  echo <text>                  – repeat back\n"
            "  joke                         – random joke\n"
            "  help                         – this menu\n"
            "  bye/exit/quit                – save & exit"
        )

    def respond(self, message: str) -> str:
        text = message.strip()
        if not text:
            return "Say something 🙂"
        if text.lower() in {"bye", "exit", "quit"}:
            return self._intent_bye()
        for pattern, handler in self.intents:
            m = pattern.match(text)
            if m:
                try:
                    return handler(m, text)
                except Exception as e:
                    return f"Oops, that blew up: {e}"
        return "Not sure. Try 'help'."

# ---------------- Entry point ----------------

def random_choice(items: List[str]) -> str:
    random.seed()
    return random.choice(items)

def normalize_key(k: str) -> str:
    return re.sub(r"\s+", " ", k.strip().lower())

def main(argv: List[str]):
    bot = Bot(name="Bobo")
    print(f"{bot.name} ready. Type 'help' to begin. Ctrl+C to quit.")
    today_log = LOG_DIR / f"{date.today().isoformat()}.txt"
    try:
        while True:
            try:
                user = input("> ").strip()
            except EOFError:
                print()
                break
            log_line(today_log, "you", user)
            reply = bot.respond(user)
            print(reply)
            log_line(today_log, bot.name.lower(), reply)
            if reply.startswith("Bye!"):
                break
    except KeyboardInterrupt:
        print("\nInterrupted. Bye!")

if __name__ == "__main__":
    sys.exit(main(sys.argv))
