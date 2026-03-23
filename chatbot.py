#!/usr/bin/env python3
"""
Local Chatbot with Multi-File Q&A (TF-IDF) + SQLite Memory + Colored Output
---------------------------------------------------------------------------
Commands:
  load <file>              - add a .txt/.md/.log (./, script dir, or ./docs)
  load folder <path>       - add all .txt/.md/.log in a folder
  list docs                - show loaded files
  clear docs               - remove all indexed files
  ask <question>           - search with TF-IDF + cosine (min score: 0.05)

  remember X is Y          - save a fact
  what's X?                - recall a fact
  /remember X: Y           - slash alias to remember
  /recall X                - slash alias to recall
  /memkeys                 - list saved memory keys
  /memsearch <text>        - search keys/values in memory
  /forget <key>            - delete a saved memory

  add <task>               - add a to-do
  list todos               - list to-dos
  done <n>                 - mark to-do done
  clear todos              - remove all

  time | date              - local time/date
  uptime                   - session uptime
  echo <text>              - repeat back
  joke                     - random joke
  help                     - this menu
  bye/exit/quit            - save & exit

Tips:
  Run with more results:   python chatbot.py --top-k 5
  Disable colors:          python chatbot.py --no-color
  Set min match score:     python chatbot.py --min-score 0.1

Notes:
- Fully local. No external deps.
- Index is in memory. Chunking is paragraph-aware (~600 chars).
- Memory is persisted via SQLite (FTS5 for fast /memsearch).
- Stopwords filtered from TF-IDF vectors for better match quality.
"""
from __future__ import annotations

import json
import math
import os
import random
import re
import sqlite3
import sys
import time
from datetime import date, datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "chat_logs"
DOCS_DIR = ROOT / "docs"
MEMORY_DB = DATA_DIR / "memory.db"
TODO_FILE = DATA_DIR / "todos.json"

for _p in (DATA_DIR, LOG_DIR, DOCS_DIR):
    _p.mkdir(parents=True, exist_ok=True)

if not TODO_FILE.exists():
    TODO_FILE.write_text(json.dumps([], indent=2), encoding="utf-8")

# ---------------- Utilities: storage, logging ----------------
def _read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def log_line(path: Path, who: str, text: str) -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"[{ts}] {who}: {text}\n")


def normalize_key(k: str) -> str:
    return re.sub(r"\s+", " ", k.strip().lower())


# ---------------- Memory 2.0: SQLite + FTS5 ----------------
SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL,
  updated_at REAL NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
  key, value, content='memories', content_rowid='rowid'
);
CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
  INSERT INTO memories_fts(rowid, key, value) VALUES (new.rowid, new.key, new.value);
END;
CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, key, value) VALUES('delete', old.rowid, old.key, old.value);
END;
CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
  INSERT INTO memories_fts(memories_fts, rowid, key, value) VALUES('delete', old.rowid, old.key, old.value);
  INSERT INTO memories_fts(rowid, key, value) VALUES (new.rowid, new.key, new.value);
END;
"""


class MemoryStore:
    def __init__(self, db_path: Path = MEMORY_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA foreign_keys=ON;")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def remember(self, key: str, value: str) -> None:
        key = key.strip()
        value = value.strip()
        now = time.time()
        self.conn.execute(
            "INSERT INTO memories(key, value, updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, value, now),
        )
        self.conn.commit()

    def recall(self, key: str) -> Optional[str]:
        cur = self.conn.execute("SELECT value FROM memories WHERE key=?", (key.strip(),))
        row = cur.fetchone()
        return row[0] if row else None

    def search(self, query: str, limit: int = 20) -> List[Tuple[str, str]]:
        query = query.strip()
        if not query:
            return []
        try:
            cur = self.conn.execute(
                "SELECT key, value FROM memories_fts WHERE memories_fts MATCH ? LIMIT ?",
                (query, limit),
            )
            rows = cur.fetchall()
            if rows:
                return rows
        except sqlite3.OperationalError:
            pass
        # Fallback LIKE search
        cur = self.conn.execute(
            "SELECT key, value FROM memories WHERE key LIKE ? OR value LIKE ? LIMIT ?",
            (f"%{query}%", f"%{query}%", limit),
        )
        return cur.fetchall()

    def forget(self, key: str) -> bool:
        cur = self.conn.execute("DELETE FROM memories WHERE key=?", (key.strip(),))
        self.conn.commit()
        return cur.rowcount > 0

    def keys(self, limit: int = 200) -> List[str]:
        cur = self.conn.execute(
            "SELECT key FROM memories ORDER BY updated_at DESC LIMIT ?", (limit,)
        )
        return [r[0] for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()


# ---------------- Q&A: tokenizer, chunking, vectors ----------------
WORD_RE = re.compile(r"[A-Za-z0-9_]+")
ALLOWED_SUFFIX = {".txt", ".md", ".log"}

# Common English stopwords to filter from TF-IDF vectors
STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "was", "are", "were", "be", "been",
    "has", "have", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "what", "which",
    "who", "how", "when", "where", "not", "no", "so", "if", "as", "up",
    "out", "about", "into", "than", "then", "s", "t",
})


def tokenize(text: str) -> List[str]:
    """Lowercase tokenize, filtering stopwords for better TF-IDF quality."""
    return [
        w.lower() for w in WORD_RE.findall(text)
        if w.lower() not in STOPWORDS
    ]


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
        self._dirty: bool = False  # track if rebuild needed

    # ----- Loading -----
    def add_file(self, path: Path) -> int:
        text = path.read_text(encoding="utf-8", errors="ignore")
        chunks = split_chunks(text)
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
        self._dirty = True
        return len(chunks)

    def add_folder(self, folder: Path) -> Tuple[int, int]:
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIX
        )
        nfiles = nchunks = 0
        for f in files:
            nchunks += self.add_file(f)
            nfiles += 1
        return nfiles, nchunks

    # ----- Vectorization -----
    def rebuild(self) -> None:
        if not self._dirty:
            return
        N = max(1, len(self.chunks))
        df: Dict[str, int] = {}
        for ch in self.chunks:
            seen = set(ch["tokens"]) if ch["tokens"] else set()
            for t in seen:
                df[t] = df.get(t, 0) + 1
        self.idf = {
            t: math.log((1 + N) / (1 + df_t)) + 1.0
            for t, df_t in df.items()
        }
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
        self._dirty = False

    # ----- Query -----
    def query(
        self, text: str, k: int = 3, min_score: float = 0.05
    ) -> List[Tuple[float, Dict[str, object]]]:
        if self._dirty:
            self.rebuild()
        q_tokens = tokenize(text)
        if not q_tokens or not self.chunks:
            return []
        tf: Dict[str, float] = {}
        inv = 1.0 / len(q_tokens)
        for t in q_tokens:
            tf[t] = tf.get(t, 0.0) + inv
        qvec: Dict[str, float] = {}
        for t, tfv in tf.items():
            idf = self.idf.get(t)
            if idf:
                qvec[t] = tfv * idf
        if not qvec:
            return []
        qnorm = math.sqrt(sum(v * v for v in qvec.values())) or 1.0
        scored: List[Tuple[float, Dict[str, object]]] = []
        for ch in self.chunks:
            vec = ch["vec"]
            if not vec:
                continue
            dot = 0.0
            small, large = (qvec, vec) if len(qvec) <= len(vec) else (vec, qvec)
            for t, v in small.items():
                if t in large:
                    dot += v * large[t]
            if dot == 0.0:
                continue
            score = dot / (qnorm * ch["norm"])
            if score >= min_score:
                scored.append((score, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:k]


# ---------------- Bot ----------------
class Bot:
    def __init__(
        self,
        name: str = "Bobo",
        top_k: int = 3,
        min_score: float = 0.05,
        use_color: bool = True,
    ):
        self.name = name
        self.top_k = max(1, int(top_k))
        self.min_score = float(min_score)
        self.use_color = bool(use_color)
        self.mem = MemoryStore()
        self.start_ts = time.time()
        self.jokes = [
            "Why do programmers prefer dark mode? Because light attracts bugs.",
            "There are 10 kinds of people: those who understand binary and those who don't.",
            "I told my computer I needed a break, and it said 'No problem, I'll go to sleep.'",
        ]
        self.index = TfidfIndex()
        random.seed()
        self.intents: List[Tuple[re.Pattern, Callable[[re.Match, str], str]]] = [
            # Q&A
            (re.compile(r"^load\s+folder\s+(?P<folder>.+)$", re.I), self._intent_load_folder),
            (re.compile(r"^load\s+(?P<file>.+)$", re.I), self._intent_load_file),
            (re.compile(r"^ask\s+(?P<q>.+)$", re.I), self._intent_ask),
            (re.compile(r"^(?:list|show)\s+docs$", re.I), self._intent_doc_list),
            (re.compile(r"^clear\s+docs$", re.I), self._intent_doc_clear),
            # Memory
            (re.compile(r"^remember (?:that )?(?P<k>[\w\s-]{1,60})\s*(?:is|=|:)\s*(?P<v>.+)$", re.I), self._intent_remember),
            (re.compile(r"^(?:what is|what's)\s+(?P<k>[\w\s-]{1,60})\??$", re.I), self._intent_recall),
            (re.compile(r"^/remember\s+(?P<k>[\w\s-]{1,60})\s*(?:=|:)\s*(?P<v>.+)$", re.I), self._intent_remember),
            (re.compile(r"^/recall\s+(?P<k>[\w\s-]{1,60})$", re.I), self._intent_recall),
            (re.compile(r"^/memkeys$", re.I), self._intent_memkeys),
            (re.compile(r"^/memsearch\s+(?P<q>.+)$", re.I), self._intent_memsearch),
            (re.compile(r"^/forget\s+(?P<k>[\w\s-]{1,60})$", re.I), self._intent_forget),
            # To-dos
            (re.compile(r"^(?:add|todo)\s+(?P<item>.+)$", re.I), self._intent_todo_add),
            (re.compile(r"^(?:list|show)\s+(?:todos?|tasks?)$", re.I), self._intent_todo_list),
            (re.compile(r"^(?:done|complete)\s+(?P<idx>\d+)$", re.I), self._intent_todo_done),
            (re.compile(r"^clear\s+(?:todos?|tasks?)$", re.I), self._intent_todo_clear),
            # Utilities
            (re.compile(r"^(?:hi|hello|hey)\b.*$", re.I), self._intent_greet),
            (re.compile(r"^(?:thanks|thank you).*$", re.I), self._intent_thanks),
            (re.compile(r"^(?:bye|exit|quit)$", re.I), self._intent_bye),
            (re.compile(r"^(?:time|what time is it)\??$", re.I), self._intent_time),
            (re.compile(r"^(?:date|what(?:'s| is) the date)\??$", re.I), self._intent_date),
            (re.compile(r"^uptime$", re.I), self._intent_uptime),
            (re.compile(r"^echo\s+(.+)$", re.I), self._intent_echo),
            (re.compile(r"^joke$", re.I), self._intent_joke),
            (re.compile(r"^help$", re.I), self._intent_help),
        ]

    # ----- Color helpers -----
    def _ansi(self, code: str) -> str:
        return code if self.use_color else ""

    def _reset(self) -> str:
        return self._ansi("\x1b[0m")

    def _bold(self, s: str) -> str:
        return self._ansi("\x1b[1m") + s + self._reset()

    def _dim(self, s: str) -> str:
        return self._ansi("\x1b[2m") + s + self._reset()

    def _fg(self, code: int, s: str) -> str:
        return self._ansi(f"\x1b[{code}m") + s + self._reset()

    def _score_style(self, score: float, rank: int) -> Callable[[str], str]:
        if rank == 0:
            return lambda s: self._bold(self._fg(32, s))
        if score >= 0.45:
            return lambda s: self._fg(32, s)
        if score >= 0.25:
            return lambda s: self._fg(33, s)
        return lambda s: self._fg(90, s)

    # ----- Memory intents -----
    def _intent_remember(self, m: re.Match, _: str) -> str:
        k = normalize_key(m.group("k"))
        v = m.group("v").strip()
        self.mem.remember(k, v)
        return f"Got it. I'll remember {k!r} = {v!r}."

    def _intent_recall(self, m: re.Match, _: str) -> str:
        k = normalize_key(m.group("k"))
        v = self.mem.recall(k)
        return (
            f"You told me {k!r} = {v!r}." if v is not None
            else f"I don't have anything saved for {k!r} yet."
        )

    def _intent_memkeys(self, *_args) -> str:
        keys = self.mem.keys()
        if not keys:
            return "🧠 Memory is empty."
        return "🧠 Keys (newest first):\n" + "\n".join(f"- {k}" for k in keys)

    def _intent_memsearch(self, m: re.Match, _: str) -> str:
        q = m.group("q").strip()
        if not q:
            return "Usage: /memsearch <text>"
        hits = self.mem.search(q, limit=20)
        if not hits:
            return "🔍 No matching memories."
        lines = [f"🔎 Matches ({len(hits)}):"]
        for k, v in hits[:20]:
            sv = str(v).replace("\n", " ")
            if len(sv) > 140:
                sv = sv[:137] + "..."
            lines.append(f"- **{k}** = {sv}")
        if len(hits) > 20:
            lines.append(f"...and {len(hits)-20} more")
        return "\n".join(lines)

    def _intent_forget(self, m: re.Match, _: str) -> str:
        k = normalize_key(m.group("k"))
        ok = self.mem.forget(k)
        return "🗑️  Forgotten." if ok else f"❓ Nothing saved for {k!r}."

    # ----- To-do intents -----
    def _intent_todo_add(self, m: re.Match, _: str) -> str:
        item = m.group("item").strip()
        todos = _read_json(TODO_FILE, [])
        todos.append({"text": item, "done": False, "ts": time.time()})
        _write_json(TODO_FILE, todos)
        return f"Added to-do #{len(todos)}: {item}"

    def _intent_todo_list(self, *_args) -> str:
        todos = _read_json(TODO_FILE, [])
        if not todos:
            return "No to-dos yet. Add one with: add <task>"
        lines = ["To-dos:"]
        for i, t in enumerate(todos, 1):
            mark = "[x]" if t.get("done") else "[ ]"
            lines.append(f"  {i:>2}. {mark} {t['text']}")
        return "\n".join(lines)

    def _intent_todo_done(self, m: re.Match, _: str) -> str:
        idx = int(m.group("idx")) - 1
        todos = _read_json(TODO_FILE, [])
        if 0 <= idx < len(todos):
            todos[idx]["done"] = True
            _write_json(TODO_FILE, todos)
            return f"Marked to-do #{idx+1} as done."
        return "Invalid index. Try: done 1"

    def _intent_todo_clear(self, *_args) -> str:
        _write_json(TODO_FILE, [])
        return "Cleared all to-dos."

    # ----- Q&A intents -----
    def _resolve_for_load(self, raw: str) -> Optional[Path]:
        raw = raw.strip().strip('"')
        candidates = [
            Path(raw).expanduser(),
            ROOT / raw,
            DOCS_DIR / raw,
        ]
        for c in candidates:
            if c.exists():
                return c.resolve()
        return None

    def _intent_load_file(self, m: re.Match, _: str) -> str:
        target = self._resolve_for_load(m.group("file"))
        if not target or not target.is_file() or target.suffix.lower() not in ALLOWED_SUFFIX:
            return "File not found or unsupported. Use .txt/.md/.log"
        n = self.index.add_file(target)
        return f"Loaded {target.name} with {n} chunks. Ask with: ask <question>"

    def _intent_load_folder(self, m: re.Match, _: str) -> str:
        folder = self._resolve_for_load(m.group("folder"))
        if not folder or not folder.is_dir():
            return "Folder not found."
        nfiles, nchunks = self.index.add_folder(folder)
        if nfiles == 0:
            return "No .txt/.md/.log files found in that folder."
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
        results = self.index.query(q, k=self.top_k, min_score=self.min_score)
        if not results:
            return f"No good matches (min score: {self.min_score}). Try rephrasing or loading more files."
        lines = [f"Top matches for: {q}"]
        for idx, (score, ch) in enumerate(results):
            sty = self._score_style(score, idx)
            name = Path(ch["doc_path"]).name
            snippet = ch["text"].strip().replace("\n", " ")
            if len(snippet) > 240:
                snippet = snippet[:237] + "..."
            head = sty(f"- [{name}] ({score:.3f})")
            if score < 0.25:
                snippet = self._dim(snippet)
            lines.append(f"{head} {snippet}")
        return "\n".join(lines)

    # ----- Small talk & utilities -----
    def _intent_greet(self, *_args) -> str:
        return random.choice(["Hey! What's up?", "Hello! Hit 'help' to see commands."])

    def _intent_thanks(self, *_args) -> str:
        return random.choice(["Anytime!", "You got it.", "Happy to help."])

    def _intent_bye(self, *_args) -> str:
        self.mem.close()
        return "Bye! (psst: your data is saved in ./data)"

    def _intent_time(self, *_args) -> str:
        return datetime.now().strftime("%I:%M %p").lstrip('0')

    def _intent_date(self, *_args) -> str:
        return date.today().strftime("%A, %B %d, %Y")

    def _intent_uptime(self, *_args) -> str:
        elapsed = time.time() - self.start_ts
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"Uptime: {h}h {m}m {s}s"
        if m > 0:
            return f"Uptime: {m}m {s}s"
        return f"Uptime: {s}s"

    def _intent_echo(self, m: re.Match, _: str) -> str:
        return m.group(1)

    def _intent_joke(self, *_args) -> str:
        return random.choice(self.jokes)

    def _intent_help(self, *_args) -> str:
        return (
            "Commands:\n"
            "  load <file>              - add a .txt/.md/.log (./, script dir, or ./docs)\n"
            "  load folder <path>       - add all .txt/.md/.log in a folder\n"
            "  list docs                - show loaded files\n"
            "  clear docs               - remove all indexed files\n"
            "  ask <question>           - search with TF-IDF + cosine\n"
            "\n"
            "  remember X is Y          - save a fact\n"
            "  what's X?                - recall a fact\n"
            "  /remember X: Y           - slash alias\n"
            "  /recall X                - slash alias\n"
            "  /memkeys                 - list memory keys\n"
            "  /memsearch <text>        - search memory\n"
            "  /forget <key>            - delete memory\n"
            "\n"
            "  add <task>               - add to-do\n"
            "  list todos               - list to-dos\n"
            "  done <n>                 - mark done\n"
            "  clear todos              - remove all\n"
            "\n"
            "  time | date | uptime     - local time/date/uptime\n"
            "  echo <text>              - repeat back\n"
            "  joke                     - random joke\n"
            "  help                     - this menu\n"
            "  bye/exit/quit            - save & exit"
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
def main(argv: List[str]):
    import argparse

    parser = argparse.ArgumentParser(
        description="Local chatbot with TF-IDF Q&A, SQLite memory, and todos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of search results (default: 3)")
    parser.add_argument("--min-score", type=float, default=0.05, help="Minimum cosine similarity (default: 0.05)")
    parser.add_argument("--no-color", action="store_true", help="Disable ANSI colors")
    args = parser.parse_args(argv[1:])

    bot = Bot(
        name="Bobo",
        top_k=args.top_k,
        min_score=args.min_score,
        use_color=not args.no_color,
    )
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
        bot.mem.close()


if __name__ == "__main__":
    sys.exit(main(sys.argv) or 0)
