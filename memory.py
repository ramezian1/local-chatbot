# memory.py
import sqlite3
from pathlib import Path
from typing import List, Tuple, Optional
import time

DEFAULT_DB = Path("data/memory.db")

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
    def __init__(self, db_path: Path = DEFAULT_DB):
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

    def search(self, query: str, limit: int = 10) -> List[Tuple[str, str]]:
        query = query.strip()
        cur = self.conn.execute(
            "SELECT key, value FROM memories_fts WHERE memories_fts MATCH ? LIMIT ?",
            (query, limit),
        )
        rows = cur.fetchall()
        if rows:
            return rows
        # fallback (LIKE) if FTS finds nothing
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
