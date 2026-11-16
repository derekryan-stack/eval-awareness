"""SQLite cache of valid judgments, keyed by a hash of (backend signature, rubric version, messages).

Only successfully parsed judgments are cached, so malformed outputs are retried on the next run.
"""

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from judge.backends import stable_json
from judge.rubric import PROMPT_VERSION


def cache_key(backend_signature, messages):
    payload = stable_json({"backend": backend_signature, "prompt_version": PROMPT_VERSION, "messages": messages})
    return hashlib.sha256(payload.encode()).hexdigest()


class JudgeCache:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("CREATE TABLE IF NOT EXISTS judgments (key TEXT PRIMARY KEY, value TEXT NOT NULL, created TEXT NOT NULL)")
        self._db.commit()

    def get(self, key):
        with self._lock:
            row = self._db.execute("SELECT value FROM judgments WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key, value):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO judgments (key, value, created) VALUES (?, ?, ?)",
                             (key, json.dumps(value), datetime.now(timezone.utc).isoformat()))
            self._db.commit()

    def __len__(self):
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM judgments").fetchone()[0]

    def close(self):
        self._db.close()


class NullCache:
    def get(self, key):
        return None

    def put(self, key, value):
        pass

    def __len__(self):
        return 0

    def close(self):
        pass
