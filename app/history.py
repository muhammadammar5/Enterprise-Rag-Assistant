
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


class HistoryStore:
    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._lock = threading.Lock()
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS history (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                   question TEXT NOT NULL,
                   answer TEXT NOT NULL,
                   sources TEXT NOT NULL,
                   total_ms REAL,
                   ttft_ms REAL
               )"""
        )
        self._conn.commit()

    def add(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        total_ms: Optional[float],
        ttft_ms: Optional[float] = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO history (question, answer, sources, total_ms, ttft_ms) "
                "VALUES (?, ?, ?, ?, ?)",
                (question, answer, json.dumps(sources), total_ms, ttft_ms),
            )
            self._conn.commit()

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, created_at, question, answer, sources, total_ms, ttft_ms "
                "FROM history ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "id": r[0],
                "created_at": r[1],
                "question": r[2],
                "answer": r[3],
                "sources": json.loads(r[4]),
                "total_ms": r[5],
                "ttft_ms": r[6],
            }
            for r in rows
        ]
