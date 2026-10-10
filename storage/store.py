"""SQLite 落盘：记忆、人格轨迹、关系、对话、事件。

为什么要落盘
------------
不落盘的话世界一重启记忆就归零，而"持久的记忆"正是这个项目的核心目标之一：
只有记得住，才谈得上"人格随经历漂移"和"独特的轨迹"。

时间用**绝对分钟**（`day * 1440 + 当日分钟`）
-------------------------------------------
记忆要跨天比较新旧。如果只用当日分钟，第二天 07:40 的 `now` 会比昨天
08:00 的记忆还小，时效性算出来是负的。所以落盘层统一用绝对分钟，
显示时再 mod 1440 还原成 "HH:MM"。

线程安全
--------
对话在 `asyncio.to_thread` 里推进，写入会发生在工作线程；而 flush 在主循环。
所以连接用 `check_same_thread=False` + 一把锁——我们的写入量很小，
不值得为它引入连接池。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    agent_id    TEXT NOT NULL,
    kind        TEXT NOT NULL,
    text        TEXT NOT NULL,
    importance  INTEGER NOT NULL,
    created     REAL NOT NULL,
    last_access REAL NOT NULL,
    subject     TEXT,
    meta        TEXT,
    day         INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_mem_agent ON memories(agent_id, created);

CREATE TABLE IF NOT EXISTS trait_changes (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    day         INTEGER, clock REAL, agent_id TEXT NOT NULL, trait TEXT NOT NULL,
    old REAL, new REAL, reason TEXT, source TEXT, counterpart TEXT
);
CREATE INDEX IF NOT EXISTS idx_trait_agent ON trait_changes(agent_id, seq);

CREATE TABLE IF NOT EXISTS relations (
    a TEXT NOT NULL, b TEXT NOT NULL,
    familiarity REAL, affinity REAL, trust REAL, updated REAL,
    PRIMARY KEY (a, b)
);

CREATE TABLE IF NOT EXISTS relation_changes (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    day INTEGER, clock REAL, a TEXT, b TEXT, before_json TEXT, after_json TEXT, reason TEXT
);

CREATE TABLE IF NOT EXISTS dialogues (
    id TEXT PRIMARY KEY, started REAL, clock TEXT, place TEXT,
    a TEXT, b TEXT, topic TEXT, turns INTEGER, summary TEXT, transcript TEXT,
    day INTEGER
);

CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    t REAL, clock TEXT, type TEXT, agent TEXT, name TEXT, payload TEXT, day INTEGER
);
CREATE INDEX IF NOT EXISTS idx_event_type ON events(type);
"""


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self._migrate()
        self.conn.commit()
        self.writes = 0

    def _migrate(self) -> None:
        """轻量迁移：老库缺列就补上。够用即可——真正的迁移框架等有真实数据再说。"""
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(memories)")}
        if "day" not in cols:
            self.conn.execute("ALTER TABLE memories ADD COLUMN day INTEGER DEFAULT 0")
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(trait_changes)")}
        if "day" not in cols:
            self.conn.execute("ALTER TABLE trait_changes ADD COLUMN day INTEGER DEFAULT 0")

    # ------------------------------------------------------------ 写入

    def save_memories(self, items: Iterable[Any], day: int = 0) -> int:
        rows = [
            (m.id, m.agent_id, m.kind, m.text, m.importance, m.created,
             m.last_access, m.subject, json.dumps(m.meta, ensure_ascii=False), day)
            for m in items
        ]
        if not rows:
            return 0
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
            self.conn.commit()
        self.writes += len(rows)
        return len(rows)

    def save_trait_change(self, agent_id: str, ch: Any, day: int = 0) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO trait_changes (day, clock, agent_id, trait, old, new, reason, source, counterpart)"
                " VALUES (?,?,?,?,?,?,?,?,?)",
                (day, ch.clock, agent_id, ch.trait, ch.old, ch.new, ch.reason, ch.source, ch.counterpart))
            self.conn.commit()
        self.writes += 1

    def save_relation(self, a: str, b: str, rel: Any, updated: float = 0.0) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO relations VALUES (?,?,?,?,?,?) ON CONFLICT(a,b) DO UPDATE SET"
                " familiarity=excluded.familiarity, affinity=excluded.affinity,"
                " trust=excluded.trust, updated=excluded.updated",
                (a, b, rel.familiarity, rel.affinity, rel.trust, updated))
            self.conn.commit()

    def save_relation_change(self, ch: Any, day: int = 0) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO relation_changes (day, clock, a, b, before_json, after_json, reason)"
                " VALUES (?,?,?,?,?,?,?)",
                (day, ch.clock, ch.a, ch.b, json.dumps(ch.before, ensure_ascii=False),
                 json.dumps(ch.after, ensure_ascii=False), ch.reason))
            self.conn.commit()

    def save_dialogue(self, d: dict[str, Any], day: int = 0) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO dialogues VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (d["id"], d.get("started", 0), d.get("at", ""), d.get("place", ""),
                 d.get("a"), d.get("b"), d.get("topic", ""), len(d.get("turns", [])),
                 d.get("summary", ""), json.dumps(d.get("turns", []), ensure_ascii=False), day))
            self.conn.commit()

    def save_events(self, events: Iterable[dict[str, Any]], day: int = 0) -> int:
        rows = []
        for e in events:
            rows.append((e.get("t", 0), e.get("clock", ""), e.get("type", ""),
                         e.get("agent") or e.get("speaker"), e.get("name"),
                         json.dumps(e, ensure_ascii=False), day))
        if not rows:
            return 0
        with self._lock:
            self.conn.executemany(
                "INSERT INTO events (t, clock, type, agent, name, payload, day) VALUES (?,?,?,?,?,?,?)",
                rows)
            self.conn.commit()
        self.writes += len(rows)
        return len(rows)

    # ------------------------------------------------------------ 读取（恢复与观测）

    def load_memories(self, agent_id: str | None = None, limit: int = 400) -> list[dict[str, Any]]:
        q = "SELECT id, agent_id, kind, text, importance, created, last_access, subject, meta FROM memories"
        args: list[Any] = []
        if agent_id:
            q += " WHERE agent_id = ?"
            args.append(agent_id)
        q += " ORDER BY created DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        return [
            {"id": r[0], "agent_id": r[1], "kind": r[2], "text": r[3], "importance": r[4],
             "created": r[5], "last_access": r[6], "subject": r[7],
             "meta": json.loads(r[8]) if r[8] else {}}
            for r in rows
        ][::-1]

    def load_traits(self, agent_id: str) -> dict[str, float]:
        """当前值 = 每个特质最后一条变化记录的 new。"""
        with self._lock:
            rows = self.conn.execute(
                "SELECT trait, new FROM trait_changes WHERE agent_id = ?"
                " AND seq IN (SELECT MAX(seq) FROM trait_changes WHERE agent_id = ? GROUP BY trait)",
                (agent_id, agent_id)).fetchall()
        return {r[0]: r[1] for r in rows}

    def load_relations(self) -> dict[tuple[str, str], tuple[float, float, float]]:
        with self._lock:
            rows = self.conn.execute("SELECT a, b, familiarity, affinity, trust FROM relations").fetchall()
        return {(r[0], r[1]): (r[2], r[3], r[4]) for r in rows}

    def trait_timeline(self, agent_id: str | None = None) -> list[dict[str, Any]]:
        q = "SELECT day, clock, agent_id, trait, old, new, reason, source FROM trait_changes"
        args: list[Any] = []
        if agent_id:
            q += " WHERE agent_id = ?"
            args.append(agent_id)
        q += " ORDER BY seq"
        with self._lock:
            rows = self.conn.execute(q, args).fetchall()
        return [
            {"day": r[0], "t_abs": (r[0] or 0) * 1440 + (r[1] or 0), "clock": r[1],
             "agent_id": r[2], "trait": r[3], "old": r[4], "new": r[5],
             "reason": r[6], "source": r[7]}
            for r in rows
        ]

    def has_data(self) -> bool:
        """库里有没有东西。空库启动不该被当成"新的一天"。"""
        with self._lock:
            row = self.conn.execute(
                "SELECT (SELECT COUNT(*) FROM memories) + (SELECT COUNT(*) FROM trait_changes)"
            ).fetchone()
        return bool(row and row[0])

    def latest_day(self) -> int:
        with self._lock:
            row = self.conn.execute(
                "SELECT MAX(day) FROM (SELECT day FROM trait_changes UNION ALL SELECT day FROM memories)").fetchone()
        return int(row[0] or 0) if row else 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            def one(sql: str) -> int:
                return int(self.conn.execute(sql).fetchone()[0])
            return {
                "path": str(self.path),
                "memories": one("SELECT COUNT(*) FROM memories"),
                "trait_changes": one("SELECT COUNT(*) FROM trait_changes"),
                "relations": one("SELECT COUNT(*) FROM relations"),
                "dialogues": one("SELECT COUNT(*) FROM dialogues"),
                "events": one("SELECT COUNT(*) FROM events"),
                "writes": self.writes,
            }

    def close(self) -> None:
        with self._lock:
            self.conn.commit()
            self.conn.close()
