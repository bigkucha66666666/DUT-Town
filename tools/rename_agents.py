"""改名迁移：把已落盘的数据里的旧名字一并改掉。

为什么需要它
------------
人设文件改名只影响"以后"。库里已经写下的记忆文本、对话记录、事件里
仍带着旧名字（「和 陈昊 在走廊聊了几句」），而这个人现在叫郭佳仑——
演示时一眼就能看出对不上。

用法
----
    python tools/rename_agents.py                       # 用内置的默认映射
    python tools/rename_agents.py --map 林小雨=李子雨 --map 陈昊=郭佳仑
    python tools/rename_agents.py --db saves/demo_1f/world.db --dry-run
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# 本项目 2026-10 那次改名的映射（旧 → 新）
DEFAULT_MAP = {
    "林小雨": "李子雨",
    "陈昊": "郭佳仑",
    "苏芮": "房舜尧",
    "王建国": "王士鹏",
    "李明远": "贾大巍",
}


def rename_in_db(db: Path, mapping: dict[str, str], dry_run: bool = False) -> dict[str, int]:
    con = sqlite3.connect(db)
    counts = {"memories": 0, "dialogues": 0, "events": 0, "trait_changes": 0}

    def apply(text: str) -> tuple[str, bool]:
        before = text
        for old, new in mapping.items():
            text = text.replace(old, new)
        return text, text != before

    # 记忆：正文与"相关对象"
    rows = con.execute("SELECT id, text, subject FROM memories").fetchall()
    for mid, text, subject in rows:
        new_text, c1 = apply(text or "")
        new_subj, c2 = apply(subject or "")
        if c1 or c2:
            counts["memories"] += 1
            if not dry_run:
                con.execute("UPDATE memories SET text=?, subject=? WHERE id=?",
                            (new_text, new_subj or None, mid))

    # 对话：逐句记录（含说话人 id 对应的显示名）与摘要
    rows = con.execute("SELECT id, transcript, summary FROM dialogues").fetchall()
    for did, transcript, summary in rows:
        turns = json.loads(transcript or "[]")
        changed = False
        for t in turns:
            new_name, c = apply(t.get("name", ""))
            if c:
                t["name"] = new_name
                changed = True
        new_summary, c = apply(summary or "")
        changed = changed or c
        if changed:
            counts["dialogues"] += 1
            if not dry_run:
                con.execute("UPDATE dialogues SET transcript=?, summary=? WHERE id=?",
                            (json.dumps(turns, ensure_ascii=False), new_summary, did))

    # 事件：整条 payload 里替换
    rows = con.execute("SELECT seq, payload FROM events").fetchall()
    for seq, payload in rows:
        new_payload, changed = apply(payload or "")
        if changed:
            counts["events"] += 1
            if not dry_run:
                con.execute("UPDATE events SET payload=? WHERE seq=?", (new_payload, seq))

    # 人格轨迹的理由里也可能提到人
    rows = con.execute("SELECT seq, reason FROM trait_changes").fetchall()
    for seq, reason in rows:
        new_reason, changed = apply(reason or "")
        if changed:
            counts["trait_changes"] += 1
            if not dry_run:
                con.execute("UPDATE trait_changes SET reason=? WHERE seq=?", (new_reason, seq))

    if not dry_run:
        con.commit()
    con.close()
    return counts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="把落盘数据里的旧名字改成新名字")
    ap.add_argument("--db", default=None, help="指定一个库；不给就处理 saves/ 下所有库")
    ap.add_argument("--map", action="append", default=None,
                    help="旧=新，可重复；不给就用内置默认映射")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    mapping = DEFAULT_MAP
    if args.map:
        mapping = {}
        for pair in args.map:
            if "=" not in pair:
                print(f"✗ 格式应为 旧=新，收到 {pair!r}")
                return 2
            old, new = pair.split("=", 1)
            mapping[old.strip()] = new.strip()

    dbs = [Path(args.db)] if args.db else sorted((ROOT / "saves").rglob("*.db"))
    if not dbs:
        print("saves/ 下没有找到 .db")
        return 1

    print("映射：" + "，".join(f"{k}→{v}" for k, v in mapping.items()))
    print()
    total = 0
    for db in dbs:
        if not db.exists():
            print(f"跳过不存在的 {db}")
            continue
        c = rename_in_db(db, mapping, args.dry_run)
        n = sum(c.values())
        total += n
        flag = "（试运行，未写入）" if args.dry_run else ""
        print(f"{db.relative_to(ROOT)}：改了 {n} 处 {c} {flag}")
    print(f"\n合计 {total} 处。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
