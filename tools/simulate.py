"""跑一段模拟并导出回放轨迹。

    python tools/simulate.py --minutes 40 --out saves/demo_1f/trace.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.engine import Engine  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="运行校园模拟并导出回放轨迹")
    ap.add_argument("--map", default="world/maps/sem_1f.json")
    ap.add_argument("--personas", default="agents/personas")
    ap.add_argument("--start", default="07:40", help="起始游戏时刻 HH:MM")
    ap.add_argument("--minutes", type=float, default=40, help="模拟多少游戏分钟")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="saves/demo_1f/trace.json")
    args = ap.parse_args(argv)

    eng = Engine(map_path=args.map, personas_dir=args.personas,
                 start_clock=args.start, seed=args.seed)
    eng.run(args.minutes)
    path = eng.write_trace(args.out)

    tr = eng.trace()
    print(f"地图　　：{tr['meta']['map_name']}")
    print(f"时段　　：{args.start} → {tr['meta']['end_clock']}"
          f"（{tr['meta']['duration_game_seconds'] / 60:.0f} 游戏分钟）")
    print(f"智能体　：{len(tr['agents'])} 人　　轨迹帧：{tr['meta']['frames']}")
    print(f"大脑　　：{tr['meta']['brain']}（占位，M2 换 LLM）")
    print()
    print(f"{'姓名':<8}{'角色':<10}{'走了':>8}{'到达':>6}  行程")
    for a in tr["agents"]:
        legs = [e for e in tr["events"] if e.get("agent") == a["id"] and e["type"] == "depart"]
        route = " → ".join(dict.fromkeys(e.get("room_name", "") for e in legs)) or "—"
        print(f"{a['name']:<8}{a['role']:<10}{a['walked_m']:>7.0f}m{a['arrivals']:>6}  {route}")
    errs = [e for e in tr["events"] if e["type"] == "error"]
    if errs:
        print("\n错误：")
        for e in errs:
            print("  ", e["message"])
    print(f"\n轨迹已写入 {path}  ({path.stat().st_size / 1024:.0f} KB)")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
