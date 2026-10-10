"""启动世界服务：python -m server"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from server.app import create_app  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="DUT 小镇 · 实时世界服务")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--map", default="world/maps/sem_1f.json")
    ap.add_argument("--personas", default="agents/personas")
    ap.add_argument("--start", default="07:40", help="起始游戏时刻 HH:MM")
    ap.add_argument("--speed", type=float, default=8.0, help="时间流速：游戏秒 / 真实秒")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save", default="saves/demo_1f/world.db",
                    help="SQLite 存档路径；传空字符串则不落盘")
    ap.add_argument("--fresh", action="store_true", help="清空存档，从零开始")
    ap.add_argument("--provider", choices=["mock", "cloud", "local"], default=None,
                    help="临时指定模型通道，不改 config.json（真实模型很花钱，这样不容易忘关）")
    ap.add_argument("--log-level", default="warning")
    args = ap.parse_args(argv)

    app = create_app(
        map_path=args.map, personas_dir=args.personas,
        start_clock=args.start, speed=args.speed, seed=args.seed,
        save_path=args.save or None, resume=not args.fresh,
        provider=args.provider,
    )
    print(f"地图查看器：http://{args.host}:{args.port}/web/")
    print(f"世界状态　：http://{args.host}:{args.port}/api/state")
    print(f"WebSocket ：ws://{args.host}:{args.port}/ws")
    print("Ctrl+C 停止")
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
