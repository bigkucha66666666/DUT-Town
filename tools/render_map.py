"""把语义地图渲染成 PNG 预览图（Pillow）。

    python tools/render_map.py --map world/maps/dut_sem.json --out docs/preview/dut_sem.png

配色走"制图仪器"路线：暖纸底 + 墨线 + 大工蓝作为主体建筑强调色。
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from world.map_loader import MapData, load_map, rasterize  # noqa: E402

# ---------------------------------------------------------------- 配色

PAPER = (242, 237, 226)
PAPER_DK = (233, 227, 214)
INK = (28, 28, 30)
INK_SOFT = (110, 106, 100)
ACCENT = (11, 77, 143)
ACCENT_SOFT = (74, 124, 176)
WARM = (196, 92, 44)

FILL = {
    "road": (206, 199, 184),
    "path": (223, 216, 201),
    "plaza": (228, 219, 200),
    "parking": (214, 209, 197),
    "shelter": (203, 212, 214),
    "green": (186, 205, 172),
}

FONT_CANDIDATES = {
    "serif": ["/System/Library/Fonts/Supplemental/Songti.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"],
    "sans": ["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Medium.ttc"],
}


def pick_font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES[kind]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default(size)


def text_size(draw: ImageDraw.ImageDraw, text: str, font) -> tuple[int, int]:
    l, t, r, b = draw.textbbox((0, 0), text, font=font)
    return r - l, b - t


def wrap_cjk(text: str, font, max_w: int, draw) -> list[str]:
    """按字符宽度折行（中文场景够用，避免引入额外依赖）。"""
    lines, cur = [], ""
    for ch in text:
        trial = cur + ch
        if text_size(draw, trial, font)[0] > max_w and cur:
            lines.append(cur)
            cur = ch
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def draw_label(draw, text, cx, cy, box_w, box_h, font, fill, line_gap=2, max_lines=4):
    """在给定矩形内居中绘制（自动折行）；放不下就跳过，返回是否绘制。"""
    if box_w < 24 or box_h < 14:
        return False
    lines = wrap_cjk(text, font, box_w - 8, draw)
    if len(lines) > max_lines:
        return False
    lh = text_size(draw, "字", font)[1] + line_gap
    total_h = lh * len(lines)
    if total_h > box_h - 6:
        return False
    y = cy - total_h / 2
    for line in lines:
        w, _ = text_size(draw, line, font)
        draw.text((cx - w / 2, y), line, font=font, fill=fill)
        y += lh
    return True


# ---------------------------------------------------------------- 渲染


def render(m: MapData, out: Path, scale: int = 4, margin: int = 58, header: int = 166) -> Path:
    W = float(m.world["width"])
    H = float(m.world["height"])
    map_w, map_h = int(W * scale), int(H * scale)
    img_w, img_h = map_w + margin * 2, map_h + margin * 2 + header

    img = Image.new("RGB", (img_w, img_h), PAPER)
    d = ImageDraw.Draw(img)

    f_title = pick_font("serif", 40)
    f_sub = pick_font("sans", 17)
    f_bname = pick_font("sans", 19)
    f_room = pick_font("sans", 12)
    f_tiny = pick_font("sans", 11)
    f_legend = pick_font("sans", 13)

    ox, oy = margin, margin + header

    def T(rect):
        return (
            ox + rect.x * scale,
            oy + rect.y * scale,
            ox + rect.x2 * scale,
            oy + rect.y2 * scale,
        )

    # --- 底纹：细网格，传达"图纸"质感
    for gx in range(0, int(W) + 1, 20):
        x = ox + gx * scale
        d.line([(x, oy), (x, oy + map_h)], fill=PAPER_DK, width=1)
    for gy in range(0, int(H) + 1, 20):
        y = oy + gy * scale
        d.line([(ox, y), (ox + map_w, y)], fill=PAPER_DK, width=1)

    # --- 地面层
    order = {"green": 0, "parking": 1, "shelter": 2, "path": 3, "plaza": 4, "road": 5}
    for f in sorted(m.ground, key=lambda f: order.get(f.category, 9)):
        d.rectangle(T(f.rect), fill=FILL.get(f.category, PAPER_DK))
    for f in m.ground:
        if f.category == "road":
            d.rectangle(T(f.rect), outline=(190, 182, 165), width=1)

    # --- 建筑
    for b in m.buildings:
        focus = bool(b.get("focus"))
        box = T(b.rect)
        d.rectangle(box, fill=ACCENT if focus else (150, 148, 152), outline=INK, width=2)
        if focus:
            # 主体建筑加一圈"光晕"，强调这是本图主角
            d.rectangle(
                (box[0] - 5, box[1] - 5, box[2] + 5, box[3] + 5),
                outline=ACCENT_SOFT,
                width=2,
            )

    # --- 室内房间（覆盖在建筑之上，空隙自然读作墙体）
    for r in m.rooms:
        is_corridor = r.category == "corridor"
        d.rectangle(T(r.rect), fill=(233, 226, 211) if is_corridor else (250, 246, 236),
                    outline=(170, 160, 142), width=1)

    # --- 家具（PNG 里只画简化色块，精细像素画在 web 端）
    for f in m.furniture:
        box = T(f.rect)
        if f.get("solid"):
            d.rectangle(box, fill=(150, 122, 88), outline=(96, 72, 48), width=1)
        else:
            d.rectangle(box, fill=(206, 186, 152), outline=(168, 146, 112), width=1)

    # --- 出生点
    for sp in m.spawns:
        x = ox + sp["pos"][0] * scale
        y = oy + sp["pos"][1] * scale
        rr = 5
        d.polygon([(x, y - rr), (x + rr, y), (x, y + rr), (x - rr, y)], fill=WARM, outline=PAPER)

    # --- 建筑名：主体建筑标在楼体上方，邻楼标在楼体下方，避免与图例、室内标签打架
    for b in m.buildings:
        focus = bool(b.get("focus"))
        label = b.raw.get("name", b.name)
        fnt = pick_font("sans", 20 if focus else 17)
        tw, th = text_size(d, label, fnt)
        bx = ox + b.rect.center()[0] * scale
        by = oy + (b.rect.y * scale - 12 if focus else b.rect.y2 * scale + 14)
        pad = 6
        d.rectangle(
            (bx - tw / 2 - pad, by - th / 2 - pad + 2, bx + tw / 2 + pad, by + th / 2 + pad),
            fill=PAPER, outline=ACCENT if focus else (188, 182, 170), width=2 if focus else 1,
        )
        d.text((bx - tw / 2, by - th / 2 - 1), label, font=fnt, fill=ACCENT if focus else INK)

    # --- 房间名
    for r in m.rooms:
        cx, cy = r.rect.center()
        draw_label(
            d, r.raw.get("name", r.name),
            ox + cx * scale, oy + cy * scale,
            r.rect.w * scale, r.rect.h * scale,
            f_room, (96, 92, 86) if r.category == "corridor" else (60, 58, 54),
            max_lines=3,
        )

    # --- 出生点标签
    for sp in m.spawns:
        x = ox + sp["pos"][0] * scale
        y = oy + sp["pos"][1] * scale
        label = sp.get("name", sp["id"])
        tw, th = text_size(d, label, f_tiny)
        pad = 3
        d.rectangle((x + 8, y - th / 2 - pad, x + 8 + tw + pad * 2, y + th / 2 + pad), fill=PAPER)
        d.text((x + 8 + pad, y - th / 2), label, font=f_tiny, fill=(150, 62, 30))

    # --- 图框
    d.rectangle((ox - 1, oy - 1, ox + map_w, oy + map_h), outline=INK, width=2)

    # --- 指北针
    nx, ny = ox + map_w - 46, oy + 46
    d.polygon([(nx, ny - 22), (nx + 9, ny + 14), (nx, ny + 6), (nx - 9, ny + 14)],
              fill=INK, outline=INK)
    d.text((nx - 6, ny + 16), "N", font=f_legend, fill=INK)

    # --- 比例尺
    sx, sy = ox + 24, oy + map_h - 30
    seg = 50 * scale
    for i in range(2):
        d.rectangle(
            (sx + i * seg, sy, sx + (i + 1) * seg, sy + 9),
            fill=INK if i % 2 == 0 else PAPER, outline=INK,
        )
    d.text((sx, sy - 22), "0", font=f_tiny, fill=INK)
    d.text((sx + seg - 10, sy - 22), "50", font=f_tiny, fill=INK)
    d.text((sx + 2 * seg - 22, sy - 22), "100 m", font=f_tiny, fill=INK)

    # --- 图例（标题下方横排四列两行，避免压住地图内容）
    lx, ly = margin, 102
    col_w, row_h = 142, 21
    entries = [(FILL.get(c, PAPER_DK), n) for c, n in
               [("road", "道路"), ("plaza", "广场"), ("green", "绿地"),
                ("parking", "停车场"), ("shelter", "车棚"), ("path", "步道")]]
    entries.append((ACCENT, "经济管理学院楼"))
    entries.append((WARM, "出生点 / 导航点"))
    for i, (color, name) in enumerate(entries):
        col, row = divmod(i, 4)
        ex = lx + col * col_w
        ey = ly + row * row_h
        d.rectangle((ex, ey + 3, ex + 15, ey + 14), fill=color, outline=(150, 145, 135))
        d.text((ex + 22, ey + 1), name, font=f_legend, fill=(70, 68, 64))

    # --- 标题区
    d.text((margin, 20), m.meta["name"], font=f_title, fill=INK)
    d.text(
        (margin + 2, 76),
        f"{m.meta['id']} · v{m.meta['version']} · 网格 {m.tile_size:g}m · "
        f"{int(W)}m × {int(H)}m · 房间 {len(m.rooms)} · 家具 {len(m.furniture)}",
        font=f_sub, fill=INK_SOFT,
    )
    d.line([(margin, header - 12), (img_w - margin, header - 12)], fill=(200, 192, 178), width=1)
    d.text((img_w - margin - 236, 26), "DUT TOWN · 校园多智能体模拟", font=f_sub, fill=ACCENT_SOFT)

    # --- 页脚免责声明（完整版，底部留白处）
    footer = m.meta["disclaimer"]
    fw, _ = text_size(d, footer, f_tiny)
    max_fw = img_w - margin * 2
    while fw > max_fw and len(footer) > 12:
        footer = footer[:-2]
        fw, _ = text_size(d, footer + "…", f_tiny)
    d.text((margin, img_h - 26), footer + ("…" if footer != m.meta["disclaimer"] else ""),
           font=f_tiny, fill=(158, 152, 144))

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="渲染地图 PNG 预览")
    ap.add_argument("--map", default="world/maps/dut_sem.json")
    ap.add_argument("--out", default="docs/preview/dut_sem.png")
    ap.add_argument("--scale", type=int, default=4, help="每米像素数")
    args = ap.parse_args(argv)

    m = load_map(args.map)
    path = render(m, Path(args.out), scale=args.scale)
    grid = rasterize(m)
    print(f"已生成 {path}  ({path.stat().st_size / 1024:.0f} KB, {grid.cols}x{grid.rows} 格)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
