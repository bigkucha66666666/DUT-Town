/* ============================================================
   DUT 小镇 · 星露谷风格地图查看器
   ------------------------------------------------------------
   · 像素美术：所有图形按 8 art-px/米 的程序化像素画绘制，
     预渲染进离屏 canvas 后以 nearest-neighbor 放大，保证硬边像素观感。
   · 地板层整层预渲染成一张 buffer，只在缩放变化时重建（平移不重建）。
   · 走位判定与 Python 端 world/map_loader.py 的栅格化优先级保持一致：
     ground → buildings → rooms → 实体家具。
   ============================================================ */


const MAPS = {
  sem_1f:  { file: '../world/maps/sem_1f.json',  label: '学院楼一层',
              trace: '../saves/demo_1f/trace.json' },
  dut_sem: { file: '../world/maps/dut_sem.json', label: '学院周边', trace: null },
};

const CAT_LABEL = {
  road: '道路', plaza: '广场', green: '草地', parking: '停车场', shelter: '车棚', path: '步道',
  academic: '学院建筑', neighbor: '相邻建筑',
  corridor: '走廊', lobby: '大厅', hall: '多功能厅', lab: '实验室',
  classroom: '教室', office: '教师办公室', service: '服务用房', vertical: '垂直交通',
};
const ROLE_LABEL = { student: '学生', teacher: '教师', staff: '后勤' };
const SPOT_LABEL = { entrance: '出入口', landmark: '地标 / 集散点', transport: '交通点', gateway: '场地边界' };


/* ------------------------------------------------ 状态 */
const state = {
  map: null, mapKey: null,
  scale: 12, ox: 0, oy: 0,
  layers: { ground: true, shell: true, rooms: true, furniture: true, spawns: true,
            agents: true, labels: true, grid: false },
  floor: null,
  hover: null, selected: null, cursor: null, drag: null,
  picks: [], doors: [],
  floorDirty: true,
  // 回放
  trace: null, playT: 0, playing: true, speed: 8, loopPlay: true,
  agentPhase: {}, agentPrev: {}, agentPicks: [], agentLive: {},
  lastFrameTime: 0,
  // 实时模式
  liveOk: false, liveMapMismatch: false, mode: 'idle', pickAgent: null, liveEvents: [],
  trail: null,
  bubbles: [], bubbleMs: 5200,
};

const cvs = document.getElementById('map');
const ctx = cvs.getContext('2d');
const floorBuf = document.createElement('canvas');   // 地板层整层预渲染缓存
let cssW = 0, cssH = 0;

/* ------------------------------------------------ 载入 */

async function boot() {
  const p = new URLSearchParams(location.search);
  const key = MAPS[p.get('map')] ? p.get('map') : 'sem_1f';
  await loadScene(key);
  resize();
  fit(2);                      // 凑近到能看清智能体；按 0 或点 ⤢ 回到全览
  applyViewParams(p);
  updateStatus();
  requestAnimationFrame(loop);

  // 先给实时世界一点时间；连不上就退回回放（纯静态服务也能用）
  if (!(await waitForLive(1500))) await loadTrace();
  updateStatus();
}

function waitForLive(ms) {
  if (state.liveOk) return Promise.resolve(true);
  return new Promise((resolve) => {
    const timer = setTimeout(() => resolve(state.liveOk), ms);
    Live.on('hello', () => { clearTimeout(timer); resolve(true); });
    Live.on('unavailable', () => { clearTimeout(timer); resolve(false); });
  });
}

function enterLive(hello) {
  state.liveOk = true;
  state.trace = null;
  state.liveEvents = (hello.recent_events || []).slice();
  state.liveMapMismatch = hello.map !== state.map.meta.id;
  buildPlaybar();
  updatePlayLabel();
  updateStatus();
  toast(`已连上实时世界 · ${hello.map_name}`, true);
}

/* 视图深链：?z=缩放(px/m) &cx=&cy=视野中心(米)。便于分享某个机位。 */
function applyViewParams(p) {
  const z = parseFloat(p.get('z'));
  if (Number.isFinite(z) && z > 0) {
    state.scale = Math.max(3, Math.min(64, z));
    state.floorDirty = true;
  }
  const cx = parseFloat(p.get('cx')), cy = parseFloat(p.get('cy'));
  if (Number.isFinite(cx)) state.ox = cssW / 2 - cx * state.scale;
  if (Number.isFinite(cy)) state.oy = cssH / 2 - cy * state.scale;
  // 深链：?agent=stu_01 打开就选中这个人的档案（也方便截图与分享）
  const who = p.get('agent');
  if (who) state.autoSelect = who;

  // 气泡停留时长：默认 5.2 秒，可用 ?bubble_ms= 调（截图/演示时有用）
  const bms = parseFloat(p.get('bubble_ms'));
  if (Number.isFinite(bms) && bms > 0) state.bubbleMs = bms;
  const t = parseFloat(p.get('t'));
  if (Number.isFinite(t) && t >= 0) {
    state.playT = t;
    state.playing = p.get('play') === '1';
    updatePlayLabel();
  }
}

async function loadScene(key) {
  const res = await fetch(MAPS[key].file);
  if (!res.ok) throw new Error(`地图加载失败 ${res.status}：${MAPS[key].file}`);
  const raw = await res.json();
  raw.furniture = expandFurniture(raw.furniture || []);
  state.map = raw;
  state.mapKey = key;
  state.floor = state.map.meta.default_floor || (state.map.floors[0] || {}).id;
  state.selected = state.hover = null;
  state.floorDirty = true;
  state.errShown = false;
  state.scale = 0;                     // 强制 fit
  document.getElementById('mapfile').textContent = MAPS[key].file;
  document.title = `DUT 小镇 · ${state.map.meta.name}`;
  state.doors = computeDoors(state.map);
  state.sceneReady = false;
  buildScenes(); buildFloors(); buildToggles(); buildLegend(); showDetail(null);
  state.selected = null;
  state.mode = 'idle';
  state.playT = 0;                    // 换场景才归零；URL 里的 ?t= 由 applyViewParams 覆盖
  state.liveEvents = [];
  if (state.liveOk) { state.trace = null; buildPlaybar(); }
  else await loadTrace();
  state.sceneReady = true;
}

/* 与 Python 端 world/map_loader.expand_furniture 保持一致：
   把 grid 重复声明展开成一个个具体家具。 */
function expandFurniture(items) {
  const out = [];
  for (const it of items) {
    if (!it.grid) { out.push(it); continue; }
    const g = it.grid;
    for (let r = 0; r < g.rows; r++) {
      for (let c = 0; c < g.cols; c++) {
        const { grid, ...rest } = it;
        out.push({
          ...rest,
          id: `${it.id}_r${r + 1}c${c + 1}`,
          rect: { x: g.x + c * g.dx, y: g.y + r * g.dy, w: g.w, h: g.h },
        });
      }
    }
  }
  return out;
}

/* ------------------------------------------------ 自动推导门的位置
   房间与走廊相接的那条公共边，取中间 2m 作为门。
   这既是渲染用的门槛，也是后面给智能体用的出入口数据。 */

function computeDoors(m) {
  const out = [];
  const rooms = m.rooms.filter((r) => r.floor === state.floor);
  const corridors = rooms.filter((r) => r.category === 'corridor');
  const near = (a, b) => Math.abs(a - b) < 0.01;

  for (const r of rooms) {
    if (r.category === 'corridor') continue;
    for (const c of corridors) {
      const a = r.rect, b = c.rect;
      const xo = Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x);
      const yo = Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y);
      if (xo > 0 && near(a.y + a.h, b.y)) {
        out.push({ room: r.id, rect: { x: Math.max(a.x, b.x) + xo / 2 - 1, y: b.y - 0.35, w: 2, h: 0.7 }, axis: 'h' });
      } else if (xo > 0 && near(b.y + b.h, a.y)) {
        out.push({ room: r.id, rect: { x: Math.max(a.x, b.x) + xo / 2 - 1, y: a.y - 0.35, w: 2, h: 0.7 }, axis: 'h' });
      } else if (yo > 0 && near(a.x + a.w, b.x)) {
        out.push({ room: r.id, rect: { x: b.x - 0.35, y: Math.max(a.y, b.y) + yo / 2 - 1, w: 0.7, h: 2 }, axis: 'v' });
      } else if (yo > 0 && near(b.x + b.w, a.x)) {
        out.push({ room: r.id, rect: { x: a.x - 0.35, y: Math.max(a.y, b.y) + yo / 2 - 1, w: 0.7, h: 2 }, axis: 'v' });
      }
    }
  }
  return out;
}

/* ------------------------------------------------ UI */

function buildScenes() {
  const box = document.getElementById('scenes');
  box.innerHTML = '';
  for (const [key, def] of Object.entries(MAPS)) {
    const b = document.createElement('button');
    b.textContent = def.label;
    b.className = key === state.mapKey ? 'on' : '';
    b.onclick = async () => { await loadScene(key); resize(); fit(); updateStatus(); };
    box.appendChild(b);
  }
}

function buildFloors() {
  const box = document.getElementById('floors');
  box.innerHTML = '';
  for (const f of state.map.floors) {
    const b = document.createElement('button');
    b.className = f.id === state.floor ? 'on' : '';
    b.textContent = f.id;
    b.title = f.name + (f.status === 'pending' ? '（未建模）' : '');
    b.disabled = f.status === 'pending';
    b.onclick = () => { state.floor = f.id; state.doors = computeDoors(state.map); state.floorDirty = true; buildFloors(); };
    box.appendChild(b);
  }
}

const TOGGLE_DEFS = [
  ['ground', '地面场地'], ['shell', '墙体'], ['rooms', '房间'], ['furniture', '家具'],
  ['spawns', '出生点'], ['agents', '智能体'], ['labels', '文字'], ['grid', '米格'],
];

function buildToggles() {
  const box = document.getElementById('toggles');
  box.innerHTML = '';
  for (const [k, label] of TOGGLE_DEFS) {
    const l = document.createElement('label');
    l.className = 'toggle';
    l.innerHTML = `<input type="checkbox" ${state.layers[k] ? 'checked' : ''}><span class="box"></span><span>${label}</span>`;
    l.querySelector('input').onchange = (e) => { state.layers[k] = e.target.checked; state.floorDirty = true; };
    box.appendChild(l);
  }
}

function buildLegend() {
  const items = [
    ['#c89b62', '木地板'], ['#ddd8c9', '走廊地砖'], ['#b5744f', '地毯'],
    ['#d2cbbb', '水磨石'], ['#a9c8d4', '卫生间'], ['#6f9b4a', '草地'],
    ['#cfc3a8', '广场'], ['#c8a06a', '步道'],
  ];
  document.getElementById('legend').innerHTML = items
    .map(([c, n]) => `<li><i style="background:${c}"></i>${n}</li>`).join('');
}

/* ------------------------------------------------ 视图 */

function worldSize() { return { w: state.map.meta.world.width, h: state.map.meta.world.height }; }
function toScreen(x, y) { return [state.ox + x * state.scale, state.oy + y * state.scale]; }
function toWorld(sx, sy) { return [(sx - state.ox) / state.scale, (sy - state.oy) / state.scale]; }

function resize() {
  const dpr = window.devicePixelRatio || 1;
  cssW = cvs.clientWidth; cssH = cvs.clientHeight;
  cvs.width = Math.round(cssW * dpr);
  cvs.height = Math.round(cssH * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.imageSmoothingEnabled = false;
}

function fit(factor = 1) {
  const { w, h } = worldSize();
  const pad = 26;
  let s = Math.min((cssW - pad * 2) / w, (cssH - pad * 2) / h);
  // 有智能体时默认凑近看：整层楼装不下没关系，0 键随时切回全览
  if (factor !== 1) s = Math.min(20, s * factor);
  state.scale = s;
  state.ox = (cssW - w * s) / 2;
  state.oy = (cssH - h * s) / 2;
  state.floorDirty = true;
}

function zoomAt(sx, sy, factor) {
  const ns = Math.max(3, Math.min(64, state.scale * factor));
  const k = ns / state.scale;
  state.ox = sx - (sx - state.ox) * k;
  state.oy = sy - (sy - state.oy) * k;
  state.scale = ns;
  state.floorDirty = true;
}

/* ------------------------------------------------ 地板层预渲染 */

function styleOfRoom(r) {
  if (r.category === 'service' && /卫生间/.test(r.name)) return 'restroom';
  return ROOM_STYLE[r.category] || 'wood';
}

function renderFloor() {
  const { w: W, h: H } = worldSize();
  const s = state.scale;
  const bw = Math.max(1, Math.ceil(W * s));
  const bh = Math.max(1, Math.ceil(H * s));
  if (floorBuf.width !== bw || floorBuf.height !== bh) { floorBuf.width = bw; floorBuf.height = bh; }

  const g = floorBuf.getContext('2d');
  g.imageSmoothingEnabled = false;
  g.clearRect(0, 0, bw, bh);

  const R = (r) => [Math.round(r.x * s), Math.round(r.y * s), Math.round(r.w * s), Math.round(r.h * s)];

  // 1) 场地
  if (state.layers.ground) {
    for (const f of state.map.ground) {
      const st = GROUND_STYLE[f.category] || GROUND_STYLE.green;
      g.fillStyle = st.base;
      g.fillRect(...R(f.rect));
      if (f.category === 'green') {
        grassTufts(g, f.rect, s);
      } else if (f.category === 'plaza' || f.category === 'path') {
        slabLines(g, f.rect, s, f.category === 'path' ? 1 : 2, st.line);
      }
    }
  } else {
    g.fillStyle = '#5d8540';
    g.fillRect(0, 0, bw, bh);
  }

  // 2) 建筑墙体（房间之外的部分自然成为墙）
  if (state.layers.shell) {
    for (const b of state.map.buildings) {
      const [x, y, w, h] = R(b.rect);
      g.fillStyle = 'rgba(20,10,4,.28)'; g.fillRect(x + 3, y + 5, w, h);   // 投影
      g.fillStyle = '#7a5a3a'; g.fillRect(x, y, w, h);                     // 墙体
      g.fillStyle = '#96714a'; g.fillRect(x, y, w, Math.max(2, Math.round(0.7 * s)));       // 顶面亮边
      g.fillStyle = '#5a3f28'; g.fillRect(x, y + h - Math.max(2, Math.round(0.7 * s)), w, Math.max(2, Math.round(0.7 * s)));
      g.strokeStyle = C.outline; g.lineWidth = 3; g.strokeRect(x + 1.5, y + 1.5, w - 3, h - 3);
    }
  }

  // 3) 房间地板
  if (state.layers.rooms) {
    for (const r of state.map.rooms) {
      if (r.floor !== state.floor) continue;
      const st = FLOORS[styleOfRoom(r)];
      const [x, y, w, h] = R(r.rect);
      g.fillStyle = st.base;
      g.fillRect(x, y, w, h);
      drawFloorDetail(g, r.rect, st, s);
    }
    // 门槛：房间与走廊的公共边中间 2m
    for (const d of state.doors) {
      const [x, y, w, h] = R(d.rect);
      g.fillStyle = '#b98d55'; g.fillRect(x, y, w, h);
      g.fillStyle = C.woodDark;
      if (d.axis === 'h') { g.fillRect(x, y, Math.max(1, w / 6), h); g.fillRect(x + w - Math.max(1, w / 6), y, Math.max(1, w / 6), h); }
      else { g.fillRect(x, y, w, Math.max(1, h / 6)); g.fillRect(x, y + h - Math.max(1, h / 6), w, Math.max(1, h / 6)); }
    }
  }

  state.floorDirty = false;
}

function grassTufts(g, rect, s) {
  g.fillStyle = 'rgba(40,80,28,.30)';
  const px = Math.max(1, Math.round(s * 0.16));
  for (let x = Math.floor(rect.x); x < rect.x + rect.w; x += 2) {
    for (let y = Math.floor(rect.y); y < rect.y + rect.h; y += 2) {
      const hsh = ((x * 73856093) ^ (y * 19349663)) % 1000;
      if (hsh < 0) continue;
      if (hsh % 1000 < 260) {
        g.fillRect(Math.round((x + (hsh % 7) / 4) * s), Math.round((y + (hsh % 5) / 4) * s), px, px);
      }
    }
  }
}

function slabLines(g, rect, s, gap, color) {
  g.fillStyle = color;
  const t = Math.max(1, Math.round(s * 0.12));
  for (let x = Math.ceil(rect.x / gap) * gap; x < rect.x + rect.w; x += gap) {
    g.fillRect(Math.round(x * s), Math.round(rect.y * s), t, Math.round(rect.h * s));
  }
  for (let y = Math.ceil(rect.y / gap) * gap; y < rect.y + rect.h; y += gap) {
    g.fillRect(Math.round(rect.x * s), Math.round(y * s), Math.round(rect.w * s), t);
  }
}

function drawFloorDetail(g, rect, st, s) {
  const t = Math.max(1, Math.round(s * 0.1));
  g.fillStyle = st.line;
  if (st.kind === 'planks') {
    for (let y = Math.ceil(rect.y); y < rect.y + rect.h; y += 1) {
      g.fillRect(Math.round(rect.x * s), Math.round(y * s), Math.round(rect.w * s), t);
    }
    for (let y = Math.ceil(rect.y); y < rect.y + rect.h; y += 1) {
      const off = (Math.floor(y) % 2) * 2;
      for (let x = Math.ceil(rect.x) + off; x < rect.x + rect.w; x += 4) {
        g.fillRect(Math.round(x * s), Math.round(y * s), t, Math.round(s));
      }
    }
  } else if (st.kind === 'grid') {
    const gap = st.gap || 1;
    for (let x = Math.ceil(rect.x / gap) * gap; x < rect.x + rect.w; x += gap) {
      g.fillRect(Math.round(x * s), Math.round(rect.y * s), t, Math.round(rect.h * s));
    }
    for (let y = Math.ceil(rect.y / gap) * gap; y < rect.y + rect.h; y += gap) {
      g.fillRect(Math.round(rect.x * s), Math.round(y * s), Math.round(rect.w * s), t);
    }
  } else if (st.kind === 'dots') {
    for (let x = Math.ceil(rect.x); x < rect.x + rect.w; x += 1) {
      for (let y = Math.ceil(rect.y); y < rect.y + rect.h; y += 1) {
        if ((x + y) % 2 === 0) continue;
        g.fillRect(Math.round((x + 0.4) * s), Math.round((y + 0.4) * s), t, t);
      }
    }
  }
}

/* ------------------------------------------------ 主绘制 */

function loop() {
  try {
    advancePlayback();
    draw();
  } catch (err) {
    if (!state.errShown) {
      state.errShown = true;
      console.error(err);
      document.getElementById('detail').innerHTML =
        `<p class="detail__empty" style="color:#b5462f">渲染出错：<br>${err && err.message ? err.message : err}</p>`;
    }
    return;                       // 停止循环，避免每帧刷屏
  }
  requestAnimationFrame(loop);
}

function draw() {
  if (!state.map || cssW === 0) return;
  const { w: W, h: H } = worldSize();
  if (state.floorDirty) renderFloor();

  ctx.fillStyle = '#6f9b4a';
  ctx.fillRect(0, 0, cssW, cssH);

  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(floorBuf, Math.round(state.ox), Math.round(state.oy));

  state.picks = [];

  if (state.layers.grid) drawGrid();

  const [mx, my] = toScreen(0, 0);
  const [Mx, My] = toScreen(W, H);
  ctx.strokeStyle = C.outline; ctx.lineWidth = 3;
  ctx.strokeRect(mx - 1.5, my - 1.5, Mx - mx + 3, My - my + 3);

  if (state.liveOk && state.trail && state.trail.length > 1) drawTrail();
  if (state.layers.furniture) drawFurniture();
  if (state.layers.spawns) drawSpawns();
  if (state.layers.agents) drawAgents();
  if (state.liveOk) drawBubbles();
  if (state.layers.labels) drawLabels();

  drawSelection();
}

function drawGrid() {
  const { w: W, h: H } = worldSize();
  const [mx, my] = toScreen(0, 0);
  const [Mx, My] = toScreen(W, H);
  ctx.save();
  ctx.beginPath(); ctx.rect(mx, my, Mx - mx, My - my); ctx.clip();
  ctx.strokeStyle = 'rgba(43,29,16,.16)'; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let x = 0; x <= W; x += 5) {
    const sx = Math.round(toScreen(x, 0)[0]) + .5;
    ctx.moveTo(sx, my); ctx.lineTo(sx, My);
  }
  for (let y = 0; y <= H; y += 5) {
    const sy = Math.round(toScreen(0, y)[1]) + .5;
    ctx.moveTo(mx, sy); ctx.lineTo(Mx, sy);
  }
  ctx.stroke();
  ctx.restore();
}

function drawFurniture() {
  const s = state.scale;
  const items = state.map.furniture
    .filter((f) => !f.floor || f.floor === state.floor)
    .slice()
    .sort((a, b) => (a.rect.y + a.rect.h) - (b.rect.y + b.rect.h));

  for (const f of items) {
    const [x, y] = toScreen(f.rect.x, f.rect.y);
    const w = Math.max(1, Math.round(f.rect.w * s));
    const h = Math.max(1, Math.round(f.rect.h * s));
    const px = Math.round(x), py = Math.round(y);

    // 落影
    ctx.fillStyle = 'rgba(30,16,6,.26)';
    ctx.fillRect(px + Math.max(1, Math.round(s * 0.18)), py + Math.max(1, Math.round(s * 0.22)), w, h);

    ctx.drawImage(sprite(f.category, f.rect.w, f.rect.h), px, py, w, h);
    state.picks.push({ f, kind: 'furniture' });
  }
}

function drawSpawns() {
  const s = state.scale;
  const r = Math.max(5, Math.min(11, s * 0.55));
  for (const sp of state.map.spawns) {
    const [x, y] = toScreen(sp.pos[0], sp.pos[1]);
    ctx.fillStyle = 'rgba(30,16,6,.3)';
    ctx.beginPath(); ctx.ellipse(x, y + r * 0.7, r * 0.8, r * 0.35, 0, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = '#e8b33a';
    ctx.beginPath();
    ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y);
    ctx.closePath(); ctx.fill();
    ctx.strokeStyle = C.outline; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = '#fff6dd';
    ctx.fillRect(x - r * 0.28, y - r * 0.28, r * 0.56, r * 0.56);
    state.picks.push({ f: sp, kind: 'spawn' });
  }
}

function drawLabels() {
  const s = state.scale;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';

  for (const b of state.map.buildings) {
    const fs = Math.max(13, Math.min(26, s * 1.5));
    ctx.font = `700 ${fs}px "PingFang SC","Hiragino Sans GB",sans-serif`;
    const cx = toScreen(b.rect.x + b.rect.w / 2, 0)[0];
    const cy = toScreen(0, b.rect.y)[1] - fs;
    plaque(ctx, b.name, cx, cy, fs, C.woodXl);
  }

  if (s >= 9) {
    for (const r of state.map.rooms) {
      if (r.floor !== state.floor) continue;
      if (r.rect.w * s < 54 || r.rect.h * s < 20) continue;
      const fs = Math.max(10, Math.min(15, s * 0.72));
      ctx.font = `700 ${fs}px "PingFang SC","Hiragino Sans GB",sans-serif`;
      const lines = wrapText(ctx, r.name, r.rect.w * s - 10);
      const lh = fs * 1.25;
      if (lines.length * lh > r.rect.h * s - 8) continue;
      const cx = toScreen(r.rect.x + r.rect.w / 2, 0)[0];
      const cy = toScreen(0, r.rect.y + r.rect.h / 2)[1];
      lines.forEach((ln, i) => {
        const ty = cy + (i - (lines.length - 1) / 2) * lh;
        ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(255,248,228,.85)';
        ctx.strokeText(ln, cx, ty);
        ctx.fillStyle = '#4a2f18';
        ctx.fillText(ln, cx, ty);
      });
    }
  }

  // 出生点名字：缩放到一定程度才画，木牌放在菱形正上方，避免压住房间内容
  if (s >= 13) {
    const fs = 10.5;
    ctx.font = `700 ${fs}px "PingFang SC","Hiragino Sans GB",sans-serif`;
    for (const sp of state.map.spawns) {
      const [x, y] = toScreen(sp.pos[0], sp.pos[1]);
      plaque(ctx, sp.name, x, y - 17 - fs / 2, fs, '#e8b33a');
    }
    ctx.textAlign = 'center';
  }
}

function plaque(ctx, text, cx, cy, fs, bg) {
  const w = ctx.measureText(text).width + 12;
  const h = fs + 8;
  ctx.fillStyle = bg;
  ctx.fillRect(cx - w / 2, cy - h / 2, w, h);
  ctx.strokeStyle = C.outline; ctx.lineWidth = 2;
  ctx.strokeRect(cx - w / 2, cy - h / 2, w, h);
  ctx.fillStyle = '#3b2412';
  ctx.fillText(text, cx, cy + 1);
}

function wrapText(c, text, maxW) {
  const TAIL = '）】」》〉”’、。，；：！？%';
  const out = []; let cur = '';
  for (const ch of text) {
    if (c.measureText(cur + ch).width > maxW && cur) {
      if (TAIL.includes(ch) && cur.length > 1) { out.push(cur.slice(0, -1)); cur = cur.slice(-1) + ch; }
      else { out.push(cur); cur = ch; }
    } else cur += ch;
  }
  if (cur) out.push(cur);
  return out;
}

function drawSelection() {
  const t = state.hover || state.selected;
  if (!t) return;
  const s = state.scale;
  ctx.save();
  if (t.kind === 'agent') {
    const live = state.agentLive[t.id];
    if (live) {
      const [x, y] = toScreen(live.x, live.y);
      ctx.strokeStyle = '#e8b33a';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(x, y - s * 0.55, Math.max(9, s * 0.8), 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([4, 3]);
      ctx.strokeStyle = 'rgba(43,29,16,.6)';
      ctx.beginPath();
      ctx.arc(x, y - s * 0.55, Math.max(12, s * 1.05), 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  } else if (t.kind === 'spawn') {
    const [x, y] = toScreen(t.f.pos[0], t.f.pos[1]);
    ctx.strokeStyle = '#fff6dd'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(x, y, Math.max(10, s * 0.7), 0, Math.PI * 2); ctx.stroke();
  } else {
    const [x, y] = toScreen(t.f.rect.x, t.f.rect.y);
    const w = t.f.rect.w * s, h = t.f.rect.h * s;
    ctx.setLineDash([6, 4]);
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(43,29,16,.75)';
    ctx.strokeRect(x - 2, y - 2, w + 4, h + 4);
    ctx.strokeStyle = t === state.selected ? '#e8b33a' : '#fff6dd';
    ctx.lineWidth = 2;
    ctx.strokeRect(x - 2, y - 2, w + 4, h + 4);
    ctx.setLineDash([]);
  }
  ctx.restore();
}

/* ------------------------------------------------ 命中测试 */

function pickAt(sx, sy) {
  const [wx, wy] = toWorld(sx, sy);
  for (const a of state.agentPicks) {
    if (Math.hypot(wx - a.x, wy - a.y) < 0.9) return { kind: 'agent', id: a.id, x: a.x, y: a.y };
  }
  for (let i = state.picks.length - 1; i >= 0; i--) {
    const p = state.picks[i];
    if (p.kind === 'spawn') {
      if (Math.hypot(wx - p.f.pos[0], wy - p.f.pos[1]) * state.scale < 12) return p;
    } else {
      const r = p.f.rect;
      if (wx >= r.x && wx <= r.x + r.w && wy >= r.y && wy <= r.y + r.h) return p;
    }
  }
  return null;
}

/* ------------------------------------------------ 走位判定（对齐 Python 端） */

function layerAt(x, y) {
  const m = state.map;
  const inR = (r) => x >= r.x && x < r.x + r.w && y >= r.y && y < r.y + r.h;
  let hit = { label: '图外', walkable: false };

  for (const f of m.ground) {
    if (inR(f.rect)) hit = { label: (GROUND_STYLE[f.category] ? CAT_LABEL[f.category] : '场地'), walkable: true };
  }
  for (const b of m.buildings) if (inR(b.rect)) hit = { label: '墙体', walkable: false };
  for (const r of m.rooms) {
    if (r.floor !== state.floor) continue;
    if (inR(r.rect)) hit = { label: CAT_LABEL[r.category] || '室内', walkable: true };
  }
  for (const f of m.furniture) {
    if (!f.solid) continue;
    if (inR(f.rect)) hit = { label: FURNITURE_LABEL[f.category] || '家具', walkable: false };
  }
  return hit;
}

/* ------------------------------------------------ 详情面板 */

function showDetail(pick) {
  const el = document.getElementById('detail');
  const hint = document.getElementById('detail-hint');
  if (!pick) {
    hint.textContent = '点一下地图上的东西';
    el.innerHTML = `<p class="detail__empty">拖动看四周，滚轮放大缩小。<br>点房间、家具或出生点看属性。</p>`;
    return;
  }
  hint.textContent = '已选中';

  // 智能体 pick 带的是 id 而不是 f，必须**在**碰 pick.f 之前分流——
  // 否则这里会抛异常，面板停在上一个要素的内容上，看起来就是"点了没反应"。
  if (pick.kind === 'agent') {
    const a = agentInfo(pick.id);
    if (!a) { showDetail(null); return; }
    document.getElementById('detail-hint').textContent = '已选中';
    const where = a.live ? (a.why || '待着') : (agentActivity(a.id, state.playT) || '待着');
    const stats = a.live
      ? `<dt>位置</dt><dd><code>(${Number(a.x).toFixed(1)}, ${Number(a.y).toFixed(1)})</code></dd>
         <dt>速度</dt><dd><code>${a.speed} m/s</code></dd>`
      : `<dt>行走</dt><dd><code>${a.walked_m} m</code> · 到达 ${a.arrivals} 次</dd>
         <dt>速度</dt><dd><code>${a.speed} m/s</code></dd>`;
    const hint = state.mode === 'pickTarget'
      ? '正等他要去的地方——点地图上任意位置（Esc 取消）'
      : '点「派他去某处」，再点地图上的落点';
    el.innerHTML = `<div class="detail__name">${a.name}</div>
      <div class="detail__id" data-focus="1">${a.id}${a.live ? ' · 实时' : ''}</div>
      <dl>
        <dt>角色</dt><dd>${ROLE_LABEL[a.role] || a.role}</dd>
        <dt>身份</dt><dd>${a.identity || '—'}</dd>
        <dt>此刻</dt><dd>${where}</dd>
        ${stats}
      </dl>
      ${a.traits && a.traits.length ? `<div class="detail__note">性格 ${a.traits.join(' · ')}</div>` : ''}
      ${a.speech_style ? `<div class="detail__note">说话风格：${a.speech_style}</div>` : ''}
      ${a.goal ? `<div class="detail__note"><b>目标</b><br>${a.goal}</div>` : ''}
      ${a.live ? `<div class="detail__note"><b>干预</b>
        <div class="detail__roles" style="margin-top:7px">
          <button class="act" data-act="dispatch">派他去某处</button>
          <button class="act" data-act="stop">让他停下</button>
        </div>
        <div style="margin-top:6px;font-size:11px;color:var(--ink-3)">${hint}</div></div>` : ''}
      ${a.live ? '<div class="mem" id="mem"><p class="detail__empty">正在读取记忆…</p></div>' : ''}`;
    // 面板在侧栏底部，选中后不滚过去等于没显示
    requestAnimationFrame(() => {
      const card = el.closest('.card');
      if (card) card.scrollIntoView({ block: 'start', behavior: 'smooth' });
    });
    if (a.live) {
      const want = pick.id;
      loadTrail(want);
      loadAgentMemory(want).then((info) => {
        const box = document.getElementById('mem');
        if (!box || !info) return;
        if (!state.selected || state.selected.id !== want) return;
        box.innerHTML = memoryBlock(info);
        requestAnimationFrame(() => drawTraitChart(info));
      });
    }
    return;
  }

  const f = pick.f, rows = [];
  rows.push(['编号', `<code>${f.id}</code>`]);

  if (pick.kind === 'spawn') {
    rows.push(['坐标', `<code>(${f.pos[0]}, ${f.pos[1]})</code>`]);
    rows.push(['用途', SPOT_LABEL[f.kind] || f.kind]);
  } else if (pick.kind === 'furniture') {
    rows.push(['类型', FURNITURE_LABEL[f.category] || f.category]);
    rows.push(['所在房间', f.room]);
    rows.push(['占位', `<code>${f.rect.w} × ${f.rect.h} m</code>`]);
    rows.push(['是否挡路', f.solid ? '是（参与碰撞）' : '否（可穿过）']);
  } else if (pick.kind === 'building') {
    rows.push(['类型', CAT_LABEL[f.category] || f.category]);
    rows.push(['占地', `<code>${f.rect.w} × ${f.rect.h} m</code>`]);
    if (f.area_m2) rows.push(['建筑面积', `<code>${f.area_m2.toLocaleString()} m²</code>`]);
    if (f.height_m) rows.push(['建筑高度', `${f.height_m} m`]);
    if (f.modeled_floors) rows.push(['已建模', `<code>${f.modeled_floors.join(' / ')}</code>（共 ${f.floors.length} 层）`]);
    if (f.entrances) rows.push(['出入口', f.entrances.map((e) => e.name).join('、')]);
  } else {
    const r = f.rect;
    rows.push(['类型', CAT_LABEL[f.category] || f.category]);
    rows.push(['位置', `<code>(${r.x}, ${r.y})</code>`]);
    rows.push(['尺寸', `<code>${r.w} × ${r.h} m</code>`]);
    rows.push(['面积', `<code>${(r.w * r.h).toLocaleString()} m²</code>`]);
  }

  const title = pick.kind === 'building' ? f.name : (FURNITURE_LABEL[f.category] || f.name);
  const head = pick.kind === 'building' ? f.name : `${f.name}`;

  let html = `<div class="detail__name">${head}</div>
    <div class="detail__id" data-focus="${f.focus ? 1 : 0}">${f.id}</div>
    <dl>${rows.map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join('')}</dl>`;

  if (f.capacity) html += `<div class="detail__note">容量 <b>${f.capacity}</b> 人 · 开放时间 ${f.open_hours || '—'}</div>`;
  if (f.allowed_roles) {
    html += `<div class="detail__note">可进入角色<div class="detail__roles">
      ${f.allowed_roles.map((r) => `<span class="role" data-r="${r}">${ROLE_LABEL[r] || r}</span>`).join('')}</div></div>`;
  }
  if (f.note) html += `<div class="detail__note">${f.note}</div>`;
  el.innerHTML = html;
}

/* ------------------------------------------------ 内心世界 */

async function loadAgentMemory(id) {
  if (!state.liveOk) return null;
  try {
    const [rm, rt] = await Promise.all([
      fetch(`/api/memory?agent=${encodeURIComponent(id)}&limit=10`, { cache: 'no-store' }),
      fetch(`/api/timeline?agent=${encodeURIComponent(id)}`, { cache: 'no-store' }),
    ]);
    if (!rm.ok) return null;
    const d = await rm.json();
    const info = d.agents[id];
    if (!info) return null;
    if (rt.ok) {
      const tl = await rt.json();
      info._series = (tl.agents && tl.agents[id] && tl.agents[id].series) || [];
    }
    info._relations = Object.entries(d.relations || {})
      .filter(([k]) => k.split('|').includes(id))
      .map(([k, v]) => ({ other: k.split('|').find((x) => x !== id), ...v }))
      .sort((x, y) => y.affinity - x.affinity);
    return info;
  } catch (err) {
    console.warn('读取记忆失败', err);
    return null;
  }
}

const TRAIT_LABEL = { social: '社交活跃', diligence: '学业投入', initiative: '主动性', stability: '情绪稳定' };

async function loadTrail(id) {
  if (!state.liveOk) return;
  try {
    const r = await fetch(`/api/trail?agent=${encodeURIComponent(id)}`, { cache: 'no-store' });
    if (!r.ok) return;
    const d = await r.json();
    if (state.selected && state.selected.id === id) state.trail = d.trail || [];
  } catch (err) {
    console.warn('读取轨迹失败', err);
  }
}
const KIND_LABEL = { observation: '观察', event: '经历', dialogue: '对话', reflection: '反思' };

function memoryBlock(info) {
  const p = info.personality || {};
  const bars = Object.entries(TRAIT_LABEL).map(([k, label]) => {
    const v = p[k] ?? 0.5;
    return `<div class="trait"><span>${label}</span>
      <i class="track"><b style="width:${Math.round(v * 100)}%"></b></i>
      <em>${v.toFixed(2)}</em></div>`;
  }).join('');

  const mems = (info.recent || []).slice(0, 8).map((m) => `
    <li class="k-${m.kind}"><i>${m.clock}</i><span class="imp">${m.importance}</span>
      ${m.text}<em>${KIND_LABEL[m.kind] || m.kind}${m.subject ? ' · ' + m.subject : ''}</em></li>`).join('')
    || '<li class="muted">还没有记忆</li>';

  const drifts = (info.trait_history || []).slice(-6).reverse().map((c) => `
    <li class="drift"><i>${c.at}</i>${c.label} ${c.old.toFixed(2)} → ${c.new.toFixed(2)}
      <em>${c.source === 'dialogue' ? '对话' : '反思'}：${c.reason}</em></li>`).join('')
    || '<li class="muted">人格还没变过</li>';

  const rels = (info._relations || []).slice(0, 5).map((r) => `
    <li class="rel"><i>${(r.other || '').slice(0, 8)}</i>
      熟悉 ${Number(r.familiarity).toFixed(2)} · 好感 ${Number(r.affinity).toFixed(2)}
      <i class="track sm"><b style="width:${Math.round(Number(r.affinity) * 100)}%"></b></i></li>`).join('')
    || '<li class="muted">还没跟谁熟起来</li>';

  return `
    <h3 class="mem__h">人格（会随经历漂移）</h3>
    <div class="traits">${bars}</div>
    <h3 class="mem__h">人格变化曲线 <small>横轴＝游戏时间</small></h3>
    <div class="chartbox"><canvas id="trait-chart" width="260" height="96"></canvas></div>
    <div class="chartkey" id="trait-key"></div>
    <h3 class="mem__h">关系</h3>
    <ul class="memlist">${rels}</ul>
    <h3 class="mem__h">最近的记忆 <small>共 ${info.stats ? info.stats.count : 0} 条</small></h3>
    <ul class="memlist">${mems}</ul>
    <h3 class="mem__h">人格变化</h3>
    <ul class="memlist">${drifts}</ul>`;
}

/* 人格变化曲线：把 trait_changes 折成阶梯线。
   横轴是绝对游戏分钟（跨天也不会倒流），纵轴 0–1。
   没有变化记录的特质画一条平线——那本身就是"她一直没变"这个事实。 */
const TRAIT_COLOR = { social: '#e8b33a', diligence: '#7cb342', initiative: '#6ec6e8', stability: '#c47f8c' };

function drawTraitChart(info) {
  const cv = document.getElementById('trait-chart');
  const key = document.getElementById('trait-key');
  if (!cv) return;
  const rows = info._series || [];
  const cur = info.personality || {};
  const traits = Object.keys(TRAIT_LABEL);

  // 折成阶梯：每条特质从首次变化的 old 出发，之后每次变化走一步
  const series = {};
  for (const r of rows) {
    if (!series[r.trait]) series[r.trait] = [{ t: r.t_abs, v: r.old }];
    series[r.trait].push({ t: r.t_abs, v: r.new });
  }
  let tMin = Infinity, tMax = -Infinity;
  for (const pts of Object.values(series)) {
    for (const p of pts) { tMin = Math.min(tMin, p.t); tMax = Math.max(tMax, p.t); }
  }
  const hasHistory = Number.isFinite(tMin) && tMax > tMin;

  const dpr = window.devicePixelRatio || 1;
  const W = 260, H = 96, padL = 22, padR = 6, padT = 8, padB = 14;
  cv.width = W * dpr; cv.height = H * dpr;
  cv.style.width = W + 'px'; cv.style.height = H + 'px';
  const g = cv.getContext('2d');
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);

  // 背景与网格
  g.fillStyle = '#f7f0dd';
  g.fillRect(padL, padT, W - padL - padR, H - padT - padB);
  g.strokeStyle = '#ddd2b4';
  g.lineWidth = 1;
  g.font = '9px "SF Mono",monospace';
  g.fillStyle = '#98754a';
  for (const v of [0, 0.5, 1]) {
    const y = Math.round(padT + (1 - v) * (H - padT - padB)) + .5;
    g.beginPath(); g.moveTo(padL, y); g.lineTo(W - padR, y); g.stroke();
    g.fillText(v.toFixed(1), 2, y + 3);
  }

  const X = (t) => hasHistory
    ? padL + (t - tMin) / (tMax - tMin) * (W - padL - padR)
    : padL;
  const Y = (v) => padT + (1 - Math.max(0, Math.min(1, v))) * (H - padT - padB);

  for (const k of traits) {
    const pts = series[k];
    g.strokeStyle = TRAIT_COLOR[k];
    g.lineWidth = 2;
    g.beginPath();
    if (!pts || !pts.length) {
      const v = cur[k] ?? 0.5;
      g.moveTo(padL, Y(v)); g.lineTo(W - padR, Y(v));
    } else if (!hasHistory) {
      g.moveTo(padL, Y(cur[k] ?? pts[0].v)); g.lineTo(W - padR, Y(cur[k] ?? pts[0].v));
    } else {
      let prevY = Y(pts[0].v);
      g.moveTo(X(pts[0].t), prevY);
      for (let i = 1; i < pts.length; i++) {
        const x = X(pts[i].t), y = Y(pts[i].v);
        g.lineTo(x, prevY);        // 阶梯：先横走
        g.lineTo(x, y);            // 再竖跳
        prevY = y;
      }
      g.lineTo(W - padR, prevY);   // 一直延伸到"现在"
    }
    g.stroke();

    // 末点画个点
    const v = cur[k] ?? 0.5;
    g.fillStyle = TRAIT_COLOR[k];
    g.beginPath(); g.arc(W - padR - 1, Y(v), 2.5, 0, Math.PI * 2); g.fill();
  }

  // 起点/终点时刻
  g.fillStyle = '#98754a';
  g.font = '9px "SF Mono",monospace';
  if (hasHistory) {
    const fmt = (t) => {
      const m = Math.round(t) % 1440;
      return `${String(Math.floor(m / 60)).padStart(2, '0')}:${String(m % 60).padStart(2, '0')}`;
    };
    g.fillText(fmt(tMin), padL, H - 3);
    const end = fmt(tMax);
    g.fillText(end, W - padR - g.measureText(end).width, H - 3);
  } else {
    g.fillText('还没有变化', padL, H - 3);
  }

  if (key) {
    key.innerHTML = traits.map((k) =>
      `<span><i style="background:${TRAIT_COLOR[k]}"></i>${TRAIT_LABEL[k]}</span>`).join('');
  }
}

/* ------------------------------------------------ 事件 */

document.getElementById('detail').addEventListener('click', (e) => {
  const btn = e.target.closest('button.act');
  if (!btn || !state.selected || state.selected.kind !== 'agent') return;
  const id = state.selected.id;
  if (btn.dataset.act === 'dispatch') {
    state.mode = 'pickTarget';
    state.pickAgent = id;
    cvs.classList.add('picking');
    showDetail(state.selected);
    updateStatus();
    toast('点地图上任意位置，就派他过去');
  } else if (btn.dataset.act === 'stop') {
    if (!Live.send({ cmd: 'stop', agent: id })) toast('实时连接已断开', false);
  }
});

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && state.mode === 'pickTarget') {
    state.mode = 'idle';
    state.pickAgent = null;
    cvs.classList.remove('picking');
    showDetail(state.selected);
    updateStatus();
  }
});

cvs.addEventListener('pointerdown', (e) => {
  state.drag = { x: e.clientX, y: e.clientY, ox: state.ox, oy: state.oy, moved: false };
  cvs.setPointerCapture(e.pointerId);
  cvs.classList.add('dragging');
});

cvs.addEventListener('pointermove', (e) => {
  const b = cvs.getBoundingClientRect();
  const sx = e.clientX - b.left, sy = e.clientY - b.top;
  const [wx, wy] = toWorld(sx, sy);
  state.cursor = [wx, wy];

  if (state.drag) {
    const dx = e.clientX - state.drag.x, dy = e.clientY - state.drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) state.drag.moved = true;
    state.ox = state.drag.ox + dx;
    state.oy = state.drag.oy + dy;
  }

  const hit = pickAt(sx, sy);
  state.hover = hit;
  cvs.classList.toggle('hot', !!hit && !state.drag);

  const tip = document.getElementById('tip');
  if (hit && !state.drag) {
    tip.hidden = false;
    tip.style.left = sx + 'px';
    tip.style.top = sy + 'px';
    if (hit.kind === 'agent') {
      const info = agentInfo(hit.id);
      tip.innerHTML = info
        ? `<b>${info.name}</b><span>${ROLE_LABEL[info.role] || info.role}${info.why ? ' · ' + info.why : ''}</span>`
        : '';
    } else {
      const l = layerAt(wx, wy);
      const sub = hit.kind === 'spawn'
        ? (SPOT_LABEL[hit.f.kind] || hit.f.kind)
        : (FURNITURE_LABEL[hit.f.category] || CAT_LABEL[hit.f.category] || hit.f.category);
      tip.innerHTML = `<b>${hit.f.name}</b><span>${sub} · ${l.label}</span>`;
    }
  } else tip.hidden = true;

  updateStatus();
});

cvs.addEventListener('pointerup', (e) => {
  const moved = state.drag && state.drag.moved;
  state.drag = null;
  cvs.classList.remove('dragging');
  if (moved) return;
  const b = cvs.getBoundingClientRect();
  const sx = e.clientX - b.left, sy = e.clientY - b.top;

  // 待选目标模式：这一次点击就是"派他去哪儿"
  if (state.mode === 'pickTarget' && state.pickAgent) {
    const [wx, wy] = toWorld(sx, sy);
    if (!Live.send({ cmd: 'dispatch', agent: state.pickAgent, x: wx, y: wy, why: '你让他过去' })) {
      toast('实时连接已断开，指令没发出去', false);
    }
    state.mode = 'idle';
    state.pickAgent = null;
    cvs.classList.remove('picking');
    showDetail(state.selected);
    return;
  }

  state.selected = pickAt(sx, sy);
  state.trail = null;
  showDetail(state.selected);
});

cvs.addEventListener('pointerleave', () => {
  document.getElementById('tip').hidden = true;
  state.hover = null; state.cursor = null;
  updateStatus();
});

cvs.addEventListener('wheel', (e) => {
  e.preventDefault();
  const b = cvs.getBoundingClientRect();
  zoomAt(e.clientX - b.left, e.clientY - b.top, Math.exp(-e.deltaY * 0.0016));
}, { passive: false });

cvs.addEventListener('dblclick', (e) => {
  const b = cvs.getBoundingClientRect();
  zoomAt(e.clientX - b.left, e.clientY - b.top, 1.6);
});

document.querySelector('.hud').addEventListener('click', (e) => {
  const btn = e.target.closest('button');
  if (!btn) return;
  const a = btn.dataset.act;
  if (a === 'in') zoomAt(cssW / 2, cssH / 2, 1.3);
  if (a === 'out') zoomAt(cssW / 2, cssH / 2, 0.77);
  if (a === 'fit') fit();
  if (a === 'grid') { state.layers.grid = !state.layers.grid; btn.classList.toggle('on', state.layers.grid); }
  if (a === 'lbl') { state.layers.labels = !state.layers.labels; btn.classList.toggle('on', state.layers.labels); }
});

window.addEventListener('keydown', (e) => {
  if (e.key === '+' || e.key === '=') zoomAt(cssW / 2, cssH / 2, 1.3);
  if (e.key === '-') zoomAt(cssW / 2, cssH / 2, 0.77);
  if (e.key === '0') fit();
  if (e.key.toLowerCase() === 'g') { state.layers.grid = !state.layers.grid; syncHud(); }
  if (e.key.toLowerCase() === 'l') { state.layers.labels = !state.layers.labels; syncHud(); }
});

function syncHud() {
  document.querySelector('[data-act="grid"]').classList.toggle('on', state.layers.grid);
  document.querySelector('[data-act="lbl"]').classList.toggle('on', state.layers.labels);
}

window.addEventListener('resize', () => {
  const pw = cssW, ph = cssH;
  resize();
  state.ox += (cssW - pw) / 2;
  state.oy += (cssH - ph) / 2;
  state.floorDirty = true;
});

/* ------------------------------------------------ 状态栏 */

function updateStatus() {
  const c = state.cursor;
  document.getElementById('st-coord').textContent = c ? `x ${c[0].toFixed(1)} / y ${c[1].toFixed(1)}` : 'x — / y —';
  document.getElementById('st-zoom').textContent = `${state.scale.toFixed(1)} px/m`;
  const m = state.map;
  if (c) {
    const l = layerAt(c[0], c[1]);
    document.getElementById('st-layer').textContent = l.label;
    const w = document.getElementById('st-walk');
    w.textContent = l.walkable ? '可通行' : '不可通行';
    w.style.color = l.walkable ? '#a8e07a' : '#ff9a7a';
  } else {
    document.getElementById('st-layer').textContent = `${m.meta.world.width} × ${m.meta.world.height} m`;
    const w = document.getElementById('st-walk');
    if (state.mode === 'pickTarget') {
      w.textContent = '点地图选落点（Esc 取消）';
      w.style.color = '#ffd479';
    } else {
      w.textContent = (state.liveOk ? '实时世界' : '回放') + ` · ${m.rooms.length} 房间 · ${m.furniture.length} 家具`;
      w.style.color = state.liveOk ? '#a8e07a' : '';
    }
  }
}


/* ================================================================
   智能体回放
   ----------------------------------------------------------------
   轨迹由 tools/simulate.py 生成（Python 端 A* 寻路 + 占位规则大脑）。
   这里只负责按游戏时间插值播放，不参与任何决策——
   决策留在 Python，避免同一套逻辑在两端各写一遍而慢慢走样。
   ================================================================ */

const AGENT_FACINGS = ['down', 'up', 'left', 'right'];

async function loadTrace() {
  const def = MAPS[state.mapKey];
  state.trace = null;
  state.agentPhase = {}; state.agentPrev = {}; state.agentLive = {};
  if (!def || !def.trace) { buildPlaybar(); return; }
  try {
    const res = await fetch(def.trace, { cache: 'no-store' });
    if (!res.ok) throw new Error(`${res.status} ${def.trace}`);
    state.trace = await res.json();
    state.traceUrl = def.trace;
    state.playing = true;
  } catch (err) {
    console.warn('回放轨迹加载失败：', err.message);
  }
  buildPlaybar();
  updatePlayLabel();
}

function traceDuration() {
  const F = state.trace && state.trace.frames;
  return F && F.length ? F[F.length - 1].t : 0;
}

function advancePlayback() {
  const now = performance.now();
  const dt = state.lastFrameTime ? Math.min(0.1, (now - state.lastFrameTime) / 1000) : 0;
  state.lastFrameTime = now;
  if (state.liveOk || !state.trace || !state.playing || dt <= 0) return;
  state.playT += dt * state.speed;
  const end = traceDuration();
  if (state.playT >= end) state.playT = state.loopPlay ? 0 : end;
  updatePlayLabel();
}

function frameIndex(t) {
  const F = state.trace.frames;
  let lo = 0, hi = F.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (F[mid].t <= t) lo = mid; else hi = mid - 1;
  }
  return lo;
}

function agentInfo(id) {
  if (state.liveOk) {
    const m = Live.info(id);
    if (!m) return null;
    const live = Live.agents.find((a) => a.id === id) || {};
    return {
      ...m, live: true,
      x: live.x, y: live.y,
      why: live.why || '', room: live.room,
      kind: live.kind || 'routine', path: live.path || [],
    };
  }
  const a = state.trace && state.trace.agents.find((x) => x.id === id);
  if (!a) return null;
  return { ...a, live: false, why: agentActivity(id, state.playT), path: [] };
}

/* 当前这一帧所有智能体的位置。实时与回放两个来源在这里统一，
   上层绘制与交互只认这个数组，不必关心数据从哪来。 */
function currentAgents() {
  if (state.liveOk) {
    if (state.liveMapMismatch) return [];
    return (Live.agents || []).map((a) => ({
      id: a.id, x: a.x, y: a.y,
      facing: AGENT_FACINGS[a.f] || 'down',
      moving: a.m === 1,
    }));
  }
  if (!state.trace) return [];
  const F = state.trace.frames;
  const i = frameIndex(state.playT);
  const f0 = F[i], f1 = F[Math.min(i + 1, F.length - 1)];
  const span = Math.max(0.001, f1.t - f0.t);
  const k = Math.min(1, Math.max(0, (state.playT - f0.t) / span));
  const out = [];
  for (const a of state.trace.agents) {
    const A = f0.a[a.id];
    if (!A) continue;
    const B = f1.a[a.id] || A;
    out.push({
      id: a.id,
      x: A[0] + (B[0] - A[0]) * k,
      y: A[1] + (B[1] - A[1]) * k,
      facing: AGENT_FACINGS[A[2]] || 'down',
      moving: A[3] === 1,
    });
  }
  return out;
}

/* 选中者的空间轨迹：越早的点越淡，末端有个小箭头表示当前朝向 */
function drawTrail() {
  const pts = state.trail;
  ctx.save();
  ctx.lineWidth = Math.max(2, state.scale * 0.18);
  ctx.lineCap = 'round';
  for (let i = 1; i < pts.length; i++) {
    const k = i / pts.length;
    const [x0, y0] = toScreen(pts[i - 1][0], pts[i - 1][1]);
    const [x1, y1] = toScreen(pts[i][0], pts[i][1]);
    ctx.strokeStyle = `rgba(196,92,44,${0.12 + 0.6 * k})`;
    ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
  }
  const first = pts[0], last = pts[pts.length - 1];
  const [fx, fy] = toScreen(first[0], first[1]);
  ctx.fillStyle = 'rgba(196,92,44,.75)';
  ctx.beginPath(); ctx.arc(fx, fy, Math.max(3, state.scale * 0.22), 0, Math.PI * 2); ctx.fill();
  const [lx, ly] = toScreen(last[0], last[1]);
  ctx.fillStyle = '#c45c2c';
  ctx.beginPath(); ctx.arc(lx, ly, Math.max(4, state.scale * 0.3), 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#fff6dd'; ctx.lineWidth = 2; ctx.stroke();
  ctx.restore();
}

function drawAgents() {
  const s = state.scale;
  const list = currentAgents();
  state.agentPicks = [];
  state.agentLive = {};
  if (!list.length) return;
  list.sort((p, q) => p.y - q.y);          // 靠下的后画，形成前后遮挡

  const w = Math.max(4, Math.round(1.0 * s));
  const h = Math.max(6, Math.round(1.75 * s));

  // 选中者先画剩余路径，压在人身下
  if (state.liveOk && state.selected && state.selected.kind === 'agent') {
    const info = agentInfo(state.selected.id);
    if (info && info.path && info.path.length) {
      ctx.save();
      ctx.strokeStyle = 'rgba(232,179,58,.85)';
      ctx.lineWidth = Math.max(2, s * 0.16);
      ctx.setLineDash([Math.max(3, s * 0.4), Math.max(3, s * 0.35)]);
      ctx.beginPath();
      const [ax, ay] = toScreen(info.x, info.y);
      ctx.moveTo(ax, ay);
      for (const pt of info.path) { const [px, py] = toScreen(pt[0], pt[1]); ctx.lineTo(px, py); }
      ctx.stroke();
      ctx.setLineDash([]);
      const last = info.path[info.path.length - 1];
      const [ex, ey] = toScreen(last[0], last[1]);
      ctx.fillStyle = '#e8b33a';
      ctx.beginPath(); ctx.arc(ex, ey, Math.max(4, s * 0.28), 0, Math.PI * 2); ctx.fill();
      ctx.strokeStyle = '#3b2412'; ctx.lineWidth = 2; ctx.stroke();
      ctx.restore();
    }
  }

  for (const p of list) {
    const prev = state.agentPrev[p.id];
    if (prev) state.agentPhase[p.id] = (state.agentPhase[p.id] || 0) + Math.hypot(p.x - prev.x, p.y - prev.y) * 3.4;
    state.agentPrev[p.id] = { x: p.x, y: p.y };
    const frame = p.moving ? Math.floor(state.agentPhase[p.id]) % 4 : 0;

    const [sx, sy] = toScreen(p.x, p.y);
    const px = Math.round(sx - w / 2);
    const py = Math.round(sy - h * 0.84);

    ctx.fillStyle = 'rgba(30,16,6,.3)';
    ctx.beginPath();
    ctx.ellipse(Math.round(sx), Math.round(sy) - 1, w * 0.46, Math.max(1.5, w * 0.2), 0, 0, Math.PI * 2);
    ctx.fill();

    const info = agentInfo(p.id) || {};
    ctx.drawImage(agentSprite(info.look, p.facing, frame), px, py, w, h);

    state.agentLive[p.id] = { x: p.x, y: p.y };
    state.agentPicks.push({ id: p.id, x: p.x, y: p.y });
  }

  if (state.layers.labels && s >= 11) {
    const fs = Math.max(10, Math.min(13, s * 0.6));
    ctx.font = `700 ${fs}px "PingFang SC","Hiragino Sans GB",sans-serif`;
    ctx.textAlign = 'center';
    for (const p of list) {
      const info = agentInfo(p.id) || {};
      const [sx, sy] = toScreen(p.x, p.y);
      plaque(ctx, info.name || p.id, sx, sy - h * 0.84 - fs,
        fs, info.role === 'teacher' ? '#9ec6e8' : '#e8b33a');
    }
  }

  // 待选目标模式：鼠标处画落点标记
  if (state.mode === 'pickTarget' && state.cursor) {
    const [cx, cy] = toScreen(state.cursor[0], state.cursor[1]);
    ctx.save();
    ctx.strokeStyle = '#e8b33a';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(cx, cy, Math.max(9, s * 0.7), 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(cx - s, cy); ctx.lineTo(cx + s, cy);
    ctx.moveTo(cx, cy - s); ctx.lineTo(cx, cy + s);
    ctx.stroke();
    ctx.restore();
  }
}

/* 说话气泡。生命周期按**真实时间**算（5 秒），
   这样不管时间流速调到多少，一句话都留得住看得清。 */
function drawBubbles() {
  const now = performance.now();
  state.bubbles = state.bubbles.filter((b) => b.until > now);
  if (!state.bubbles.length) return;
  const s = state.scale;

  for (const b of state.bubbles) {
    const live = state.agentLive[b.id];
    const wx = live ? live.x : b.x;
    const wy = live ? live.y : b.y;
    if (!Number.isFinite(wx) || !Number.isFinite(wy)) continue;   // 坐标未知就跳过，别画到画面外
    const [sx, sy] = toScreen(wx, wy);
    const fs = Math.max(11, Math.min(14, s * 0.62));
    ctx.font = `500 ${fs}px "PingFang SC","Hiragino Sans GB",sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const lines = wrapText(ctx, b.text, Math.max(120, Math.min(230, s * 13)));
    const lh = fs * 1.4;
    const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + 18;
    const h = lines.length * lh + 14 + fs;
    const bx = sx - w / 2;
    const by = sy - s * 1.9 - h - 8;

    const left = (b.until - now) / state.bubbleMs;
    ctx.globalAlpha = Math.max(0.15, Math.min(1, left * 3));

    // 小尾巴先画，再让气泡盖住接缝
    ctx.fillStyle = '#fffdf6';
    ctx.strokeStyle = '#3b2412';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx - 6, by + h - 2);
    ctx.lineTo(sx + 6, by + h - 2);
    ctx.lineTo(sx, by + h + 8);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();

    if (ctx.roundRect) {
      ctx.beginPath(); ctx.roundRect(bx, by, w, h, 7);
    } else {
      ctx.beginPath(); ctx.rect(bx, by, w, h);
    }
    ctx.fillStyle = '#fffdf6';
    ctx.fill();
    ctx.strokeStyle = '#3b2412';
    ctx.lineWidth = 2;
    ctx.stroke();

    // 说话人名字
    ctx.font = `700 ${fs - 1}px "PingFang SC","Hiragino Sans GB",sans-serif`;
    ctx.fillStyle = '#0b4d8f';
    ctx.fillText(b.name, sx, by + 8 + fs * 0.5);
    // 台词
    ctx.font = `500 ${fs}px "PingFang SC","Hiragino Sans GB",sans-serif`;
    ctx.fillStyle = '#3b2412';
    lines.forEach((ln, i) => ctx.fillText(ln, sx, by + 10 + fs + lh * (i + 0.5)));

    ctx.globalAlpha = 1;
  }
  ctx.textBaseline = 'alphabetic';
}

/* 当前在干什么：取该时刻之前最近的一次出发/到达事件 */
function agentActivity(id, t) {
  const ev = state.trace && state.trace.events;
  if (!ev) return '';
  let act = '';
  for (const e of ev) {
    if (e.t > t) break;
    if (e.agent !== id) continue;
    if (e.type === 'dialogue_start' && (e.a === id || e.b === id)) {
      const other = e.a === id ? e.b_name : e.a_name;
      act = `正和${other}聊「${e.topic}」`;
    } else if (e.type === 'dialogue_end' && (e.a === id || e.b === id)) {
      act = e.summary || '刚聊完';
    } else if (e.type === 'depart') act = `前往${e.room_name || ''} · ${e.why || ''}`;
    else if (e.type === 'arrive') act = `在${e.room_name || roomName(e.room)} · ${e.why || ''}`;
  }
  return act;
}

function roomName(id) {
  const r = state.map && state.map.rooms.find((x) => x.id === id);
  return r ? r.name : (id || '');
}

/* ------------------------------------------------ 回放控件 */

function buildPlaybar() {
  const bar = document.getElementById('playbar');
  if (!bar) return;
  const live = state.liveOk;
  if (!live && !state.trace) { bar.hidden = true; return; }
  bar.hidden = false;

  const speeds = live ? (Live.speeds || [1, 2, 4, 8, 16, 60]) : [1, 2, 4, 8, 16];
  document.getElementById('speeds').innerHTML = speeds
    .map((v) => `<button data-s="${v}">${v}×</button>`).join('');

  const lb = document.getElementById('livebadge');
  if (lb) lb.hidden = !live;

  const sc = document.getElementById('scrub');
  sc.hidden = live;                       // 实时世界没有"总长度"，不能拖
  if (!live) {
    sc.max = String(Math.max(1, Math.round(traceDuration())));
    sc.value = String(Math.round(state.playT));
  }
  const lp = document.getElementById('loopplay');
  if (lp) lp.hidden = live;
  const pb = document.getElementById('playbtn');
  if (pb) pb.textContent = (live ? Live.paused : !state.playing) ? '▶' : '❚❚';
  syncSpeedButtons();
}

/* 速度按钮选中态：实时模式下服务端会回推 speed，用它校正 UI */
function syncSpeedButtons() {
  const bar = document.getElementById('playbar');
  if (!bar) return;
  const cur = state.liveOk ? Live.speed : state.speed;
  bar.querySelectorAll('[data-s]').forEach((b) => b.classList.toggle('on', Number(b.dataset.s) === cur));
}

function toast(msg, ok = true) {
  const el = document.getElementById('toast');
  if (!el || !msg) return;
  el.textContent = msg;
  el.className = 'toast' + (ok ? '' : ' bad');
  el.hidden = false;
  clearTimeout(toast._t);
  toast._t = setTimeout(() => { el.hidden = true; }, 2600);
}

function feedVerb(e) {
  if (e.type === 'reflection') return '反思了一会儿';
  if (e.type === 'dialogue_start') return `和${e.b_name}聊起「${e.topic}」`;
  if (e.type === 'dialogue_end') return `聊完了「${e.topic}」`;
  if (e.type === 'trait_change') return `${e.label} ${e.delta > 0 ? '↑' : '↓'}`;
  if (e.type === 'depart') return `前往${e.room_name || roomName(e.room)}`;
  if (e.type === 'dispatch') return `被派往${e.room_name || '某处'}`;
  if (e.type === 'stop') return '被叫停';
  if (e.type === 'intervene_failed') return '过不去那里';
  return `到达${roomName(e.room)}`;
}

function renderFeed(events) {
  const feed = document.getElementById('feed');
  if (!feed) return;
  const ev = events.filter((e) => e.type !== 'error' && e.type !== 'intervene_failed').slice(-6).reverse();
  feed.innerHTML = ev.length
    ? ev.map((e) => {
        if (e.type === 'dialogue_line') {
          return `<li class="say"><i>${e.clock}</i><b>${e.name}</b>：${e.text}</li>`;
        }
        if (e.type === 'dialogue_end') {
          return `<li><i>${e.clock}</i>${e.a_name}×${e.b_name} 聊完「${e.topic}」<em>${e.summary || ''}</em></li>`;
        }
        if (e.type === 'reflection') {
          return `<li class="think"><i>${e.clock}</i>${e.name} 想明白了点什么<em>${(e.insights || []).join(' / ')}</em></li>`;
        }
        if (e.type === 'trait_change') {
          return `<li class="drift"><i>${e.at || e.clock}</i>${e.name} ${e.label} ${e.old} → ${e.new}<em>${e.reason}</em></li>`;
        }
        return `<li><i>${e.clock}</i>${e.name || ''} ${feedVerb(e)}<em>${e.why || ''}</em></li>`;
      }).join('')
    : '<li class="muted">暂无事件</li>';
}

function updatePlayLabel() {
  const el = document.getElementById('playclock');
  if (state.liveOk) {
    if (el) el.textContent = Live.clock || '--:--';
    renderFeed(state.liveEvents);
    syncSpeedButtons();
    const pb = document.getElementById('playbtn');
    if (pb) pb.textContent = Live.paused ? '▶' : '❚❚';
    return;
  }
  if (!state.trace) return;
  const sc = document.getElementById('scrub');
  if (sc && document.activeElement !== sc) sc.value = String(Math.round(state.playT));
  if (el) el.textContent = fmtClock(state.playT);
  renderFeed(state.trace.events.filter((e) => e.t <= state.playT));
  syncSpeedButtons();
}

function fmtClock(t) {
  const meta = state.trace.meta;
  const [h, m] = (meta.start_clock || '00:00').split(':').map(Number);
  let total = h * 60 + m + t / 60;
  total = ((total % 1440) + 1440) % 1440;
  return `${String(Math.floor(total / 60)).padStart(2, '0')}:${String(Math.floor(total % 60)).padStart(2, '0')}`;
}

function bindPlaybar() {
  const bar = document.getElementById('playbar');
  if (!bar) return;
  bar.addEventListener('click', (e) => {
    const b = e.target.closest('button');
    if (!b) return;
    if (b.id === 'playbtn') {
      if (state.liveOk) Live.send({ cmd: 'toggle_paused' });
      else { state.playing = !state.playing; b.textContent = state.playing ? '❚❚' : '▶'; }
    } else if (b.id === 'loopplay') {
      state.loopPlay = !state.loopPlay;
      b.classList.toggle('on', state.loopPlay);
    } else if (b.dataset.s) {
      const v = Number(b.dataset.s);
      if (state.liveOk) Live.send({ cmd: 'set_speed', value: v });
      else state.speed = v;
      syncSpeedButtons();
    }
  });
  const sc = document.getElementById('scrub');
  sc.addEventListener('input', () => {
    state.playT = Number(sc.value);
    state.playing = false;
    document.getElementById('playbtn').textContent = '▶';
    state.agentPhase = {}; state.agentPrev = {};
    updatePlayLabel();
  });
}

/* ------------------------------------------------ 启动 */

bindPlaybar();

/* ------------------------------------------------ 实时世界接线 */

Live.on('hello', (hello) => enterLive(hello));

Live.on('state', (msg) => {
  // 首帧之后自动选中深链指定的智能体
  if (state.autoSelect && !state.selected && state.sceneReady) {
    const id = state.autoSelect;
    if ((msg.agents || []).some((a) => a.id === id)) {
      state.autoSelect = null;
      state.selected = { kind: 'agent', id };
      state.trail = null;
      showDetail(state.selected);
    }
  }
  // 气泡以"当前进行中的对话"为准，而不是只靠一次性事件——
  // 否则刚连上页面的人什么都看不到，而且事件一旦漏收气泡就没了。
  const now = performance.now();
  for (const d of msg.dialogues || []) {
    if (!d.last) continue;
    const key = `${d.id}#${d.turns}`;
    const exist = state.bubbles.find((b) => b.key === key);
    if (exist) {
      exist.until = now + state.bubbleMs;      // 还在聊，续期
    } else {
      // 同一说话人只留最新一条，避免头顶叠一摞气泡
      state.bubbles = state.bubbles.filter((b) => b.id !== d.last.speaker);
      // 坐标必须取"这个人此刻的位置"——对话状态里的 last 只有台词，没有坐标
      const sp = (msg.agents || []).find((a) => a.id === d.last.speaker);
      state.bubbles.push({
        key, id: d.last.speaker, name: d.last.name, text: d.last.text,
        x: sp ? sp.x : null, y: sp ? sp.y : null,
        until: now + state.bubbleMs,
      });
    }
  }
  updatePlayLabel();
  if (state.mode !== 'pickTarget') updateStatus();
});

Live.on('events', (evs) => {
  state.liveEvents = state.liveEvents.concat(evs).slice(-200);
  const now = performance.now();
  for (const e of evs) {
    if (e.type === 'dialogue_line') {
      const key = `${e.dialogue}#${e.turn}`;
      if (!state.bubbles.some((b) => b.key === key)) {
        state.bubbles = state.bubbles.filter((b) => b.id !== e.speaker);
        state.bubbles.push({ key, id: e.speaker, name: e.name, text: e.text,
                             x: e.x, y: e.y, until: now + state.bubbleMs });
      }
      if (state.bubbles.length > 10) state.bubbles.shift();
    } else if (e.type === 'dispatch') {
      toast(`${e.name || ''} 被派往${e.room_name || '某处'}`, true);
    } else if (e.type === 'intervene_failed') {
      toast(`${e.name || ''} 过不去那里`, false);
    } else if (e.type === 'dialogue_start') {
      toast(`${e.a_name} 和 ${e.b_name} 聊了起来 · ${e.topic}`, true);
    }
  }
  updatePlayLabel();
});

Live.on('ack', (msg) => {
  toast(msg.message, msg.ok);
  const pb = document.getElementById('playbtn');
  if (pb) pb.textContent = msg.paused ? '▶' : '❚❚';
  syncSpeedButtons();
});

Live.on('disconnect', () => toast('实时连接断开，正在重连…', false));

Live.on('unavailable', () => {
  state.liveOk = false;
  if (!state.trace) loadTrace();
});

boot().catch((err) => {
  console.error(err);
  document.getElementById('detail').innerHTML =
    `<p class="detail__empty" style="color:#b5462f">地图加载失败：${err.message}<br><br>
     请用本地服务打开（别直接双击 html）：<br><code>python tools/serve.py</code><br>
     然后访问 <code>http://127.0.0.1:8123/web/</code></p>`;
});
