/* ============================================================
   DUT 小镇 · 像素美术库
   ------------------------------------------------------------
   全部图形都是程序化像素画，不依赖任何外部素材。
   约定：1 米 = ART 个 art pixel；先在 art 分辨率下绘制到离屏 canvas，
   再用 nearest-neighbor 放大，因此任意缩放下都是硬边像素。
   ============================================================ */

const ART = 8;                       // art 像素 / 米
/* ------------------------------------------------ 像素调色板 */
const C = {
  woodDark: '#5a3a1e', wood: '#8a6234', woodLight: '#b98d55', woodXl: '#d3a970',
  dark: '#3b2412', paper: '#f2e6c8', metal: '#b9bcc0', metalDark: '#8a8e93',
  red: '#b5462f', blue: '#3f6fa8', green: '#5d9b3f', greenLight: '#7cb342',
  water: '#7fd4f0', outline: '#2b1d10',
};

const FLOORS = {
  wood:     { base: '#c89b62', line: '#a97c46', kind: 'planks' },
  tile:     { base: '#ddd8c9', line: '#c2bcaa', kind: 'grid', gap: 1 },
  carpet:   { base: '#b5744f', line: '#9c5f3e', kind: 'dots' },
  concrete: { base: '#d2cbbb', line: '#bdb5a3', kind: 'grid', gap: 2 },
  restroom: { base: '#a9c8d4', line: '#8fb0bd', kind: 'grid', gap: 1 },
};

const GROUND_STYLE = {
  green:   { base: '#6f9b4a', line: '#5d8540' },
  plaza:   { base: '#cfc3a8', line: '#bdb096' },
  path:    { base: '#c8a06a', line: '#b08a55' },
  road:    { base: '#9a9a96', line: '#84847f' },
  parking: { base: '#a8a8a4', line: '#93938e' },
  shelter: { base: '#9fb3b8', line: '#8a9ea4' },
};

const ROOM_STYLE = {
  corridor: 'tile', lobby: 'concrete', hall: 'carpet',
  office: 'wood', classroom: 'wood', vertical: 'concrete', service: 'wood',
};

/* ------------------------------------------------ 像素画：家具 */
const DRAW = {
  desk(g, W, H) {
    g.fillStyle = C.wood; g.fillRect(0, 1, W, H - 2);
    g.fillStyle = C.woodLight; g.fillRect(1, 2, W - 2, 2);
    g.fillStyle = C.blue; g.fillRect(3, 3, 2, 2);
    g.fillStyle = C.dark; g.fillRect(0, H - 1, W, 1); g.fillRect(0, 1, 1, H - 2); g.fillRect(W - 1, 1, 1, H - 2);
  },
  chair(g, W, H) {
    g.fillStyle = C.woodDark; g.fillRect(1, 1, W - 2, H - 3);
    g.fillStyle = C.wood; g.fillRect(2, 2, W - 4, H - 6);
    g.fillStyle = C.dark; g.fillRect(2, H - 2, 1, 1); g.fillRect(W - 3, H - 2, 1, 1);
  },
  office_desk(g, W, H) {
    g.fillStyle = C.wood; g.fillRect(0, 1, W, H - 2);
    g.fillStyle = C.woodLight; g.fillRect(1, 2, W - 2, 2);
    g.fillStyle = C.metalDark; g.fillRect(2, 3, 5, 3);
    g.fillStyle = C.water; g.fillRect(3, 4, 3, 1);
    g.fillStyle = C.paper; g.fillRect(9, 4, 4, 2);
    g.fillStyle = C.dark; g.fillRect(0, H - 1, W, 1);
  },
  bookshelf(g, W, H) {
    g.fillStyle = C.woodDark; g.fillRect(0, 0, W, H);
    g.fillStyle = C.wood; g.fillRect(1, 1, W - 2, H - 2);
    const cols = [C.red, C.blue, C.green, '#d8a13a', '#8a5fb0', C.paper];
    for (let i = 0; i < W - 3; i += 2) {
      g.fillStyle = cols[(i / 2) % cols.length];
      g.fillRect(2 + i, 2, 1, H - 4);
    }
    g.fillStyle = C.dark; g.fillRect(0, 0, W, 1); g.fillRect(0, H - 1, W, 1);
  },
  plant(g, W, H) {
    g.fillStyle = '#9c5f3e'; g.fillRect(1, H - 3, W - 2, 2);
    g.fillStyle = C.green; g.fillRect(2, 1, W - 4, 4);
    g.fillStyle = C.greenLight; g.fillRect(1, 2, 2, 2); g.fillRect(W - 3, 2, 2, 2); g.fillRect(3, 0, 2, 2);
  },
  sofa(g, W, H) {
    g.fillStyle = '#8a4f5e'; g.fillRect(0, 0, W, H);
    g.fillStyle = '#a9606f'; g.fillRect(1, 1, W - 2, H - 3);
    g.fillStyle = '#c47f8c'; g.fillRect(2, 2, W / 2 - 3, 2); g.fillRect(W / 2 + 1, 2, W / 2 - 3, 2);
    g.fillStyle = C.dark; g.fillRect(0, H - 1, W, 1);
  },
  table_round(g, W, H) {
    const cx = W / 2, cy = H / 2;
    g.fillStyle = C.woodDark;
    g.beginPath(); g.ellipse(cx, cy + 1, W / 2 - 1, H / 2 - 2, 0, 0, Math.PI * 2); g.fill();
    g.fillStyle = C.woodLight;
    g.beginPath(); g.ellipse(cx, cy - 1, W / 2 - 2, H / 2 - 3, 0, 0, Math.PI * 2); g.fill();
    g.fillStyle = C.paper; g.fillRect(cx - 2, cy - 3, 3, 2);
  },
  meeting_table(g, W, H) {
    g.fillStyle = C.woodDark; g.fillRect(0, 0, W, H);
    g.fillStyle = '#9a6f3c'; g.fillRect(1, 1, W - 2, H - 2);
    g.fillStyle = C.woodLight; g.fillRect(2, 2, W - 4, H - 5);
  },
  counter(g, W, H) {
    g.fillStyle = C.woodDark; g.fillRect(0, 0, W, H);
    g.fillStyle = C.wood; g.fillRect(0, 2, W, H - 3);
    g.fillStyle = C.woodXl; g.fillRect(0, 0, W, 2);
    g.fillStyle = C.dark; g.fillRect(0, H - 1, W, 1);
  },
  reception(g, W, H) {
    DRAW.counter(g, W, H);
    g.fillStyle = C.paper; g.fillRect(W / 2 - 3, 2, 6, 2);
    g.fillStyle = C.red; g.fillRect(W / 2 - 2, 2, 4, 1);
  },
  blackboard(g, W, H) {
    g.fillStyle = C.woodDark; g.fillRect(0, 0, W, H);
    g.fillStyle = '#2f3b33'; g.fillRect(1, 1, W - 2, H - 2);
    g.fillStyle = '#dfe8dd';
    for (let x = 4; x < W - 8; x += 9) { g.fillRect(x, 3, 5, 1); g.fillRect(x + 1, 5, 3, 1); }
  },
  whiteboard(g, W, H) {
    g.fillStyle = C.metalDark; g.fillRect(0, 0, W, H);
    g.fillStyle = '#f7f7f4'; g.fillRect(1, 1, W - 2, H - 2);
    g.fillStyle = C.blue; g.fillRect(3, 2, W - 14, 1);
    g.fillStyle = C.red; g.fillRect(3, 4, W - 24, 1);
  },
  podium(g, W, H) {
    g.fillStyle = C.woodDark; g.fillRect(2, 0, W - 4, H);
    g.fillStyle = C.wood; g.fillRect(3, 1, W - 6, H - 2);
    g.fillStyle = C.woodXl; g.fillRect(1, 0, W - 2, 2);
  },
  elevator(g, W, H) {
    g.fillStyle = C.metalDark; g.fillRect(0, 0, W, H);
    g.fillStyle = C.metal; g.fillRect(1, 2, W / 2 - 2, H - 3); g.fillRect(W / 2 + 1, 2, W / 2 - 2, H - 3);
    g.fillStyle = C.dark; g.fillRect(W / 2 - 1, 0, 2, H);
    g.fillStyle = C.water; g.fillRect(W / 2 - 3, 0, 6, 1);
  },
  stairs(g, W, H) {
    for (let i = 0; i < H; i += 2) {
      g.fillStyle = i % 4 === 0 ? '#cfc8b8' : '#b8b0a0';
      g.fillRect(0, i, W, 2);
    }
    g.fillStyle = C.dark; g.fillRect(0, 0, 1, H); g.fillRect(W - 1, 0, 1, H);
  },
  coffee_machine(g, W, H) {
    g.fillStyle = C.metalDark; g.fillRect(1, 0, W - 2, H - 2);
    g.fillStyle = '#6b4a28'; g.fillRect(2, 1, W - 4, 2);
    g.fillStyle = C.paper; g.fillRect(3, H - 3, 2, 2);
  },
  water_cooler(g, W, H) {
    g.fillStyle = '#e8e8e4'; g.fillRect(1, 3, W - 2, H - 4);
    g.fillStyle = C.water; g.fillRect(2, 0, W - 4, 4);
  },
  sink(g, W, H) {
    g.fillStyle = '#e8e8e4'; g.fillRect(0, 1, W, H - 2);
    g.fillStyle = C.metal; g.fillRect(1, 2, W / 2 - 2, 3); g.fillRect(W / 2 + 1, 2, W / 2 - 2, 3);
    g.fillStyle = C.water; g.fillRect(2, 3, 2, 1); g.fillRect(W / 2 + 2, 3, 2, 1);
  },
  bulletin(g, W, H) {
    g.fillStyle = C.woodDark; g.fillRect(0, 0, W, H);
    g.fillStyle = '#c9a06a'; g.fillRect(1, 1, W - 2, H - 2);
    const cols = [C.paper, '#ffe9a8', '#d9ecff', '#ffd9d9'];
    for (let i = 0; i < 4; i++) {
      g.fillStyle = cols[i];
      g.fillRect(3 + i * ((W - 8) / 4), 2, 3, Math.max(2, H - 5));
    }
  },
  mat(g, W, H) {
    g.fillStyle = '#8a5a3a'; g.fillRect(0, 0, W, H);
    g.fillStyle = '#a9723a';
    for (let i = 0; i < W; i += 4) g.fillRect(i, 1, 2, H - 2);
    g.fillStyle = C.dark; g.fillRect(0, 0, W, 1); g.fillRect(0, H - 1, W, 1);
  },
};

const FURNITURE_LABEL = {
  desk: '课桌', 'office_desk': '办公桌', chair: '座椅', bookshelf: '书架', plant: '绿植',
  sofa: '沙发', table_round: '圆桌', meeting_table: '会议桌', counter: '吧台',
  reception: '前台', blackboard: '黑板 / 幕布', whiteboard: '白板', podium: '讲台',
  elevator: '电梯', stairs: '楼梯', coffee_machine: '咖啡机', water_cooler: '饮水机',
  sink: '洗手台', bulletin: '公告栏', mat: '地垫',
};
const spriteCache = new Map();

/* ------------------------------------------------ 像素精灵缓存 */

function sprite(type, wm, hm) {
  const key = `${type}_${wm}x${hm}`;
  if (spriteCache.has(key)) return spriteCache.get(key);
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(wm * ART));
  c.height = Math.max(1, Math.round(hm * ART));
  const g = c.getContext('2d');
  g.imageSmoothingEnabled = false;
  const fn = DRAW[type] || DRAW.desk;
  fn(g, c.width, c.height);
  spriteCache.set(key, c);
  return c;
}

/* ------------------------------------------------ 像素画：智能体
   8 × 14 art px ≈ 1m × 1.75m。四个朝向 × 四帧走路循环。
   发色/上衣/裤子/肤色由人设的 look 决定，所以每个人一眼能认出来。 */

const AGENT_W = 8;
const AGENT_H = 14;

function agentSprite(look, facing, frame) {
  look = look || {};
  const key = `ag|${facing}|${frame}|${look.hair}|${look.shirt}|${look.pants}|${look.skin}`;
  if (spriteCache.has(key)) return spriteCache.get(key);

  const c = document.createElement('canvas');
  c.width = AGENT_W;
  c.height = AGENT_H;
  const g = c.getContext('2d');
  g.imageSmoothingEnabled = false;

  const skin = look.skin || '#e8c9a0';
  const hair = look.hair || '#2b1d10';
  const shirt = look.shirt || '#3f6fa8';
  const pants = look.pants || '#4a4a52';
  const shoe = '#2b1d10';
  const line = '#3b2412';

  // 摆动量：1 = 左腿前，-1 = 右腿前，0 = 中立
  const sw = frame === 1 ? 1 : frame === 3 ? -1 : 0;

  // 腿
  g.fillStyle = pants;
  g.fillRect(1, 10, 2, 3 - (sw > 0 ? 1 : 0));
  g.fillRect(5, 10, 2, 3 - (sw < 0 ? 1 : 0));
  // 鞋
  g.fillStyle = shoe;
  g.fillRect(1, 13 - (sw > 0 ? 1 : 0), 2, 1);
  g.fillRect(5, 13 - (sw < 0 ? 1 : 0), 2, 1);

  // 手臂（与腿反向摆）
  g.fillStyle = shirt;
  g.fillRect(0, 6 + (sw < 0 ? 1 : 0), 1, 4);
  g.fillRect(7, 6 + (sw > 0 ? 1 : 0), 1, 4);

  // 身体
  g.fillRect(1, 6, 6, 4);
  g.fillStyle = 'rgba(0,0,0,.14)';
  g.fillRect(1, 9, 6, 1);

  // 头
  g.fillStyle = skin;
  g.fillRect(1, 2, 6, 4);

  // 头发
  g.fillStyle = hair;
  g.fillRect(1, 0, 6, 2);
  g.fillRect(0, 1, 1, 3);
  g.fillRect(7, 1, 1, 3);
  if (facing === 'up') g.fillRect(1, 2, 6, 2);          // 背面全是头发

  // 眼睛
  g.fillStyle = line;
  if (facing === 'down') { g.fillRect(2, 4, 1, 1); g.fillRect(5, 4, 1, 1); }
  else if (facing === 'left') g.fillRect(2, 4, 1, 1);
  else if (facing === 'right') g.fillRect(5, 4, 1, 1);

  // 描边轮廓，让小人从地板上"跳"出来
  g.fillStyle = 'rgba(43,29,16,.35)';
  g.fillRect(1, 13, 1, 1);
  g.fillRect(6, 13, 1, 1);

  spriteCache.set(key, c);
  return c;
}
