/* ============================================================
   实时世界客户端
   ------------------------------------------------------------
   连上 ws://<host>/ws 之后，世界由服务端常驻推进，这里只收状态、
   发指令。连不上也不报错——app.js 会退回回放模式，
   所以 tools/serve.py 的纯静态模式仍然能用。
   ============================================================ */

(function () {
  const Live = {
    connected: false,
    giving_up: false,
    agents: [],
    meta: {},
    paused: false,
    speed: 8,
    speeds: [1, 2, 4, 8, 16, 60],
    clock: '--:--',
    events: [],
    ws: null,
    _retry: 0,
    _handlers: { hello: [], state: [], events: [], ack: [] },

    on(kind, fn) {
      (this._handlers[kind] || (this._handlers[kind] = [])).push(fn);
    },

    emit(kind, payload) {
      for (const fn of this._handlers[kind] || []) {
        try { fn(payload); } catch (err) { console.error(`[live:${kind}]`, err); }
      }
    },

    send(cmd) {
      if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return false;
      this.ws.send(JSON.stringify(cmd));
      return true;
    },

    /** 人设：按 id 取，给详情面板和名牌用 */
    info(id) {
      return (this.meta.agents || []).find((a) => a.id === id) || null;
    },

    connect() {
      if (this.connected || this._connecting) return;
      this._connecting = true;
      const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
      let ws;
      try {
        ws = new WebSocket(`${proto}//${location.host}/ws`);
      } catch (err) {
        this._fail(err);
        return;
      }
      this.ws = ws;

      ws.onopen = () => {
        this._connecting = false;
        this._retry = 0;
      };

      ws.onmessage = (ev) => {
        let msg;
        try { msg = JSON.parse(ev.data); } catch { return; }
        switch (msg.type) {
          case 'hello':
            this.meta = msg;
            this.speeds = msg.speeds || this.speeds;
            this.speed = msg.speed ?? this.speed;
            this.paused = !!msg.paused;
            this.events = (msg.recent_events || []).slice(-200);
            this.clock = msg.clock || this.clock;
            this.connected = true;
            this.emit('hello', msg);
            break;
          case 'state':
            this.agents = msg.agents || [];
            this.clock = msg.clock;
            this.paused = !!msg.paused;
            this.speed = msg.speed ?? this.speed;
            this.t = msg.t;
            this.emit('state', msg);
            break;
          case 'events':
            this.events = (this.events.concat(msg.events || [])).slice(-200);
            this.emit('events', msg.events || []);
            break;
          case 'ack':
            this.emit('ack', msg);
            break;
          default:
            break;
        }
      };

      ws.onerror = () => { /* onclose 会跟着来，统一在那里处理 */ };

      ws.onclose = () => {
        this._connecting = false;
        const was = this.connected;
        this.connected = false;
        this.ws = null;
        if (was) this.emit('disconnect', {});
        this._fail(new Error('连接已断开'));
      };
    },

    _fail(err) {
      if (this.giving_up) return;
      this._retry += 1;
      if (this._retry > 12) {          // 大概确实没有后端，别再刷了
        this.giving_up = true;
        this.emit('unavailable', { error: String(err && err.message || err) });
        return;
      }
      const wait = Math.min(4000, 250 * 2 ** Math.min(this._retry, 4));
      setTimeout(() => this.connect(), wait);
    },
  };

  window.Live = Live;
  Live.connect();
})();
