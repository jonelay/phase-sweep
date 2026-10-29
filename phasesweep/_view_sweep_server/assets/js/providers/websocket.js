// WebSocket provider: job progress push channel with exponential-backoff
// reconnect. Speaks py/view_sweep_server/protocol.py.
//
//   subscribe(type, cb) -> unsubscribe   type: a server message type or "*"
//   subscribeJobs(ids) / unsubscribeJobs(ids) / cancel(id)  (client -> server)
//   onReconnect(cb)                       run a resync after every (re)open
//   connected -> boolean
//
// Emits vs:connection-change and vs:job-* on document (spec/events.toml)
// so panels can bind by name without holding the provider.

const emit = (name, detail) =>
  globalThis.document?.dispatchEvent(new CustomEvent(`vs:${name}`, { detail }));

export function defaultWsUrl(path = "/ws") {
  const { protocol, host } = globalThis.location ?? { protocol: "http:", host: "localhost" };
  return `${protocol === "https:" ? "wss" : "ws"}://${host}${path}`;
}

// Pure: dispatch one parsed message to listeners; exported for tests.
export function dispatch(listeners, msg) {
  if (!msg || typeof msg.type !== "string") return false;
  for (const cb of listeners.get(msg.type) ?? []) cb(msg);
  for (const cb of listeners.get("*") ?? []) cb(msg);
  return true;
}

export class WebSocketProvider {
  constructor(url = defaultWsUrl(), {
    minBackoffMs = 1000, maxBackoffMs = 30000,
    WebSocketImpl = globalThis.WebSocket,
    emitEvents = true,
  } = {}) {
    this.url = url;
    this.minBackoffMs = minBackoffMs;
    this.maxBackoffMs = maxBackoffMs;
    this.WebSocketImpl = WebSocketImpl;
    this.emitEvents = emitEvents;
    this.listeners = new Map();
    this.reconnectCbs = [];
    this.ws = null;
    this.state = "disconnected";
    this.backoffMs = minBackoffMs;
    this.closedByUser = false;
    this.timer = 0;
  }

  get connected() { return this.state === "connected"; }

  #setState(state) {
    this.state = state;
    if (this.emitEvents) emit("connection-change", { state });
  }

  connect() {
    this.closedByUser = false;
    this.#setState("connecting");
    const ws = new this.WebSocketImpl(this.url);
    this.ws = ws;
    ws.onopen = () => {
      this.backoffMs = this.minBackoffMs;
      this.#setState("connected");
      for (const cb of this.reconnectCbs) cb();
    };
    ws.onmessage = (ev) => {
      let msg;
      try { msg = JSON.parse(ev.data); } catch { return; }
      this.handle(msg);
    };
    ws.onclose = () => {
      this.#setState("disconnected");
      if (this.closedByUser) return;
      this.timer = setTimeout(() => this.connect(), this.backoffMs);
      this.backoffMs = Math.min(this.backoffMs * 2, this.maxBackoffMs);
    };
    return this;
  }

  close() {
    this.closedByUser = true;
    clearTimeout(this.timer);
    this.ws?.close();
  }

  handle(msg) {
    if (!dispatch(this.listeners, msg)) return;
    if (this.emitEvents && msg.type.startsWith("job_")) {
      const { type, ...detail } = msg;
      emit(type.replace("_", "-"), detail);
    }
  }

  send(msg) {
    if (this.ws?.readyState === 1) { this.ws.send(JSON.stringify(msg)); return true; }
    return false;
  }

  subscribe(type, cb) {
    if (!this.listeners.has(type)) this.listeners.set(type, new Set());
    this.listeners.get(type).add(cb);
    return () => this.listeners.get(type)?.delete(cb);
  }

  onReconnect(cb) {
    this.reconnectCbs.push(cb);
    return () => { this.reconnectCbs = this.reconnectCbs.filter((c) => c !== cb); };
  }

  subscribeJobs(jobIds) { return this.send({ type: "subscribe", job_ids: [...jobIds] }); }
  unsubscribeJobs(jobIds) { return this.send({ type: "unsubscribe", job_ids: [...jobIds] }); }
  cancel(jobId) { return this.send({ type: "cancel", job_id: jobId }); }

  // DataProvider.fetch is not a push concern; pair with FetchProvider.
  fetch() { return Promise.reject(new Error("WebSocketProvider has no fetch; use FetchProvider")); }
}
