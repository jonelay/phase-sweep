// Agent activity panel: every MCP request the server handled — client
// connects and tool calls with arguments, duration, and outcome — newest
// first. Polls GET /api/mcp/activity with a seq cursor; a changed boot id
// means the server restarted and the log starts over. Jobs an agent
// submits are adopted into the Jobs panel so their progress streams live.

const POLL_MS = 1500;
const MAX_ROWS = 300;

let app = null;
let listEl = null;
let hintEl = null;
let statusEl = null;
let pauseBtn = null;
let timer = null;
let boot = null;
let seq = 0;
let paused = false;
let lastTs = null;
let calls = 0;
let errors = 0;
let primed = false; // first fetch is history: shown, but its jobs not adopted

const clock = (ts) => (ts ? new Date(ts).toLocaleTimeString([], { hour12: false }) : "—");

function ago(ts) {
  if (!ts) return "never";
  const s = Math.max(0, Math.round((Date.now() - new Date(ts).getTime()) / 1000));
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

function renderStatus() {
  const err = errors ? ` · ${errors} error${errors === 1 ? "" : "s"}` : "";
  statusEl.textContent = `${calls} tool call${calls === 1 ? "" : "s"}${err}`
    + ` · last activity ${ago(lastTs)}${paused ? " · paused" : ""}`;
}

const MARK = { completed: "✓", failed: "✗", cancelled: "∅" };

function fmtValue(v) {
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (typeof v !== "number") return String(v);
  if (v !== 0 && (Math.abs(v) >= 1e5 || Math.abs(v) < 1e-3)) return v.toExponential(2);
  return String(Number(v.toPrecision(4)));
}

// How an MCP-submitted job ended: `✓ fem · motor · 9.2 s`, then the first
// result's headline metrics and a caveat badge (tooltip = caveat text).
function makeJobRow(e) {
  const li = document.createElement("li");
  li.className = "agent-row";
  li.dataset.kind = "job";
  li.dataset.ok = e.ok === false ? "false" : "true";
  const caveats = e.caveats ?? [];
  const warn = caveats.some((c) => c.severity === "warn");
  if (warn && e.ok !== false) li.dataset.warn = "true";

  const head = document.createElement("div");
  head.className = "head";
  const time = document.createElement("span");
  time.className = "time";
  time.textContent = clock(e.ts);
  time.title = e.ts;
  const name = document.createElement("span");
  name.className = "tool";
  name.textContent = `${MARK[e.status] ?? "?"} ${e.job_type} · ${e.motor}`;
  head.append(time, name);
  if (caveats.length) {
    const badge = document.createElement("span");
    badge.className = "caveat-badge";
    badge.dataset.warn = String(warn);
    badge.textContent = `⚠ ${caveats.length}`;
    badge.title = caveats.map((c) => `[${c.severity}] ${c.model}: ${c.text} (${c.basis})`).join("\n");
    head.append(badge);
  }
  const dur = document.createElement("span");
  dur.className = "ms";
  dur.textContent = e.s != null ? `${e.s.toFixed(1)} s` : "";
  head.append(dur);
  li.append(head);

  if (e.headline?.length) {
    const m = document.createElement("div");
    m.className = "metrics";
    m.textContent = e.headline
      .map((h) => `${h.key} ${fmtValue(h.value)}${h.unit && h.unit !== "1" ? ` ${h.unit}` : ""}`)
      .join(" · ");
    li.append(m);
  }
  const out = document.createElement("div");
  out.className = e.ok === false ? "outcome error" : "outcome link";
  if (e.ok === false) {
    out.textContent = e.error ?? e.status;
  } else {
    const base = e.total > 1 ? `${e.n_results}/${e.total} results` : `job ${e.job_id}`;
    const cached = !e.n_cached ? "" : e.n_cached === e.total ? " · from cache" : ` · ${e.n_cached} cached`;
    out.textContent = base + cached;
    out.title = "plot this job's results";
    out.addEventListener("click", () => showJob(e.job_id));
  }
  li.append(out);
  return li;
}

function makeRow(e) {
  if (e.kind === "job") return makeJobRow(e);
  const li = document.createElement("li");
  li.className = "agent-row";
  li.dataset.ok = e.ok === false ? "false" : "true";

  const head = document.createElement("div");
  head.className = "head";
  const time = document.createElement("span");
  time.className = "time";
  time.textContent = clock(e.ts);
  time.title = e.ts;
  const name = document.createElement("span");
  name.className = "tool";
  name.textContent = e.kind === "connect" ? "connect" : e.tool;
  const ms = document.createElement("span");
  ms.className = "ms";
  ms.textContent = e.ms != null ? `${e.ms < 10 ? e.ms.toFixed(1) : Math.round(e.ms)} ms` : "";
  head.append(time, name, ms);

  const detail = document.createElement("div");
  detail.className = "detail";
  if (e.kind === "connect") {
    detail.textContent = e.client;
  } else if (e.args && e.args !== "{}") {
    const args = document.createElement("code");
    args.textContent = e.args;
    args.title = e.args;
    detail.append(args);
  }

  const out = document.createElement("div");
  out.className = e.ok === false ? "outcome error" : "outcome";
  out.textContent = e.ok === false ? e.error : (e.summary ?? "");
  if (e.job_id) {
    out.classList.add("link");
    out.title = "plot this job's results";
    out.addEventListener("click", () => showJob(e.job_id));
  }

  li.append(head);
  if (detail.childNodes.length || detail.textContent) li.append(detail);
  if (out.textContent) li.append(out);
  return li;
}

// Same routing as clicking a completed job card; the job list is
// in-memory, so a job from before a server restart is gone.
async function showJob(jobId) {
  let job;
  try {
    job = await app.api.get(`/api/jobs/${jobId}`);
  } catch {
    app.toast(`job ${jobId} is no longer known to the server`);
    return;
  }
  if (job.status !== "completed" || !job.result_ids?.length) {
    app.toast(`job ${jobId} is ${job.status}`);
    return;
  }
  app.activateTab("results");
  app.showJobResults(job);
}

async function adopt(jobId) {
  try {
    app.adoptJob(await app.api.get(`/api/jobs/${jobId}`));
  } catch (err) {
    console.warn("agent job adopt failed", jobId, err);
  }
}

async function poll() {
  if (paused) return;
  let data;
  try {
    data = await app.api.get(`/api/mcp/activity?since=${seq}`);
  } catch (err) {
    statusEl.textContent = "server unreachable";
    return;
  }
  if (boot !== null && data.boot !== boot) {
    // server restarted: the old log is gone, and so is our cursor
    listEl.replaceChildren();
    calls = errors = 0;
    lastTs = null;
    if (data.seq > 0) {
      seq = 0;
      boot = data.boot;
      return poll();
    }
  }
  boot = data.boot;
  seq = data.seq;
  for (const e of data.entries) {
    if (e.kind === "tool") {
      calls += 1;
      if (e.ok === false) errors += 1;
    }
    lastTs = e.ts;
    const li = makeRow(e);
    if (primed) li.classList.add("fresh");
    listEl.prepend(li);
    if (primed && e.tool === "submit_job" && e.job_id) adopt(e.job_id);
  }
  while (listEl.children.length > MAX_ROWS) listEl.lastChild.remove();
  primed = true;
  hintEl.style.display = listEl.children.length ? "none" : "";
  renderStatus();
}

function schedule() {
  clearInterval(timer);
  timer = setInterval(() => {
    if (document.visibilityState === "visible") poll();
    else renderStatus();
  }, POLL_MS);
}

export default {
  name: "mcp_activity",
  label: "Agent Activity",
  accepts: [],

  init(container, appCtx, headerSlot) {
    app = appCtx;

    statusEl = document.createElement("div");
    statusEl.className = "agent-status";

    pauseBtn = document.createElement("button");
    pauseBtn.className = "vs-btn";
    pauseBtn.textContent = "Pause";
    pauseBtn.title = "stop polling; resuming catches up on everything missed";
    pauseBtn.addEventListener("click", () => {
      paused = !paused;
      pauseBtn.textContent = paused ? "Resume" : "Pause";
      if (!paused) poll();
      else renderStatus();
    });
    const clearBtn = document.createElement("button");
    clearBtn.className = "vs-btn";
    clearBtn.textContent = "Clear";
    clearBtn.title = "clear this view (the server keeps its log)";
    clearBtn.addEventListener("click", () => {
      listEl.replaceChildren();
      calls = errors = 0;
      hintEl.style.display = "";
      renderStatus();
    });
    headerSlot?.append(pauseBtn, clearBtn);

    hintEl = document.createElement("div");
    hintEl.className = "vs-empty-hint";
    hintEl.textContent = "No MCP activity yet. Connect an agent to /mcp on this server.";
    listEl = document.createElement("ol");
    listEl.className = "agent-list";

    container.append(statusEl, hintEl, listEl);
    renderStatus();
    poll();
    schedule();
  },

  update() {},

  destroy() {
    clearInterval(timer);
    timer = null;
    boot = null;
    seq = calls = errors = 0;
    paused = primed = false;
    lastTs = null;
    app = listEl = hintEl = statusEl = pauseBtn = null;
  },
};
