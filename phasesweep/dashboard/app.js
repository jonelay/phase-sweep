// Layout glue, panel registry, WS connection management.
// Adding a visualization = one file in panels/ + one REGISTRY line.

import jobsPanel from "./panels/jobs.js";
import runHistoryPanel from "./panels/run_history.js";
import brWaveformPanel from "./panels/br_waveform.js";
import harmonicsPanel from "./panels/harmonics.js";
import crossSectionPanel from "./panels/cross_section.js";
import simWaveformsPanel from "./panels/sim_waveforms.js";
import sweepTablePanel from "./panels/sweep_table.js";
import modelComparisonPanel from "./panels/model_comparison.js";
import validationSummaryPanel from "./panels/validation_summary.js";
import nameplatePanel from "./panels/nameplate.js";
import statusbarPanel from "./panels/statusbar.js";
import configEditorPanel from "./config-editor.js";
import {
  SERIES, CATEGORICAL_PALETTES, SEQ_RAMPS, applyPalette, paletteId, applySeq, seqId,
} from "./panels/plot_theme.js";

// slot: "sidebar" or a tab id from TABS; hero: full-width top row, taller
const REGISTRY = [
  { panel: jobsPanel, slot: "sidebar" },
  { panel: runHistoryPanel, slot: "sidebar" },
  { panel: brWaveformPanel, slot: "results", hero: true },
  { panel: harmonicsPanel, slot: "results" },
  { panel: crossSectionPanel, slot: "results" },
  { panel: simWaveformsPanel, slot: "results" },
  { panel: sweepTablePanel, slot: "sweep" },
  { panel: modelComparisonPanel, slot: "validation" },
  { panel: validationSummaryPanel, slot: "validation" },
  { panel: configEditorPanel, slot: "editor" },
];

const TABS = [
  { id: "results", label: "Results" },
  { id: "sweep", label: "Sweep" },
  { id: "validation", label: "Validation" },
  { id: "editor", label: "Editor" },
];

const MAX_COMPLETE_FETCHES = 12; // cap result fetches on job_complete of a big sweep

// -- REST helpers ------------------------------------------------------------

async function request(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new Error(`${r.status}: ${detail}`);
  }
  return r.json();
}

const api = {
  get: (path) => request(path),
  post: (path, body) => request(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }),
  put: (path, body) => request(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  }),
  del: (path) => request(path, { method: "DELETE" }),
};

// -- app context passed to panels ---------------------------------------------

const activeJobs = new Set(); // non-terminal job ids, re-subscribed on reconnect
const seenResults = new Set();
let ws = null;
let activeConfig = null; // sidebar config KEY (API calls take this)
let activeMotor = null; // its Motor display name (compare-set identity)
// Bumped on every plain-click config switch: browse-path fetches capture
// it and discard their response if a switch happened while in flight —
// otherwise a stale routeLatest resurrects the abandoned config as a
// chip. Job/history routes deliberately don't take part (late completions
// SHOULD land).
let routeEpoch = 0;

// Two name spaces: sidebar configs are keyed by config-file name, while
// results carry Motor.name (display). The compare set keys on the
// DISPLAY name — it's what routed results and trace names carry — and
// these maps translate at the sidebar/API boundary.
const motorNameByKey = new Map(); // config key -> motor display name
const keyByMotorName = new Map(); // reverse, for routeLatest on rebuild
const configsByKey = new Map(); // config key -> /api/configs summary (nameplate)

function wsSend(msg) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(msg));
}

const configListeners = [];
const jobCompleteListeners = [];
const quantityListeners = []; // validation matrix cell -> comparison panel
const highlightListeners = []; // linked highlight bus
let highlightedResult = null;
const resultCache = new Map(); // result_id -> Promise<full result>

// -- compare set: membership + config→color -----------------

// Color is assigned per CONFIG at membership time and held until the
// config leaves the set — session-stable, never re-hued,
// so a chip's dot always matches the plotted traces regardless of the
// order results arrive in. Membership registers on ROUTING, not only on
// sidebar clicks: a run-history row or a job finishing after a config
// switch lands a chip too, so chips represent what's plotted. The active
// config is a member (has a color) but is shown by the nameplate strip,
// not a chip.
const compareSet = new Map(); // motor DISPLAY name -> { slot, key, resultIds: Set }
const compareListeners = [];
const freeSlots = [...SERIES.keys()];

function notifyCompare() {
  renderChips();
  syncSidebarClasses();
  for (const fn of compareListeners) fn();
}

function ensureMember(motorName) {
  let m = compareSet.get(motorName);
  if (m) return m;
  if (!freeSlots.length) {
    // config-level cap (8 = SERIES length, effectively unreachable):
    // evict the oldest overlay, never the active config — its chip
    // vanishing is the visible signal
    const oldest = [...compareSet.keys()].find((n) => n !== activeMotor);
    dropMember(oldest);
  }
  m = {
    slot: freeSlots.shift(),
    key: keyByMotorName.get(motorName), // undefined for orphaned results
    resultIds: new Set(), // overlay-pool ids (chip ·N count)
    routedIds: new Set(), // every id routed while a member — rebuild source
  };
  compareSet.set(motorName, m);
  notifyCompare();
  return m;
}

// Removing a config removes all its traces by rebuilding the panels
// from the members that remain — the trace pools have no per-config
// removal. Colors survive: slots stay bound to their configs.
// Survivors rebuild from their own routed ids (resultCache makes this
// cheap): a latest-per-model re-fetch would collapse job replays to one
// trace per model and 404 for orphaned members whose config is gone.
function dropMember(motorName) {
  const m = compareSet.get(motorName);
  if (!m) return;
  compareSet.delete(motorName);
  freeSlots.push(m.slot);
  clearPanels();
  routeEpoch += 1; // in-flight browse fetches for the dropped member discard
  const epoch = routeEpoch;
  for (const other of compareSet.values()) {
    const ids = [...other.routedIds];
    other.resultIds.clear();
    other.routedIds.clear();
    for (const id of ids) routeResult(id, epoch);
  }
  notifyCompare();
}

function resetCompareSet() {
  compareSet.clear();
  freeSlots.length = 0;
  freeSlots.push(...SERIES.keys());
}

// "Clear overlay" header buttons: drop every overlay, keep the active
// config — one rebuild, chips and panels stay in lockstep.
function clearOverlays() {
  let dropped = false;
  for (const [name, m] of [...compareSet]) {
    if (name === activeMotor) continue;
    compareSet.delete(name);
    freeSlots.push(m.slot);
    dropped = true;
  }
  if (!dropped) return;
  clearPanels();
  routeEpoch += 1; // in-flight browse fetches for dropped overlays discard
  const act = compareSet.get(activeMotor);
  const ids = act ? [...act.routedIds] : [];
  act?.resultIds.clear();
  act?.routedIds.clear();
  if (ids.length) {
    const epoch = routeEpoch;
    for (const id of ids) routeResult(id, epoch);
  } else if (activeConfig) {
    routeLatest(activeConfig);
  }
  notifyCompare();
}

function renderChips() {
  const host = document.getElementById("compare-chips");
  host.replaceChildren();
  for (const [name, m] of compareSet) {
    if (name === activeMotor) continue;
    const chip = document.createElement("span");
    chip.className = "compare-chip";
    const dot = document.createElement("span");
    dot.className = "chip-dot";
    dot.style.background = SERIES[m.slot];
    const label = document.createElement("span");
    label.textContent = m.resultIds.size > 1 ? `${name} ·${m.resultIds.size}` : name;
    const x = document.createElement("button");
    x.className = "chip-x";
    x.textContent = "×";
    x.title = `remove ${name} from compare`;
    x.addEventListener("click", () => dropMember(name));
    chip.append(dot, label, x);
    host.append(chip);
  }
}

function syncSidebarClasses() {
  for (const li of document.querySelectorAll("#config-list li")) {
    const key = li.dataset.name;
    const motor = motorNameByKey.get(key) ?? key;
    li.classList.toggle("selected", key === activeConfig);
    li.classList.toggle("overlaid", motor !== activeMotor && compareSet.has(motor));
    // rail dot mirrors the config's compare-set color (chips, traces);
    // hollow when the config isn't plotted
    const m = compareSet.get(motor);
    const dot = li.querySelector(".cfg-dot");
    if (dot) dot.style.background = m ? SERIES[m.slot] : "transparent";
  }
}

const RESULT_CACHE_MAX = 256; // LRU cap: insertion order = recency (re-set on hit)

function fetchResult(resultId) {
  let p = resultCache.get(resultId);
  if (p) {
    resultCache.delete(resultId);
    resultCache.set(resultId, p);
    return p;
  }
  p = api.get(`/api/results/${resultId}`);
  p.catch(() => resultCache.delete(resultId)); // don't cache failures
  resultCache.set(resultId, p);
  if (resultCache.size > RESULT_CACHE_MAX) {
    resultCache.delete(resultCache.keys().next().value);
  }
  return p;
}

// Concurrent-dedup for the results list: a config click triggers both
// routeLatest and the comparison panel's refresh in the same tick — one
// request serves both.
const pendingResultLists = new Map(); // motor -> in-flight list promise

function listResults(motor) {
  let p = pendingResultLists.get(motor);
  if (!p) {
    p = api.get(`/api/results?motor=${encodeURIComponent(motor)}`)
      .finally(() => pendingResultLists.delete(motor));
    pendingResultLists.set(motor, p);
  }
  return p;
}

function notifyJobComplete(jobId, resultIds) {
  for (const fn of jobCompleteListeners) fn({ jobId, resultIds });
}

const app = {
  api,
  fetchResult,
  listResults,
  routeResult,
  getActiveConfig: () => activeConfig,
  // active config's Motor DISPLAY name — what routed results carry
  // (name-space rule above); the nameplate filters on it
  getActiveMotor: () => activeMotor,
  // /api/configs summary of the active config (n_p, topology, motor_name)
  activeConfigInfo: () => configsByKey.get(activeConfig),
  // compare-set view for panels: config→color slot, overlay names,
  // change subscription (chips, badges, trace colors all read this)
  // compare-set view for panels, keyed by Motor DISPLAY name (what
  // results and trace names carry): config→color slot, overlay names,
  // change subscription (chips, badges, trace colors all read this)
  compare: {
    slotFor: (motorName) => compareSet.get(motorName)?.slot,
    overlays: () => [...compareSet.keys()].filter((n) => n !== activeMotor),
    onChange: (fn) => compareListeners.push(fn),
    clearOverlays,
  },
  onConfigChange: (fn) => configListeners.push(fn),
  // fires with the FULL result-id list of a completed job — panels that need
  // every point of a big sweep (sweep table) use this instead of the capped
  // routeFresh path
  onJobComplete: (fn) => jobCompleteListeners.push(fn),
  // cross-panel jump: validation summary cell -> model comparison quantity
  onQuantitySelect: (fn) => quantityListeners.push(fn),
  selectQuantity(q) {
    for (const fn of quantityListeners) fn(q);
  },
  // linked highlight: hover with a result_id anywhere
  // spotlights that result in every subscribed panel — plots fatten the
  // trace, tables tint the row; null clears. origin lets a panel skip
  // its own announcements (a plot hover never restyles the plot being
  // hovered — Plotly's native hover already emphasizes there).
  onHighlight: (fn) => highlightListeners.push(fn),
  highlightResult(id, origin) {
    if (id === highlightedResult) return;
    highlightedResult = id;
    for (const fn of highlightListeners) fn(id, origin);
  },
  watchJob(jobId) {
    activeJobs.add(jobId);
    statusbarPanel.setJobCount(activeJobs.size);
    wsSend({ type: "subscribe", job_ids: [jobId] });
  },
  // clicking a completed job card switches the active config to the job's
  // motor (single-config mode, the compare set collapses) and replays the
  // job's specific results to the plots — the same routing a live
  // completion uses (waveforms capped, full id list to the sweep table),
  // but direct so it re-plots even after the overlay was cleared. If the
  // motor's config is gone it stays an overlay onto the current active one.
  showJobResults(job) {
    const ids = job?.result_ids ?? [];
    if (!ids.length) return;
    const key = job.motor_name;
    if (key && configsByKey.has(key) && key !== activeConfig) activateConfig(key);
    for (const id of ids.slice(-MAX_COMPLETE_FETCHES)) routeResult(id);
    notifyJobComplete(job.id, ids);
  },
  // any panel can submit; the jobs panel shows the row immediately
  async submitJob(body) {
    const job = await api.post("/api/jobs", body);
    jobsPanel.update({ kind: "job", job });
    if (job.status === "completed") {
      // full cache hit: job_complete was broadcast before we could
      // subscribe — route the results from the POST response instead
      routeFresh(job.result_ids);
      notifyJobComplete(job.id, job.result_ids);
      notifyTerminal({ type: "job_complete", result_ids: job.result_ids });
    } else {
      app.watchJob(job.id);
    }
    return job;
  },
  reloadConfigs: loadConfigs,
};

// -- completion notification ----------------------------------------------------

// toast: single element, timer-reset on refire (donor pattern)
let toastTimer = 0;
function toast(text) {
  let el = document.getElementById("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    document.body.append(el);
  }
  el.textContent = text;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 4000);
}

// title badge counts completions while the window is hidden
let hiddenCompletions = 0;
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) {
    hiddenCompletions = 0;
    document.title = "phase-sweep";
  }
});

function notifyTerminal(msg) {
  if (msg.type === "job_complete") {
    const n = msg.result_ids.length;
    toast(`job complete · ${n} result${n === 1 ? "" : "s"}`);
  } else if (msg.type === "job_failed") {
    toast("job failed — see Jobs panel");
  } else {
    return; // cancelled was user-initiated; no notification
  }
  if (document.hidden) {
    hiddenCompletions += 1;
    document.title = `(${hiddenCompletions}) phase-sweep`;
  }
}

// -- panel mounting ----------------------------------------

const mainPanels = [];
const panelTabs = new Map(); // panel -> tab id, for tab attention dots
let tabButtons = null;
let activeTabId = null;
let activateTab = null; // set by mountTabs; the keyboard layer drives it

// -- tab tiling: order-driven grid areas -------------------------

// Derived from registry ORDER, never hand-authored: reordering the
// registry (or a future drag-to-reorder splice) must move panels
// without touching this code. Area names are panel names; a short last row spans.
const tabPanes = new Map(); // tab id -> { pane, entries: [{panel, hero}] }

function columnsForViewport() {
  if (window.matchMedia("(max-width: 1000px)").matches) return 1;
  if (window.matchMedia("(max-width: 1400px)").matches) return 2;
  return 3;
}

function areaRows(entries, cols) {
  const rows = [];
  const rest = [...entries];
  if (rest.length > 1 && rest[0].hero) {
    rows.push(Array(cols).fill(rest.shift().panel.name));
  }
  for (let i = 0; i < rest.length; i += cols) {
    const chunk = rest.slice(i, i + cols);
    const row = [];
    const base = Math.floor(cols / chunk.length);
    let extra = cols % chunk.length;
    for (const e of chunk) {
      const span = base + (extra > 0 ? 1 : 0);
      extra -= 1;
      for (let s = 0; s < span; s += 1) row.push(e.panel.name);
    }
    rows.push(row);
  }
  return rows;
}

function layoutTab(tabId) {
  const t = tabPanes.get(tabId);
  if (!t || !t.entries.length) return;
  const cols = Math.min(columnsForViewport(),
    Math.max(t.entries.filter((e) => !e.hero).length, 1));
  t.pane.style.gridTemplateColumns = `repeat(${cols}, minmax(0, 1fr))`;
  t.pane.style.gridTemplateAreas =
    areaRows(t.entries, cols).map((r) => `"${r.join(" ")}"`).join(" ");
}

function layoutAllTabs() {
  for (const id of tabPanes.keys()) layoutTab(id);
}

// -- drag-to-reorder ---------------------------------------------

// Panels reorder by splicing the per-tab entries array; the area strings
// are derived from that order (areaRows), so a splice + re-layout moves
// panels with no hand-authored area edits. Order persists per browser.
const panelOrderKey = (tabId) => `ps-panel-order:${tabId}`;

function persistOrder(tabId) {
  localStorage.setItem(panelOrderKey(tabId),
    JSON.stringify(tabPanes.get(tabId).entries.map((e) => e.panel.name)));
}

function reorderPanels(tabId, srcName, dstName) {
  const t = tabPanes.get(tabId);
  if (!t || srcName === dstName) return;
  const from = t.entries.findIndex((e) => e.panel.name === srcName);
  const to = t.entries.findIndex((e) => e.panel.name === dstName);
  if (from < 0 || to < 0) return; // cross-tab drop: source isn't in this tab
  const [moved] = t.entries.splice(from, 1);
  t.entries.splice(to, 0, moved);
  persistOrder(tabId);
  layoutTab(tabId);
  window.dispatchEvent(new Event("resize"));
}

// stored order sorts entries on mount; names not in the stored list sink
// to the registry tail so a newly added panel still appears
function applyStoredOrder(tabId) {
  const t = tabPanes.get(tabId);
  let stored;
  try { stored = JSON.parse(localStorage.getItem(panelOrderKey(tabId))); }
  catch { stored = null; }
  if (!Array.isArray(stored)) return;
  const rank = (name) => { const i = stored.indexOf(name); return i < 0 ? 1e9 : i; };
  t.entries.sort((a, b) => rank(a.panel.name) - rank(b.panel.name));
}

for (const w of ["1000px", "1400px"]) {
  window.matchMedia(`(max-width: ${w})`).addEventListener("change", () => {
    layoutAllTabs();
    window.dispatchEvent(new Event("resize")); // plots re-fit to new tracks
  });
}

// -- per-panel maximize ------------------------------------------

function toggleMaximize(card, pane) {
  const on = !card.classList.contains("maximized");
  if (on) {
    const prev = pane.querySelector(".panel-card.maximized");
    if (prev) prev.classList.remove("maximized");
  }
  card.classList.toggle("maximized", on);
  pane.classList.toggle("has-max", on);
  window.dispatchEvent(new Event("resize"));
}

document.addEventListener("keydown", (ev) => {
  if (ev.key !== "Escape" || kbdOverlayOpen()) return; // overlay's Escape wins
  if (isTextTarget(ev.target)) return; // Escape in an input blurs, not restores
  // only the visible pane: a maximized card on a hidden tab keeps its
  // state until that tab is shown again
  const card = document.querySelector(".tab-pane:not([hidden]) .panel-card.maximized");
  if (card) toggleMaximize(card, card.parentElement);
});

// dot on a tab button when results land there while another tab is active
function markTabAttention(tabId) {
  if (!tabId || tabId === activeTabId) return;
  tabButtons?.get(tabId)?.classList.add("attention");
}

function mountTabs() {
  const tabBar = document.getElementById("main-tabs");
  const paneHost = document.getElementById("tab-panes");
  const buttons = new Map();
  const panes = new Map();
  tabButtons = buttons;
  const activate = (id) => {
    for (const { id: tid } of TABS) {
      panes.get(tid).hidden = tid !== id;
      const b = buttons.get(tid);
      b.setAttribute("aria-selected", tid === id);
      // roving tabindex: the active tab is the only tab-stop
      b.tabIndex = tid === id ? 0 : -1;
    }
    activeTabId = id;
    buttons.get(id).classList.remove("attention");
    localStorage.setItem("ps-active-tab", id);
    // plots built while their pane was hidden re-fit on reveal
    // (responsive: true — same trick as collapse-expand)
    window.dispatchEvent(new Event("resize"));
  };
  activateTab = activate;
  for (const [i, { id, label }] of TABS.entries()) {
    const btn = document.createElement("button");
    btn.setAttribute("role", "tab");
    btn.id = `tab-${id}`;
    btn.setAttribute("aria-controls", `pane-${id}`);
    btn.textContent = label;
    btn.title = `${label} — press ${i + 1}`;
    btn.addEventListener("click", () => activate(id));
    tabBar.append(btn);
    buttons.set(id, btn);
    const pane = document.createElement("div");
    pane.className = "tab-pane";
    pane.id = `pane-${id}`;
    pane.setAttribute("role", "tabpanel");
    pane.setAttribute("aria-labelledby", `tab-${id}`);
    pane.hidden = true;
    paneHost.append(pane);
    panes.set(id, pane);
  }
  // arrows move within the tab bar, selection follows focus
  // (roving-tabindex tablist)
  tabBar.addEventListener("keydown", (ev) => {
    const order = TABS.map((t) => t.id);
    let i = order.indexOf(activeTabId);
    if (ev.key === "ArrowRight") i = (i + 1) % order.length;
    else if (ev.key === "ArrowLeft") i = (i + order.length - 1) % order.length;
    else if (ev.key === "Home") i = 0;
    else if (ev.key === "End") i = order.length - 1;
    else return;
    ev.preventDefault();
    activate(order[i]);
    buttons.get(order[i]).focus();
  });
  const stored = localStorage.getItem("ps-active-tab");
  activate(panes.has(stored) ? stored : TABS[0].id);
  return panes;
}

function mountPanels() {
  const panes = mountTabs();
  const slots = {
    sidebar: document.getElementById("sidebar-panels"),
    ...Object.fromEntries(panes),
  };
  for (const { panel, slot, hero } of REGISTRY) {
    const card = document.createElement("section");
    card.className = "panel-card";
    const header = document.createElement("header");
    const title = document.createElement("h2");
    title.textContent = panel.label;
    // small per-panel actions render here, beside the window buttons
    // (panel contract: init(body, app, headerSlot?))
    const headerSlot = document.createElement("div");
    headerSlot.className = "header-slot";
    const btn = document.createElement("button");
    btn.className = "collapse-btn";
    const storeKey = `ps-collapsed:${panel.name}`;
    const setCollapsed = (collapsed) => {
      const wasCollapsed = card.classList.contains("collapsed");
      card.classList.toggle("collapsed", collapsed);
      btn.textContent = collapsed ? "+" : "−";
      btn.title = collapsed ? "expand" : "collapse";
      // plotly panels re-fit on expand (responsive: true listens for this)
      if (wasCollapsed && !collapsed) window.dispatchEvent(new Event("resize"));
    };
    setCollapsed(localStorage.getItem(storeKey) === "1"
      || (localStorage.getItem(storeKey) === null && !!panel.startCollapsed));
    btn.addEventListener("click", () => {
      const collapsed = !card.classList.contains("collapsed");
      setCollapsed(collapsed);
      localStorage.setItem(storeKey, collapsed ? "1" : "0");
    });
    header.append(title, headerSlot);
    const body = document.createElement("div");
    body.className = "panel-body";
    if (slot !== "sidebar") {
      card.style.gridArea = panel.name;
      // drag-to-reorder (H2): handle starts the drag, the whole card
      // is the drop target — dropping splices this tab's panel order
      const handle = document.createElement("span");
      handle.className = "drag-handle";
      handle.textContent = "⠿";
      handle.title = "drag to reorder";
      handle.draggable = true;
      handle.addEventListener("dragstart", (ev) => {
        ev.dataTransfer.setData("text/panel", panel.name);
        ev.dataTransfer.effectAllowed = "move";
      });
      header.prepend(handle);
      // only react to our own panel drags — preventDefault on a text or
      // file drag would break dropping text into editor inputs and show
      // a false dragover affordance
      card.addEventListener("dragover", (ev) => {
        if (!ev.dataTransfer.types.includes("text/panel")) return;
        ev.preventDefault();
        card.classList.add("dragover");
      });
      card.addEventListener("dragleave", () => card.classList.remove("dragover"));
      card.addEventListener("drop", (ev) => {
        if (!ev.dataTransfer.types.includes("text/panel")) return;
        ev.preventDefault();
        card.classList.remove("dragover");
        reorderPanels(slot, ev.dataTransfer.getData("text/panel"), panel.name);
      });
      const maxBtn = document.createElement("button");
      maxBtn.className = "collapse-btn";
      maxBtn.textContent = "⛶";
      maxBtn.title = "maximize (Esc restores)";
      maxBtn.addEventListener("click", () => toggleMaximize(card, slots[slot]));
      header.append(maxBtn);
    }
    header.append(btn);
    card.append(header, body);
    slots[slot].append(card);
    panel.init(body, app, headerSlot);
    if (slot !== "sidebar") {
      mainPanels.push(panel);
      panelTabs.set(panel, slot);
      if (!tabPanes.has(slot)) tabPanes.set(slot, { pane: slots[slot], entries: [] });
      tabPanes.get(slot).entries.push({ panel, hero: !!hero });
    }
  }
  for (const id of tabPanes.keys()) applyStoredOrder(id);
  layoutAllTabs();
}

// -- theme --------------------------------------------------------

// index.html sets data-theme from localStorage before the stylesheet loads;
// here we only label the button and re-render plots on a switch (panels
// read the --chart-* vars at layout-build time via their retheme() hook).
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const btn = document.getElementById("theme-toggle");
  btn.textContent = theme === "dark" ? "Dark" : "Light";
  btn.title = theme === "dark" ? "switch to light theme" : "switch to dark theme";
  for (const p of mainPanels) p.retheme?.();
}

document.getElementById("theme-toggle").addEventListener("click", () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("ps-theme", next);
  applyTheme(next);
});

// -- color palette selectors --------------------------------------

// Two persisted axes orthogonal to light/dark: the categorical SERIES
// palette and the |B| ramp. A palette switch re-hues every slot at once —
// applyPalette mutates SERIES in place, so recoloring the chip/rail dots
// (which read SERIES[slot]) and rethemeing every panel is all it takes; the
// slot→config binding is untouched.
function populateSelect(el, entries, current) {
  for (const [id, { label }] of Object.entries(entries)) {
    const opt = document.createElement("option");
    opt.value = id;
    opt.textContent = label;
    el.append(opt);
  }
  el.value = current;
}

const paletteSel = document.getElementById("palette-select");
populateSelect(paletteSel, CATEGORICAL_PALETTES, paletteId());
paletteSel.addEventListener("change", () => {
  applyPalette(paletteSel.value);
  renderChips();
  syncSidebarClasses();
  for (const p of mainPanels) p.retheme?.();
});

const seqSel = document.getElementById("seq-select");
populateSelect(seqSel, SEQ_RAMPS, seqId());
seqSel.addEventListener("change", () => {
  applySeq(seqSel.value);
  for (const p of mainPanels) p.retheme?.();
});

// -- composite PNG export ------------------------------------------

// Report figure: every visible plot on the active tab stacked onto one
// canvas with a title + timestamp header and each panel's provenance
// stamp burned in, so the export stays attributable.
function loadImage(src) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = rej;
    im.src = src;
  });
}

async function exportComposite() {
  const pane = document.getElementById(`pane-${activeTabId}`);
  if (!pane) return;
  const items = [];
  for (const card of pane.querySelectorAll(".panel-card")) {
    if (card.classList.contains("collapsed")) continue;
    const el = card.querySelector(".js-plotly-plot");
    if (!el || el.style.display === "none" || !el.clientWidth) continue;
    items.push({
      el,
      title: card.querySelector("h2")?.textContent ?? "",
      prov: card.querySelector(".provenance")?.textContent ?? "",
    });
  }
  if (!items.length) { toast("no plots to export on this tab"); return; }
  toast("rendering PNG…");
  const scale = 2;
  let imgs;
  try {
    const urls = await Promise.all(items.map((it) => Plotly.toImage(it.el, {
      format: "png", scale, width: it.el.clientWidth, height: it.el.clientHeight })));
    imgs = await Promise.all(urls.map(loadImage));
  } catch (e) {
    console.warn("export failed", e);
    toast("export failed");
    return;
  }
  const css = (v) => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  const ink = css("--text-primary");
  const muted = css("--text-muted");
  const pad = 16 * scale;
  const headH = 48 * scale;
  const titleH = 18 * scale;
  const provH = 15 * scale;
  const gap = 12 * scale;
  const canvas = document.createElement("canvas");
  canvas.width = Math.max(...imgs.map((im) => im.width)) + 2 * pad;
  canvas.height = headH + imgs.reduce((s, im, i) =>
    s + titleH + (items[i].prov ? provH : 0) + im.height + gap, 0) + pad;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = getComputedStyle(document.body).backgroundColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.textBaseline = "top";
  const tab = TABS.find((t) => t.id === activeTabId)?.label ?? "";
  const name = document.querySelector("#nameplate .nameplate-name")?.textContent ?? "";
  const stamp = new Date().toISOString().slice(0, 19).replace("T", " ");
  ctx.fillStyle = ink;
  ctx.font = `600 ${14 * scale}px system-ui, sans-serif`;
  ctx.fillText(`phase-sweep · ${tab}${name ? ` · ${name}` : ""}`, pad, pad);
  ctx.fillStyle = muted;
  ctx.font = `${11 * scale}px system-ui, sans-serif`;
  ctx.fillText(stamp, pad, pad + 21 * scale);
  let y = headH;
  items.forEach((it, i) => {
    ctx.fillStyle = ink;
    ctx.font = `600 ${12 * scale}px system-ui, sans-serif`;
    ctx.fillText(it.title, pad, y);
    y += titleH;
    if (it.prov) {
      ctx.fillStyle = muted;
      ctx.font = `${10 * scale}px ui-monospace, monospace`;
      ctx.fillText(it.prov, pad, y);
      y += provH;
    }
    ctx.drawImage(imgs[i], pad, y);
    y += imgs[i].height + gap;
  });
  canvas.toBlob((blob) => {
    if (!blob) { toast("export failed"); return; }
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `phase-sweep_${(tab || "view").toLowerCase()}_${stamp.replace(/[: ]/g, "-")}.png`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    toast("PNG downloaded");
  }, "image/png");
}

document.getElementById("export-png").addEventListener("click", exportComposite);

// -- result routing ---------------------------------------------

// accepts: a model-name list, or a predicate over {model, source} for
// panels whose interest doesn't reduce to a name list (e.g. "any computed").
function panelAccepts(panel, r) {
  return typeof panel.accepts === "function"
    ? panel.accepts(r) : panel.accepts.includes(r.model);
}

// `epoch` (browse path only): pass the routeEpoch captured before the
// triggering fetch — the route is discarded if the user switched configs
// while it was in flight. Job/history callers omit it (late completions
// deliberately land).
async function routeResult(resultId, epoch) {
  seenResults.add(resultId);
  let result;
  try {
    result = await fetchResult(resultId);
  } catch (e) {
    console.warn("result fetch failed", resultId, e);
    return;
  }
  if (epoch !== undefined && epoch !== routeEpoch) return;
  // Foreign-route auto-registration: membership (and the
  // config's color) is established BEFORE panels draw, so overlay-pool
  // panels always find a slot. Only overlay-pool results register — a
  // drive_sim-only route shows on its panel without minting a chip.
  const motorName = result.config?.motor?.name;
  if (motorName && mainPanels.some((p) => p.overlayPool && panelAccepts(p, result))) {
    const m = ensureMember(motorName);
    if (!m.resultIds.has(resultId)) {
      m.resultIds.add(resultId);
      notifyCompare(); // chip trace-count refresh
    }
  }
  // Existing members log every routed id (drive_sim included) so a
  // dropMember/clearOverlays rebuild can restore exactly what was shown.
  compareSet.get(motorName)?.routedIds.add(resultId);
  for (const panel of mainPanels) {
    if (panelAccepts(panel, result)) {
      panel.update(result);
      markTabAttention(panelTabs.get(panel));
    }
  }
  statusbarPanel.stampResult();
}

// -- WebSocket ----------------------------------------------------

let backoffMs = 1000;
let everConnected = false; // banner only after a real drop, not at boot

// The ● indicator lives in the status bar. The banner
// latches: shown on a real drop, held through the connecting attempts
// (data may still be stale), cleared only on reconnect — the amber
// "connecting" dot alone doesn't warrant the banner.
function setWsState(state) {
  if (state === "connected") everConnected = true;
  statusbarPanel.setWsState(state);
  const banner = document.getElementById("conn-banner");
  if (state === "disconnected" && everConnected) banner.hidden = false;
  else if (state === "connected") banner.hidden = true;
}

function routeFresh(resultIds) {
  const fresh = resultIds.filter((id) => !seenResults.has(id));
  if (fresh.length > MAX_COMPLETE_FETCHES) {
    console.warn(`routing last ${MAX_COMPLETE_FETCHES} of ${fresh.length} results`);
  }
  for (const id of fresh.slice(-MAX_COMPLETE_FETCHES)) routeResult(id);
}

// Live-sweep routing cap: job_progress routes every point of a running
// sweep, so a big sweep would pile up ~point-count same-color traces
// (the old 8-trace overlay cap does not cover this path;
// MAX_COMPLETE_FETCHES only guards the job_complete backfill). Track
// live-routed ids per job and
// batch-evict the oldest past the cap — eviction rebuilds survivors
// from routedIds (the dropMember mechanism; trace pools have no
// per-trace removal), batched at 2× the cap so a long sweep rebuilds
// once per cap-full of points, not per point.
const liveRouted = new Map(); // job_id -> ordered live-routed result ids

function pruneLiveTraces(jobId) {
  const q = liveRouted.get(jobId);
  if (!q || q.length < MAX_COMPLETE_FETCHES * 2) return;
  const evict = new Set(q.splice(0, q.length - MAX_COMPLETE_FETCHES));
  let hit = false;
  for (const m of compareSet.values()) {
    for (const id of evict) {
      if (m.resultIds.delete(id)) hit = true;
      if (m.routedIds.delete(id)) hit = true;
    }
  }
  if (!hit) return;
  clearPanels();
  const epoch = routeEpoch;
  for (const m of compareSet.values()) {
    const ids = [...m.routedIds];
    m.resultIds.clear();
    m.routedIds.clear();
    for (const id of ids) routeResult(id, epoch);
  }
  notifyCompare();
}

function handleWsMessage(msg) {
  jobsPanel.update({ kind: "ws", msg });
  if (msg.type === "job_progress") {
    if (msg.latest_result_id) {
      let q = liveRouted.get(msg.job_id);
      if (!q) {
        q = [];
        liveRouted.set(msg.job_id, q);
      }
      q.push(msg.latest_result_id);
      routeResult(msg.latest_result_id);
      pruneLiveTraces(msg.job_id);
    }
    return;
  }
  liveRouted.delete(msg.job_id);
  activeJobs.delete(msg.job_id); // terminal: complete / failed / cancelled
  statusbarPanel.setJobCount(activeJobs.size);
  notifyTerminal(msg);
  if (msg.type === "job_complete") {
    routeFresh(msg.result_ids);
    notifyJobComplete(msg.job_id, msg.result_ids);
  }
}

async function resync() {
  // Reconnect contract: re-fetch via REST, re-subscribe.
  let jobs;
  try {
    jobs = await api.get("/api/jobs");
  } catch (e) {
    console.warn("resync failed", e);
    return;
  }
  activeJobs.clear();
  for (const j of jobs) {
    if (j.status === "pending" || j.status === "running") activeJobs.add(j.id);
  }
  wsSend({ type: "subscribe", job_ids: [...activeJobs] });
  statusbarPanel.setJobCount(activeJobs.size);
  jobsPanel.update({ kind: "jobs", jobs });
}

function connectWs() {
  setWsState("connecting");
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen = () => {
    backoffMs = 1000;
    setWsState("connected");
    resync();
  };
  ws.onmessage = (ev) => handleWsMessage(JSON.parse(ev.data));
  ws.onclose = () => {
    setWsState("disconnected");
    setTimeout(connectWs, backoffMs);
    backoffMs = Math.min(backoffMs * 2, 30000); // 1s, 2s, 4s, ..., max 30s
  };
}

// -- config selector -----------------------------------------------

// Show what the store already has: latest OK result per model any panel accepts.
async function routeLatest(name) {
  const epoch = routeEpoch;
  let rows;
  try {
    rows = await listResults(name);
  } catch (e) {
    console.warn("results fetch failed", e);
    return;
  }
  if (epoch !== routeEpoch) return; // config switched while in flight
  const latestPerModel = new Map();
  for (const row of rows) {
    if (row.status === "OK" && mainPanels.some((p) => panelAccepts(p, row))) {
      latestPerModel.set(row.model, row.result_id);
    }
  }
  for (const id of latestPerModel.values()) routeResult(id, epoch);
}

function clearPanels() {
  for (const p of mainPanels) p.clear?.();
}

// Plain click = single-config mode: the compare set
// collapses to the new active config — the chip row empties with it.
// Comparing is explicit via ctrl-click. Panels that accumulate
// cross-config on purpose (sweep table) don't implement clear().
// Switch the active config + collapse the compare set, but route nothing —
// callers that want the store's latest call setActiveConfig; the job-card
// path routes the job's specific results itself (showJobResults).
function activateConfig(name) {
  routeEpoch += 1; // in-flight browse fetches for the old config discard
  activeConfig = name;
  activeMotor = motorNameByKey.get(name) ?? name;
  resetCompareSet();
  clearPanels();
  ensureMember(activeMotor); // notifies: chips clear, sidebar classes sync
  for (const fn of configListeners) fn(name);
}

function setActiveConfig(name) {
  activateConfig(name);
  routeLatest(name);
}

// Ctrl/cmd-click toggles a config in the compare set;
// the chip row is the visible/removable representation.
function toggleOverlayConfig(name) {
  const motor = motorNameByKey.get(name) ?? name;
  if (motor === activeMotor) {
    // A save-as copy shares the original's [motor] name, and results
    // only carry that name — the two configs are indistinguishable to
    // the compare set. Say so instead of silently doing nothing.
    if (name !== activeConfig) {
      toast(`"${name}" shares Motor name "${motor}" with the active config`
        + " — give it a distinct [motor] name to compare");
    }
    return;
  }
  if (compareSet.has(motor)) {
    dropMember(motor);
    return;
  }
  ensureMember(motor);
  routeLatest(name);
}

async function loadConfigs(selectName) {
  const configs = await api.get("/api/configs");
  motorNameByKey.clear();
  keyByMotorName.clear();
  configsByKey.clear();
  for (const c of configs) {
    motorNameByKey.set(c.name, c.motor_name ?? c.name);
    keyByMotorName.set(c.motor_name ?? c.name, c.name);
    configsByKey.set(c.name, c);
  }
  const ul = document.getElementById("config-list");
  ul.replaceChildren();
  for (const c of configs) {
    const li = document.createElement("li");
    li.dataset.name = c.name;
    // one-line rail row: key + meta, ✎ marks user configs.
    // Identity rule: config KEY here (selection identity); Motor display
    // names appear where results carry them (chips, captions, history).
    const meta = [c.n_p != null ? `${c.n_p}p` : null, c.topology]
      .filter(Boolean).join(" · ");
    li.innerHTML = "<span class=\"cfg-dot\"></span><span class=\"cfg-key\"></span>"
      + `<span class="meta">${meta}${c.editable ? " ✎" : ""}</span>`;
    li.querySelector(".cfg-key").textContent = c.name;
    if (c.editable) li.title = `${c.name} — user config (editable)`;
    li.addEventListener("click", (ev) => {
      if (ev.ctrlKey || ev.metaKey) toggleOverlayConfig(c.name);
      else setActiveConfig(c.name);
    });
    ul.append(li);
  }
  const names = configs.map((c) => c.name);
  const pick = selectName && names.includes(selectName) ? selectName
    : names.includes(activeConfig) ? activeConfig
    : names[0];
  if (pick) setActiveConfig(pick);
}

// -- sidebar rail collapse -----------------------------------------

// Collapse-to-edge toggle, persisted per browser. Column count doesn't
// change (breakpoints are viewport matchMedia — accepted), but plots
// re-fit to the wider tracks via the resize dispatch.
function initRailToggle() {
  const layout = document.getElementById("layout");
  const btn = document.getElementById("rail-toggle");
  const apply = (collapsed) => {
    layout.classList.toggle("rail-collapsed", collapsed);
    btn.textContent = collapsed ? "»" : "«";
    btn.title = collapsed ? "expand sidebar" : "collapse sidebar";
    window.dispatchEvent(new Event("resize"));
  };
  apply(localStorage.getItem("ps-rail") === "1");
  btn.addEventListener("click", () => {
    const collapsed = !layout.classList.contains("rail-collapsed");
    localStorage.setItem("ps-rail", collapsed ? "1" : "0");
    apply(collapsed);
  });
}

// -- keyboard layer -----------------------------------------------

// Bare-key shortcuts must never fire while the user is typing — the
// config editor is a wall of text inputs and j/1 are letters people
// type. Donor bindKeys pattern: guard, preventDefault only on handled
// keys, Escape exits modes.
function isTextTarget(t) {
  return t instanceof Element
    && (t.matches("input, textarea, select") || t.isContentEditable);
}

let kbdOverlayEl = null;

function kbdOverlayOpen() {
  return !!kbdOverlayEl && !kbdOverlayEl.hidden;
}

function buildKbdOverlay() {
  const el = document.createElement("div");
  el.id = "kbd-overlay";
  el.hidden = true;
  const card = document.createElement("div");
  card.className = "kbd-card";
  const title = document.createElement("h2");
  title.textContent = "Keyboard shortcuts";
  const dl = document.createElement("dl");
  const SHORTCUTS = [
    ["1–4", "switch tab"],
    ["j / k", "next / previous config"],
    ["← →", "move between tabs (tab bar focused)"],
    ["Esc", "close overlay · restore maximized panel"],
    ["?", "toggle this overlay"],
  ];
  for (const [key, what] of SHORTCUTS) {
    const dt = document.createElement("dt");
    const kbd = document.createElement("kbd");
    kbd.textContent = key;
    dt.append(kbd);
    const dd = document.createElement("dd");
    dd.textContent = what;
    dl.append(dt, dd);
  }
  card.append(title, dl);
  el.append(card);
  el.addEventListener("click", () => toggleKbdOverlay(false));
  document.body.append(el);
  return el;
}

function toggleKbdOverlay(show = !kbdOverlayOpen()) {
  if (!kbdOverlayEl) kbdOverlayEl = buildKbdOverlay();
  kbdOverlayEl.hidden = !show;
}

// j/k step through the sidebar rail in list order, wrapping at the ends
function stepConfig(delta) {
  const keys = [...document.querySelectorAll("#config-list li[data-name]")]
    .map((li) => li.dataset.name);
  if (!keys.length) return;
  const i = keys.indexOf(activeConfig);
  setActiveConfig(keys[(i + delta + keys.length) % keys.length]);
}

function bindKeys() {
  document.addEventListener("keydown", (ev) => {
    if (isTextTarget(ev.target) || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const tabIdx = Number(ev.key) - 1;
    if (tabIdx >= 0 && tabIdx < TABS.length) activateTab(TABS[tabIdx].id);
    else if (ev.key === "j") stepConfig(1);
    else if (ev.key === "k") stepConfig(-1);
    else if (ev.key === "?") toggleKbdOverlay();
    else if (ev.key === "Escape" && kbdOverlayOpen()) toggleKbdOverlay(false);
    else return;
    ev.preventDefault();
  });
}

// -- boot -----------------------------------------------------------------------

mountPanels();
// nameplate mounts outside the card grid (its own row under the topbar)
// but joins mainPanels so routing feeds it and clearPanels resets it
nameplatePanel.init(document.getElementById("nameplate"), app);
mainPanels.push(nameplatePanel);
// status bar is a frame footer, driven directly (not routed)
statusbarPanel.init(document.getElementById("statusbar"));
initRailToggle();
bindKeys();
applyTheme(document.documentElement.dataset.theme);
loadConfigs().catch((e) => {
  console.warn("initial config load failed", e);
  const ul = document.getElementById("config-list");
  ul.replaceChildren();
  const li = document.createElement("li");
  li.className = "empty-hint";
  li.textContent = `config load failed — ${e.message ?? e}`;
  ul.append(li);
  toast("config load failed — reload to retry");
});
connectWs();
