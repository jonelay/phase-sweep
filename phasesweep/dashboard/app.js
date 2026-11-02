// Layout glue, panel registry, WS connection management.
// Adding a visualization = one file in panels/ + one REGISTRY line (§2.2).
//
// Chrome (tabs, panel cards, rail, status bar, toast), theming (palette,
// light/dark, sequential ramps), the Plotly layout builder, PNG export,
// and the HTTP/WS providers come from view-sweep (served at /view-sweep/).
// This module owns phase-sweep's domain: the config rail and its
// name-space rules, the compare set, result routing, and the keyboard
// layer — bound to view-sweep's `vs:*` events where chrome drives them.

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
import configEditorPanel from "./config-editor.js";
import designSpacePanel from "./panels/design_space.js";
import mcpActivityPanel from "./panels/mcp_activity.js";
import { initTheme, loadPalette, bindThemeToggle, bindSelect } from "/view-sweep/js/theme.js";
import { createPlot, loadPlotSpec } from "/view-sweep/js/plot.js";
import { chip } from "/view-sweep/js/widgets.js";
import {
  mountTabs, mountPanel, toggleMaximize, bindEscapeRestore, defaultIsTextTarget,
  initRailToggle, mountStatusBar, bindConnectionBanner, toast as vsToast, bindHiddenCounter,
} from "/view-sweep/js/chrome.js";
import { exportComposite } from "/view-sweep/js/export.js";
import { FetchProvider } from "/view-sweep/js/providers/fetch.js";
import { WebSocketProvider } from "/view-sweep/js/providers/websocket.js";

// slot: "sidebar" or a tab id from TABS; hero: full-width top row, taller
const REGISTRY = [
  { panel: jobsPanel, slot: "sidebar" },
  { panel: runHistoryPanel, slot: "sidebar" },
  { panel: crossSectionPanel, slot: "results", hero: true },
  { panel: brWaveformPanel, slot: "results" },
  { panel: simWaveformsPanel, slot: "results" },
  { panel: harmonicsPanel, slot: "results" },
  { panel: sweepTablePanel, slot: "sweep" },
  { panel: modelComparisonPanel, slot: "validation" },
  { panel: validationSummaryPanel, slot: "validation" },
  { panel: configEditorPanel, slot: "editor" },
  { panel: designSpacePanel, slot: "agent" },
  { panel: mcpActivityPanel, slot: "agent" },
];

const TABS = [
  { id: "results", label: "Results" },
  { id: "sweep", label: "Sweep" },
  { id: "validation", label: "Validation" },
  { id: "editor", label: "Editor" },
  { id: "agent", label: "Agent" },
];

const MAX_COMPLETE_FETCHES = 12; // cap result fetches on job_complete of a big sweep

// -- theme + plot builder ------------------------------------------------------
//
// The engineering palette is phase-sweep's light/dark set; the stylesheet
// in index.html already applied it, so initTheme only owns the variant,
// the categorical SERIES slots (--vs-series-N, inline) and the |B| ramp.
// Storage keys are phase-sweep's historical ones — index.html's anti-flash
// snippet reads the same theme key. `theme.series` is the LIVE 8-slot
// array: a palette switch re-hues every slot in place, so every
// `SERIES[slot]` read on the next retheme picks up the new hue while the
// slot→config binding (below) is untouched.
const [palette, plotSpec] = await Promise.all([
  loadPalette("/view-sweep/spec/palettes/engineering.json"),
  loadPlotSpec("/view-sweep/spec/plot.json"),
]);
const theme = initTheme({
  palette, storageKey: "ps-theme", categoricalKey: "ps-palette", sequentialKey: "ps-seq",
});
const plot = createPlot({ spec: plotSpec, theme });
const SERIES = theme.series;

// -- providers (_view_sweep_server/protocol.py) -------------------------------------

const http = new FetchProvider();
const api = {
  get: (path) => http.fetch(path),
  post: (path, body) => http.post(path, body),
  put: (path, body) => http.put(path, body),
  del: (path) => http.del(path),
};
const ws = new WebSocketProvider(); // /ws, 1s..30s backoff; emits vs:connection-change, vs:job-*

// completion toast: single element, timer-reset on refire
const toast = (text) => vsToast(text, { duration: 4000 });

// -- app context passed to panels ---------------------------------------------

const activeJobs = new Set(); // non-terminal job ids, re-subscribed on reconnect
const seenResults = new Set();
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

const configListeners = [];
const jobCompleteListeners = [];
const quantityListeners = []; // validation matrix cell -> comparison panel
const highlightListeners = []; // linked highlight bus
let highlightedResult = null;
const resultCache = new Map(); // result_id -> Promise<full result>
let statusBar = null; // view-sweep status bar (set at boot)

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
    host.append(chip({
      label: m.resultIds.size > 1 ? `${name} ·${m.resultIds.size}` : name,
      color: SERIES[m.slot],
      onRemove: () => dropMember(name),
    }));
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
  // view-sweep theme controller (series slots, variant, ramp) and Plotly
  // layout builder — panels read colors through these, never from CSS names
  theme,
  plot,
  fetchResult,
  listResults,
  routeResult,
  getActiveConfig: () => activeConfig,
  // active config's Motor DISPLAY name — what routed results carry
  // (name-space rule above); the nameplate filters on it
  getActiveMotor: () => activeMotor,
  // /api/configs summary of the active config (n_p, topology, motor_name)
  activeConfigInfo: () => configsByKey.get(activeConfig),
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
    statusBar.setJobCount(activeJobs.size);
    ws.subscribeJobs([jobId]);
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
      // the WS provider never saw this completion; mirror its event so
      // the hidden-tab title counter still counts it
      document.dispatchEvent(new CustomEvent("vs:job-complete",
        { detail: { job_id: job.id, result_ids: job.result_ids } }));
    } else {
      app.watchJob(job.id);
    }
    return job;
  },
  // a job submitted elsewhere (an MCP agent): show its card and stream
  // its progress, exactly as if this page had submitted it
  adoptJob(job) {
    jobsPanel.update({ kind: "job", job });
    if (job.status !== "pending" && job.status !== "running") return;
    app.watchJob(job.id);
    // it may have finished before the subscribe landed: re-check once and
    // replay the terminal message the WS would have sent
    api.get(`/api/jobs/${job.id}`).then((j) => {
      if (!activeJobs.has(j.id)) return; // the WS already delivered it
      const msg = j.status === "completed"
        ? { type: "job_complete", job_id: j.id, result_ids: j.result_ids }
        : j.status === "failed" ? { type: "job_failed", job_id: j.id, error: j.error ?? "" }
          : j.status === "cancelled" ? { type: "job_cancelled", job_id: j.id } : null;
      if (msg) handleWsMessage(msg);
    }).catch(() => {});
  },
  reloadConfigs: loadConfigs,
  toast,
  activateTab: (id) => tabs.activate(id),
  configKeyFor: (motorName) => keyByMotorName.get(motorName),
};

// -- completion notification ----------------------------------------------------

function notifyTerminal(msg) {
  if (msg.type === "job_complete") {
    const n = msg.result_ids.length;
    toast(`job complete · ${n} result${n === 1 ? "" : "s"}`);
  } else if (msg.type === "job_failed") {
    toast("job failed — see Jobs panel");
  }
  // cancelled was user-initiated; no notification
}

// -- panel mounting ----------------------------------------

const mainPanels = [];
const panelTabs = new Map(); // panel -> tab id, for tab attention dots
let tabs = null; // view-sweep tab controller (panes, activate, active, markAttention)

const rethemeAll = () => {
  for (const p of mainPanels) p.retheme?.();
};

// -- tab tiling: order-driven grid areas -------------------------

// Derived from registry ORDER, never hand-authored: reordering the
// registry (or a drag-to-reorder splice) must move panels without
// touching this code. Area names are panel names; a short last row
// spans. Breakpoints come from view-sweep's plot spec.
const tabPanes = new Map(); // tab id -> { pane, entries: [{panel, hero}] }
const BREAK = plotSpec.breakpoints ?? { wide: 1400, mid: 1000 };

function columnsForViewport() {
  if (window.matchMedia(`(max-width: ${BREAK.mid}px)`).matches) return 1;
  if (window.matchMedia(`(max-width: ${BREAK.wide}px)`).matches) return 2;
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
  const order = tabPanes.get(tabId).entries.map((e) => e.panel.name);
  localStorage.setItem(panelOrderKey(tabId), JSON.stringify(order));
  document.dispatchEvent(new CustomEvent("vs:panel-reorder", { detail: { tab_id: tabId, order } }));
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

for (const w of [BREAK.mid, BREAK.wide]) {
  window.matchMedia(`(max-width: ${w}px)`).addEventListener("change", () => {
    layoutAllTabs();
    window.dispatchEvent(new Event("resize")); // plots re-fit to new tracks
  });
}

// drag handle starts the drag, the whole card is the drop target —
// dropping splices this tab's panel order. Only react to our own panel
// drags: preventDefault on a text or file drag would break dropping text
// into editor inputs and show a false dragover affordance.
function bindPanelDrag(card, handle, slot, name) {
  handle.draggable = true;
  handle.addEventListener("dragstart", (ev) => {
    ev.dataTransfer.setData("text/panel", name);
    ev.dataTransfer.effectAllowed = "move";
  });
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
    reorderPanels(slot, ev.dataTransfer.getData("text/panel"), name);
  });
}

function mountPanels() {
  tabs = mountTabs({
    tabBar: document.getElementById("main-tabs"),
    paneHost: document.getElementById("tab-panes"),
    tabs: TABS,
    storageKey: "ps-active-tab",
  });
  const slots = {
    sidebar: document.getElementById("sidebar-panels"),
    ...Object.fromEntries(tabs.panes),
  };
  for (const { panel, slot, hero } of REGISTRY) {
    const main = slot !== "sidebar";
    // panel contract: init(body, app, headerSlot?); small
    // per-panel actions render in the header slot beside the window buttons
    const { card, body, headerSlot } = mountPanel({
      id: panel.name,
      title: panel.label,
      startCollapsed: !!panel.startCollapsed,
      storageKey: `ps-collapsed:${panel.name}`,
      draggable: main,
      onReveal: () => panel.retheme?.(),
    });
    if (main) {
      card.style.gridArea = panel.name;
      bindPanelDrag(card, card.querySelector(".vs-drag-handle"), slot, panel.name);
      const maxBtn = document.createElement("button");
      maxBtn.className = "vs-panel-collapse";
      maxBtn.textContent = "⛶";
      maxBtn.title = "maximize (Esc restores)";
      maxBtn.addEventListener("click", () => toggleMaximize(card, slots[slot]));
      headerSlot.after(maxBtn);
    }
    slots[slot].append(card);
    panel.init(body, app, headerSlot);
    if (main) {
      mainPanels.push(panel);
      panelTabs.set(panel, slot);
      if (!tabPanes.has(slot)) tabPanes.set(slot, { pane: slots[slot], entries: [] });
      tabPanes.get(slot).entries.push({ panel, hero: !!hero });
    }
  }
  for (const id of tabPanes.keys()) applyStoredOrder(id);
  layoutAllTabs();
}

// -- result routing --------------------------------------------------------

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
  let result;
  try {
    result = await fetchResult(resultId);
  } catch (e) {
    console.warn("result fetch failed", resultId, e);
    return;
  }
  if (epoch !== undefined && epoch !== routeEpoch) return;
  seenResults.add(resultId);
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
      tabs.markAttention(panelTabs.get(panel));
    }
  }
  document.dispatchEvent(new CustomEvent("vs:result-routed", { detail: { result_id: resultId } }));
}

// -- WebSocket ----------------------------------------------------

// Connection state, backoff, the status-bar dot, and the latching banner
// are view-sweep's (WebSocketProvider + mountStatusBar +
// bindConnectionBanner over vs:connection-change). Here: message
// handling and the reconnect resync.

function routeFresh(resultIds) {
  const fresh = resultIds.filter((id) => !seenResults.has(id));
  if (fresh.length > MAX_COMPLETE_FETCHES) {
    console.warn(`routing last ${MAX_COMPLETE_FETCHES} of ${fresh.length} results`);
  }
  for (const id of fresh.slice(-MAX_COMPLETE_FETCHES)) routeResult(id);
}

// Live-sweep routing cap: job_progress routes every point of a running
// sweep, so a big sweep would pile up ~point-count same-color traces
// (MAX_COMPLETE_FETCHES only guards the job_complete backfill). Track
// live-routed ids per job and batch-evict the oldest past the cap —
// eviction rebuilds survivors from routedIds (the dropMember mechanism;
// trace pools have no per-trace removal), batched at 2× the cap so a
// long sweep rebuilds once per cap-full of points, not per point.
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
  statusBar.setJobCount(activeJobs.size);
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
  ws.subscribeJobs(activeJobs);
  statusBar.setJobCount(activeJobs.size);
  jobsPanel.update({ kind: "jobs", jobs });
}

// -- config selector -----------------------------------------------------------

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

// -- keyboard layer -------------------------------------------------------------

// Bare-key shortcuts must never fire while the user is typing — the
// config editor is a wall of text inputs and j/1 are letters people
// type. Guard, preventDefault only on handled keys, Escape exits modes.
let kbdOverlayEl = null;

function kbdOverlayOpen() {
  return !!kbdOverlayEl && !kbdOverlayEl.hidden;
}

function buildKbdOverlay() {
  const el = document.createElement("div");
  el.className = "vs-kbd-overlay";
  el.hidden = true;
  const card = document.createElement("div");
  card.className = "vs-kbd-card";
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
  // Registered first so that, when the overlay is open, this handler's
  // guard sees it and skips; the overlay listener below then closes it.
  // One Escape closes the overlay, the next restores a maximized card.
  bindEscapeRestore({ guard: kbdOverlayOpen });
  document.addEventListener("keydown", (ev) => {
    if (defaultIsTextTarget(ev.target) || ev.ctrlKey || ev.metaKey || ev.altKey) return;
    const tabIdx = Number(ev.key) - 1;
    if (tabIdx >= 0 && tabIdx < TABS.length) tabs.activate(TABS[tabIdx].id);
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
// status bar is a frame footer: connection ● (vs:connection-change) ·
// job count (driven here) · last result (vs:result-routed)
statusBar = mountStatusBar(document.getElementById("statusbar"));
bindConnectionBanner(document.getElementById("conn-banner"));
bindHiddenCounter("phase-sweep");
initRailToggle({
  layout: document.getElementById("layout"),
  button: document.getElementById("rail-toggle"),
  storageKey: "ps-rail",
});
bindKeys();

// theme + palette selectors: a variant or ramp switch
// re-renders live plots; a categorical switch also re-hues the chip and
// rail dots, which read SERIES[slot]
bindThemeToggle(document.getElementById("theme-toggle"), theme);
bindSelect(document.getElementById("palette-select"),
  theme.categoricalOptions(), theme.categorical, (id) => theme.setCategorical(id));
bindSelect(document.getElementById("seq-select"),
  theme.sequentialOptions(), theme.sequential, (id) => theme.setSequential(id));
document.addEventListener("vs:theme-change", rethemeAll);
document.addEventListener("vs:sequential-change", rethemeAll);
document.addEventListener("vs:palette-change", () => {
  renderChips();
  syncSidebarClasses();
  rethemeAll();
});

// composite PNG export: every visible plot on the active
// tab, header `phase-sweep · tab · config`, provenance burned in
document.getElementById("export-png").addEventListener("click", () => exportComposite({
  container: tabs.panes.get(tabs.active),
  brand: "phase-sweep",
  scope: TABS.find((t) => t.id === tabs.active)?.label ?? "",
  subtitle: document.querySelector("#nameplate .nameplate-name")?.textContent ?? "",
  notify: toast,
}));

loadConfigs().catch((e) => {
  console.warn("initial config load failed", e);
  const ul = document.getElementById("config-list");
  ul.replaceChildren();
  const li = document.createElement("li");
  li.className = "vs-empty-hint";
  li.textContent = `config load failed — ${e.message ?? e}`;
  ul.append(li);
  toast("config load failed — reload to retry");
});
ws.subscribe("*", handleWsMessage);
ws.onReconnect(resync);
ws.connect();
