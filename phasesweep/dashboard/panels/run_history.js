// Run history panel: the job list is session-scoped (in-memory
// registry, lost on restart) but the result store persists — this lists the
// most recent stored results so a reload doesn't lose track of what already
// exists. Clicking an OK row routes the full result to the main panels.
// Status/model/motor filters and a "show more" reveal page over the fetched
// window; timestamps are relative, absolute on hover.

import { absTs, relTime } from "./format.js";

const FETCH = 200; // how many recent records we pull...
const PAGE = 30; // ...and how many we reveal at a time

let app = null;
let listEl = null;
let hintEl = null;
let filterBar = null;
let moreBtn = null;
let lastRows = [];
let selectedId = null;
let shown = PAGE;
const filters = { status: "", model: "", motor: "" };
const selects = {};

function distinct(key) {
  return [...new Set(lastRows.map((r) => r[key]).filter(Boolean))].sort();
}

function matches(r) {
  return (!filters.status || r.status === filters.status)
    && (!filters.model || r.model === filters.model)
    && (!filters.motor || r.motor_name === filters.motor);
}

// keep the current selection valid across refreshes; rebuild option lists
function syncFilters() {
  const opts = {
    status: distinct("status"),
    model: distinct("model"),
    motor: distinct("motor_name"),
  };
  for (const [key, sel] of Object.entries(selects)) {
    const values = opts[key];
    if (filters[key] && !values.includes(filters[key])) filters[key] = "";
    sel.replaceChildren();
    sel.add(new Option(key === "motor" ? "all motors"
      : key === "model" ? "all models" : "all statuses", ""));
    for (const v of values) sel.add(new Option(v, v));
    sel.value = filters[key];
  }
}

function render() {
  syncFilters();
  // File order is chronological; show newest first.
  const filtered = [...lastRows].reverse().filter(matches);
  hintEl.style.display = filtered.length ? "none" : "";
  hintEl.textContent = lastRows.length
    ? "No results match the current filters."
    : "No stored results yet.";
  listEl.replaceChildren();
  for (const r of filtered.slice(0, shown)) {
    const li = document.createElement("li");
    li.className = "history-row";
    li.dataset.rid = r.result_id;
    li.classList.toggle("selected", r.result_id === selectedId);
    // linked highlight: row hover spotlights the result
    li.addEventListener("mouseenter",
      () => app.highlightResult(r.result_id, "run_history"));
    li.addEventListener("mouseleave",
      () => app.highlightResult(null, "run_history"));

    const head = document.createElement("div");
    head.className = "head";
    const title = document.createElement("span");
    title.className = "title";
    title.textContent = `${r.motor_name || "?"} · ${r.model}`;
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = r.status;
    badge.dataset.status = r.status;
    head.append(title, badge);

    const sub = document.createElement("div");
    sub.className = "sub";
    const rel = document.createElement("span");
    rel.textContent = relTime(r.timestamp);
    rel.title = absTs(r.timestamp);
    sub.append(rel);
    if (r.source !== "computed") sub.append(document.createTextNode(` · ${r.source}`));

    li.append(head, sub);
    li.title = `${r.motor_name || "?"} · ${r.model} · ${r.status}`
      + ` · ${absTs(r.timestamp)}${r.source !== "computed" ? ` · ${r.source}` : ""}`;
    if (r.status === "OK") {
      li.addEventListener("click", () => {
        selectedId = r.result_id;
        app.routeResult(r.result_id);
        render();
      });
    } else {
      li.classList.add("dead");
    }
    listEl.append(li);
  }
  const hiddenCount = filtered.length - Math.min(shown, filtered.length);
  moreBtn.style.display = hiddenCount > 0 ? "" : "none";
  moreBtn.textContent = `Show ${Math.min(PAGE, hiddenCount)} more (${hiddenCount} hidden)`;
}

async function refresh() {
  try {
    lastRows = await app.api.get(`/api/results?limit=${FETCH}`);
  } catch (e) {
    console.warn("history fetch failed", e);
    return;
  }
  render();
}

export default {
  name: "run_history",
  label: "Run History",
  accepts: [],

  init(container, appCtx) {
    app = appCtx;

    filterBar = document.createElement("div");
    filterBar.className = "history-filters";
    for (const key of ["status", "model", "motor"]) {
      const sel = document.createElement("select");
      sel.addEventListener("change", () => {
        filters[key] = sel.value;
        shown = PAGE;
        render();
      });
      selects[key] = sel;
      filterBar.append(sel);
    }

    hintEl = document.createElement("div");
    hintEl.className = "empty-hint";
    hintEl.textContent = "No stored results yet.";
    listEl = document.createElement("ul");
    listEl.className = "history-list";

    moreBtn = document.createElement("button");
    moreBtn.className = "show-more";
    moreBtn.style.display = "none";
    moreBtn.addEventListener("click", () => {
      shown += PAGE;
      render();
    });

    container.append(filterBar, hintEl, listEl, moreBtn);
    // linked highlight consumer: tint the row for a result announced elsewhere
    app.onHighlight((id, origin) => {
      if (!listEl) return;
      for (const li of listEl.querySelectorAll(".linked")) li.classList.remove("linked");
      if (!id || origin === "run_history") return;
      listEl.querySelector(`[data-rid="${CSS.escape(id)}"]`)?.classList.add("linked");
    });
    refresh();
    app.onJobComplete(refresh);
  },

  destroy() {
    lastRows = [];
    shown = PAGE;
    filters.status = filters.model = filters.motor = "";
    app = listEl = hintEl = filterBar = moreBtn = null;
  },
};
