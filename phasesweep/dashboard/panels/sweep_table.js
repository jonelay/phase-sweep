// Sweep comparison panel: metric-vs-axis plot over the
// accumulated rows plus one table row per result (swept geometry + key
// scalar metrics, sortable columns); clicking a row routes the full result
// to the other panels. Rows arrive progressively via result routing; on
// job completion the FULL id list is fetched through app.onJobComplete
// (the routeFresh path is capped for big sweeps).

import { PLOT_CONFIG, SERIES, axisStyle, baseLayout } from "./plot_theme.js";

// Any computed result belongs in the table; measured/published imports
// don't (they live in the validation panels). Older records predate the
// source field, hence the fallback.
const isComputed = (r) => (r.source ?? "computed") === "computed";

// SweepAxis fields, shown in mm
const GEO_COLS = [
  ["r_outer", (r) => r.config?.motor?.geometry?.r_outer],
  ["L_stk", (r) => r.config?.motor?.L_stk],
  ["r_ag", (r) => r.config?.motor?.geometry?.r_ag],
  ["back_iron", (r) => r.config?.motor?.geometry?.back_iron_thickness],
];

// preferred metric columns, in this order; other scalars join first-seen.
// No cap — every seen metric is a column; the column chooser hides
// what you don't want, persisted per browser.
const PREFERRED = [
  "fundamental", "peak_Br", "thd_pct", "b_iron_max", "tau_mtpa", "tau_stall",
  "t_settle", "i_ss", "speed_droop", "tau_peak", "p_fe", "p_cu_avg",
  "margin", "base_speed_peak",
];

// header units where derivable; ratios (speed_droop,
// k_end) stay unitless. Geometry columns are always [mm].
const METRIC_UNITS = {
  fundamental: "T", peak_Br: "T", b_iron_max: "T", margin: "T",
  thd_pct: "%", sh_pct: "%", backemf_fundamental: "V", flux_linkage_peak: "Wb",
  tau_mtpa: "Nm", tau_stall: "Nm", tau_peak: "Nm",
  t_settle: "s", i_ss: "A", p_fe: "W", p_cu_avg: "W",
  base_speed_peak: "rad/s", base_speed_cont: "rad/s",
  max_speed_peak: "rad/s", max_speed_cont: "rad/s",
  p_max_peak: "W", p_max_cont: "W", u_max: "V",
};

// flux linkage is ~1e-4 Wb — engineering notation, not 0.0003
const ENG_COLS = new Set(["flux_linkage_peak"]);
function eng(v) {
  if (typeof v !== "number") return "—";
  if (v === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(v)) / 3) * 3;
  return e === 0 ? String(Number(v.toPrecision(4))) : `${Number((v / 10 ** e).toPrecision(4))}e${e}`;
}

const HIDDEN_KEY = "ps-sweep-hidden";
const HIDE_EMPTY_KEY = "ps-sweep-hide-empty";

let appRef = null;
let tableWrap = null;
let scrollOuter = null;
let hintEl = null;
let plotEl = null;
let axisSel = null;
let metricSel = null;
let pickerWrap = null; // <details> column chooser in the header slot
let pickerBody = null;
let overflowEl = null; // "N columns hidden" caption
const rows = new Map(); // result_id -> result
const metricCols = []; // insertion-ordered, every metric seen (uncapped)
let hidden = readHidden(); // metric keys the user chose to hide (persisted)
let hideEmpty = localStorage.getItem(HIDE_EMPTY_KEY) !== "0"; // default on
let sortKey = null;
let sortDir = 1;
let selectedId = null;
let axisPicked = false; // user's select choice wins over the auto-pick

function readHidden() {
  try {
    return new Set(JSON.parse(localStorage.getItem(HIDDEN_KEY) || "[]"));
  } catch {
    return new Set();
  }
}

const mm = (v) => (typeof v === "number" ? (v * 1000).toFixed(2) : "—");
const num = (v) => (typeof v === "number" ? Number(v.toPrecision(4)) : "—");

function metricValue(result, key) {
  const v = result.metrics?.[key];
  return typeof v === "number" ? v : null;
}

const visibleMetrics = () => metricCols.filter((k) => !hidden.has(k));

// A row with no value in any *shown* metric column is noise from a model
// whose metrics aren't on screen (e.g. drive_sim under analytical columns);
// hide it unless the user opts back in.
function isEmptyRow(result) {
  return visibleMetrics().every((k) => metricValue(result, k) === null);
}

function displayedRows() {
  const shown = hideEmpty ? sortedRows().filter((r) => !isEmptyRow(r)) : sortedRows();
  return shown;
}

function sortValue(result, key) {
  if (key === "motor") return result.config?.motor?.name ?? "";
  if (key === "model") return result.model;
  const geo = GEO_COLS.find(([name]) => name === key);
  if (geo) return geo[1](result) ?? -Infinity;
  return metricValue(result, key) ?? -Infinity;
}

function sortedRows() {
  const sorted = [...rows.values()];
  if (sortKey !== null) {
    sorted.sort((a, b) => {
      const va = sortValue(a, sortKey);
      const vb = sortValue(b, sortKey);
      return (va < vb ? -1 : va > vb ? 1 : 0) * sortDir;
    });
  }
  return sorted;
}

function addMetricCols(result) {
  const scalars = Object.entries(result.metrics ?? {})
    .filter(([, v]) => typeof v === "number")
    .map(([k]) => k);
  const present = new Set(scalars);
  for (const k of [...PREFERRED.filter((k) => present.has(k)), ...scalars]) {
    if (!metricCols.includes(k)) metricCols.push(k);
  }
}

// per-metric min/max over the displayed rows, for the extreme shading;
// only meaningful when the column has two distinct values.
function columnExtremes(displayed) {
  const ext = new Map(); // key -> {min, max}
  for (const k of visibleMetrics()) {
    let min = Infinity;
    let max = -Infinity;
    for (const r of displayed) {
      const v = metricValue(r, k);
      if (v === null) continue;
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (min < max) ext.set(k, { min, max });
  }
  return ext;
}

// -- column chooser ------------------------------------------------------------

function persistHidden() {
  localStorage.setItem(HIDDEN_KEY, JSON.stringify([...hidden]));
}

function syncPicker() {
  if (!pickerBody) return;
  pickerBody.replaceChildren();
  for (const k of metricCols) {
    const label = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = !hidden.has(k);
    cb.addEventListener("change", () => {
      if (cb.checked) hidden.delete(k);
      else hidden.add(k);
      persistHidden();
      render();
    });
    label.append(cb, document.createTextNode(k));
    pickerBody.append(label);
  }
  const nHidden = metricCols.filter((k) => hidden.has(k)).length;
  overflowEl.textContent = nHidden ? `${nHidden} column${nHidden > 1 ? "s" : ""} hidden` : "";
  overflowEl.style.display = nHidden ? "" : "none";
}

// -- metric-vs-axis plot ------------------------------------------------------

// The swept field is the geometry column with the most distinct values.
function autoAxis() {
  let best = GEO_COLS[0][0];
  let bestN = 0;
  for (const [name, get] of GEO_COLS) {
    const vals = new Set();
    for (const r of rows.values()) {
      const v = get(r);
      if (typeof v === "number") vals.add(v);
    }
    if (vals.size > bestN) {
      bestN = vals.size;
      best = name;
    }
  }
  return best;
}

function syncSelects() {
  if (!axisPicked) axisSel.value = autoAxis();
  const keep = metricSel.value;
  metricSel.replaceChildren();
  for (const k of metricCols) metricSel.add(new Option(k, k));
  metricSel.value = metricCols.includes(keep) ? keep : (metricCols[0] ?? "");
}

function buildTraces(axisKey, metricKey) {
  const geo = GEO_COLS.find(([name]) => name === axisKey);
  if (!geo || !metricKey) return [];
  const groups = new Map(); // "motor · model" -> [[x_mm, y], ...]
  for (const r of rows.values()) {
    const x = geo[1](r);
    const y = metricValue(r, metricKey);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    const key = `${r.config?.motor?.name ?? "?"} · ${r.model}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push([x * 1000, y]);
  }
  return [...groups].map(([name, pts], i) => {
    pts.sort((a, b) => a[0] - b[0]);
    return {
      name,
      x: pts.map((p) => p[0]),
      y: pts.map((p) => p[1]),
      mode: "lines+markers",
      line: { color: SERIES[i % SERIES.length], width: 2 },
      marker: { size: 6 },
    };
  });
}

function renderPlot() {
  syncSelects();
  const traces = buildTraces(axisSel.value, metricSel.value);
  // a curve needs at least two distinct x values somewhere
  const plottable = traces.some((t) => new Set(t.x).size >= 2);
  plotEl.style.display = plottable ? "" : "none";
  if (!plottable) return;
  Plotly.react(plotEl, traces, baseLayout({
    height: 260,
    // traces here are colored by GROUP index, not compare-set slot, so
    // the chips can't identify them — this panel keeps its legend
    // (baseLayout defaults to legend-free for the overlay pools)
    showlegend: true,
    xaxis: axisStyle(`${axisSel.value} [mm]`),
    yaxis: axisStyle(metricSel.value),
    // legend above: at 260 px the below-plot default collides with the
    // x-axis title in the bottom margin
    legend: { orientation: "h", y: 1.02, yanchor: "bottom" },
    margin: { l: 55, r: 12, t: 30, b: 42 },
  }), PLOT_CONFIG);
}

// -- table ---------------------------------------------------------------------

function updateScrollHint() {
  const more = tableWrap.scrollWidth - tableWrap.clientWidth - tableWrap.scrollLeft > 1;
  scrollOuter.classList.toggle("more-right", more);
}

function render() {
  hintEl.style.display = rows.size ? "none" : "";
  scrollOuter.style.display = rows.size ? "" : "none";
  syncPicker();
  if (!rows.size) {
    plotEl.style.display = "none";
    return;
  }

  const metrics = visibleMetrics();
  const cols = ["motor", "model", ...GEO_COLS.map(([name]) => name), ...metrics];
  const table = document.createElement("table");
  table.className = "sweep-table";

  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const c of cols) {
    const th = document.createElement("th");
    const geoCol = GEO_COLS.some(([name]) => name === c);
    const unit = geoCol ? "mm" : METRIC_UNITS[c];
    th.textContent = c + (unit ? ` [${unit}]` : "");
    const arrow = document.createElement("span");
    arrow.className = "sort-arrow";
    arrow.textContent = sortKey === c ? (sortDir > 0 ? "↑" : "↓") : "";
    th.append(arrow);
    th.addEventListener("click", () => {
      sortDir = sortKey === c ? -sortDir : 1;
      sortKey = c;
      render();
    });
    headRow.append(th);
  }
  thead.append(headRow);

  const displayed = displayedRows();
  const extremes = columnExtremes(displayed);

  const tbody = document.createElement("tbody");
  for (const result of displayed) {
    const tr = document.createElement("tr");
    tr.dataset.rid = result.result_id;
    tr.classList.toggle("selected", result.result_id === selectedId);
    tr.append(cell(result.config?.motor?.name ?? "?"));
    tr.append(cell(result.model));
    for (const [, get] of GEO_COLS) tr.append(cell(mm(get(result))));
    for (const k of metrics) {
      const raw = metricValue(result, k);
      const td = cell(ENG_COLS.has(k) ? eng(raw) : num(raw));
      const ext = extremes.get(k);
      if (ext && raw !== null) {
        if (raw === ext.min) td.dataset.extreme = "min";
        else if (raw === ext.max) td.dataset.extreme = "max";
      }
      tr.append(td);
    }
    tr.addEventListener("click", () => {
      selectedId = result.result_id;
      appRef.routeResult(result.result_id);
      render();
    });
    // linked highlight: row hover spotlights the result
    tr.addEventListener("mouseenter",
      () => appRef.highlightResult(result.result_id, "sweep_table"));
    tr.addEventListener("mouseleave",
      () => appRef.highlightResult(null, "sweep_table"));
    tbody.append(tr);
  }
  table.append(thead, tbody);
  tableWrap.replaceChildren(table);
  updateScrollHint();
  renderPlot();
}

function cell(text) {
  const td = document.createElement("td");
  td.textContent = text;
  return td;
}

// -- CSV export ------------------------------------------------------------------

const csvCell = (v) => {
  const s = String(v ?? "");
  return /[",\n]/.test(s) ? `"${s.replaceAll('"', '""')}"` : s;
};

// Raw values (geometry in mm, metrics unrounded), current sort order. Exports
// ALL metric columns, not just the shown ones — the chooser is a view filter.
function exportCsv() {
  const cols = ["motor", "model",
    ...GEO_COLS.map(([name]) => `${name}_mm`), ...metricCols];
  const lines = [cols.map(csvCell).join(",")];
  for (const r of sortedRows()) {
    const geo = GEO_COLS.map(([, get]) => {
      const v = get(r);
      return typeof v === "number" ? v * 1000 : "";
    });
    const mets = metricCols.map((k) => metricValue(r, k) ?? "");
    lines.push([r.config?.motor?.name ?? "?", r.model, ...geo, ...mets]
      .map(csvCell).join(","));
  }
  const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "sweep-comparison.csv";
  a.click();
  URL.revokeObjectURL(a.href);
}

// Caller renders — the job-complete backfill batches many adds per redraw.
function addRow(result) {
  if (!result || result.status !== "OK" || !isComputed(result)) return false;
  if (rows.has(result.result_id)) return false;
  rows.set(result.result_id, result);
  addMetricCols(result);
  return true;
}

export default {
  name: "sweep_table",
  label: "Sweep Comparison",
  accepts: isComputed,

  init(container, app, headerSlot) {
    appRef = app;

    // column chooser lives in the header slot
    pickerWrap = document.createElement("details");
    pickerWrap.className = "col-picker";
    const pickSummary = document.createElement("summary");
    pickSummary.textContent = "Columns";
    pickerBody = document.createElement("div");
    pickerBody.className = "col-picker-body";
    pickerWrap.append(pickSummary, pickerBody);
    headerSlot?.append(pickerWrap);

    const toolbar = document.createElement("div");
    toolbar.className = "panel-toolbar";
    axisSel = document.createElement("select");
    axisSel.title = "plot x-axis (swept geometry field)";
    for (const [name] of GEO_COLS) axisSel.add(new Option(`${name} [mm]`, name));
    axisSel.addEventListener("change", () => {
      axisPicked = true;
      renderPlot();
    });
    metricSel = document.createElement("select");
    metricSel.title = "plot metric";
    metricSel.addEventListener("change", renderPlot);

    const emptyLabel = document.createElement("label");
    emptyLabel.className = "toolbar-check";
    emptyLabel.title = "hide rows whose shown metric columns are all empty";
    const emptyCb = document.createElement("input");
    emptyCb.type = "checkbox";
    emptyCb.checked = hideEmpty;
    emptyCb.addEventListener("change", () => {
      hideEmpty = emptyCb.checked;
      localStorage.setItem(HIDE_EMPTY_KEY, hideEmpty ? "1" : "0");
      render();
    });
    emptyLabel.append(emptyCb, document.createTextNode("hide empty"));

    const exportBtn = document.createElement("button");
    exportBtn.textContent = "Export CSV";
    exportBtn.title = "exports all metric columns, not just the shown ones";
    exportBtn.addEventListener("click", () => {
      if (rows.size) exportCsv();
    });
    const clearBtn = document.createElement("button");
    clearBtn.textContent = "Clear table";
    clearBtn.addEventListener("click", () => {
      rows.clear();
      metricCols.length = 0;
      sortKey = null;
      selectedId = null;
      axisPicked = false;
      render();
    });
    toolbar.append(axisSel, metricSel, emptyLabel, exportBtn, clearBtn);

    overflowEl = document.createElement("div");
    overflowEl.className = "panel-caption sweep-overflow";
    overflowEl.style.display = "none";

    hintEl = document.createElement("div");
    hintEl.className = "empty-hint";
    hintEl.textContent = "No results yet — submit a sweep (or any job) to fill the table.";

    plotEl = document.createElement("div");
    plotEl.style.display = "none";

    scrollOuter = document.createElement("div");
    scrollOuter.className = "scroll-shadow";
    tableWrap = document.createElement("div");
    tableWrap.className = "sweep-table-wrap";
    tableWrap.addEventListener("scroll", updateScrollHint);
    scrollOuter.append(tableWrap);

    container.append(toolbar, overflowEl, hintEl, plotEl, scrollOuter);
    // linked highlight consumer: tint the row for a result announced elsewhere
    app.onHighlight((id, origin) => {
      if (!tableWrap) return;
      for (const tr of tableWrap.querySelectorAll("tr.linked")) tr.classList.remove("linked");
      if (!id || origin === "sweep_table") return;
      tableWrap.querySelector(`tr[data-rid="${CSS.escape(id)}"]`)?.classList.add("linked");
    });
    render();

    app.onJobComplete(async ({ resultIds }) => {
      const pending = resultIds.filter((id) => !rows.has(id));
      const BATCH = 4;
      for (let i = 0; i < pending.length; i += BATCH) {
        const added = await Promise.all(pending.slice(i, i + BATCH).map((id) =>
          app.fetchResult(id).then(addRow)
            .catch((e) => console.warn("table fetch failed", id, e)),
        ));
        if (added.some(Boolean)) render();
      }
    });
  },

  update(result) {
    if (addRow(result)) render();
  },

  retheme: () => renderPlot(),

  destroy() {
    rows.clear();
    metricCols.length = 0;
    axisPicked = false;
    tableWrap = scrollOuter = hintEl = plotEl = axisSel = metricSel = appRef = null;
    pickerWrap = pickerBody = overflowEl = null;
  },
};
