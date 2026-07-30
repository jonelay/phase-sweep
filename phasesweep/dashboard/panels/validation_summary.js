// Validation summary panel: agreement matrix of model
// pairs x quantities from GET /api/validation/{motor}. Green <= tol,
// yellow 1-2x tol, red > 2x tol (thresholds via the
// server's ComparisonRow). Clicking a cell jumps the model comparison
// panel to that quantity. The diagnosis line is the
// diagnostic pattern.

let appRef = null;
let diagEl = null;
let modelsEl = null;
let echoEl = null;
let hintEl = null;
let matrixOuter = null;
let matrixWrap = null;
let legendEl = null;
let detailEl = null;
let summary = null;
let refreshTok = 0;

// tone thresholds (tolerances) — stated in the UI, not just code
const LEGEND = [
  ["pass", "≤ tol"], ["warn", "≤ 2× tol"], ["fail", "> 2× tol"], ["skip", "skipped"],
];

function tone(row) {
  if (row.comparison_type === "skipped") return "skip";
  if (row.comparison_type === "bound") return row.passed ? "pass" : "fail";
  if (row.passed) return "pass";
  return row.rel_pct <= 2 * row.tol_pct ? "warn" : "fail";
}

const SEVERITY = { skip: 0, pass: 1, warn: 2, fail: 3 };

function diagnosisTone(d) {
  if (d === "validated" || d === "models agree") return "good";
  if (d === "insufficient data for diagnosis") return "muted";
  return "warn";
}

const fmt = (v) => (typeof v === "number" ? Number(v.toPrecision(5)) : "—"); // NaN served as null

function cellTitle(row) {
  const rel = row.comparison_type === "bound"
    ? `margin ${row.rel_pct.toFixed(1)}%`
    : `Δ ${row.rel_pct.toFixed(1)}% (tol ±${row.tol_pct.toFixed(0)}%)`;
  let t = `${row.model_a}: ${fmt(row.val_a)} vs ${row.model_b}: ${fmt(row.val_b)} — ${rel}`;
  if (row.note) t += ` — ${row.note}`;
  return t;
}

function render() {
  const rows = summary?.rows ?? [];
  diagEl.style.display = summary ? "" : "none";
  modelsEl.style.display = summary?.models?.length ? "" : "none";
  hintEl.style.display = rows.length ? "none" : "";
  matrixOuter.style.display = rows.length ? "" : "none";
  legendEl.style.display = rows.length ? "" : "none";
  detailEl.textContent = ""; // pinned cell detail is stale once the matrix rebuilds
  if (summary) {
    diagEl.textContent = summary.diagnosis;
    diagEl.dataset.tone = diagnosisTone(summary.diagnosis);
    modelsEl.textContent = summary.models?.length
      ? `models with results: ${summary.models.join(", ")}` : "";
  }
  // Params derived from a dataset make agreement with that
  // dataset an echo — flag it, don't let it read as validation
  const derived = Object.entries(summary?.derived_params ?? {});
  echoEl.style.display = derived.length ? "" : "none";
  echoEl.textContent = derived
    .map(([ds, params]) => `⚠ ${params.join(", ")} derived from dataset "${ds}"`
      + " — agreement there echoes the derivation, not independent validation")
    .join("  ·  ");
  if (!rows.length) return;

  // pair columns x quantity rows; worst row wins a contested cell
  const pairs = [];
  const quantities = [];
  const cells = new Map(); // `${pair}|${q}` -> row
  for (const row of rows) {
    const pair = `${row.model_a} ↔ ${row.model_b}`;
    if (!pairs.includes(pair)) pairs.push(pair);
    if (!quantities.includes(row.quantity)) quantities.push(row.quantity);
    const key = `${pair}|${row.quantity}`;
    const prev = cells.get(key);
    if (!prev || SEVERITY[tone(row)] > SEVERITY[tone(prev)]) cells.set(key, row);
  }
  quantities.sort();

  const table = document.createElement("table");
  table.className = "val-matrix";
  const thead = document.createElement("thead");
  const headRow = document.createElement("tr");
  headRow.append(document.createElement("th"));
  for (const p of pairs) {
    const th = document.createElement("th");
    th.textContent = p;
    headRow.append(th);
  }
  thead.append(headRow);

  const tbody = document.createElement("tbody");
  for (const q of quantities) {
    const tr = document.createElement("tr");
    const th = document.createElement("th");
    th.textContent = q;
    tr.append(th);
    for (const p of pairs) {
      const td = document.createElement("td");
      const row = cells.get(`${p}|${q}`);
      if (row) {
        td.dataset.tone = tone(row);
        td.textContent = row.comparison_type === "skipped"
          ? "—" : `${row.rel_pct.toFixed(1)}%`;
        td.title = cellTitle(row);
        td.addEventListener("click", () => {
          detailEl.textContent = `${row.quantity} · ${cellTitle(row)}`;
          appRef.selectQuantity(row.quantity);
        });
      }
      tr.append(td);
    }
    tbody.append(tr);
  }
  table.append(thead, tbody);
  matrixWrap.replaceChildren(table);
  updateScrollHint();
}

// right-edge fade when pair columns run past the visible width
function updateScrollHint() {
  const more = matrixWrap.scrollWidth - matrixWrap.clientWidth - matrixWrap.scrollLeft > 1;
  matrixOuter.classList.toggle("more-right", more);
}

async function refresh() {
  const motor = appRef.getActiveConfig();
  const tok = ++refreshTok;
  let next = null;
  if (motor) {
    try {
      next = await appRef.api.get(`/api/validation/${encodeURIComponent(motor)}`);
    } catch (e) {
      console.warn("validation refresh failed", e);
    }
  }
  if (tok !== refreshTok) return;
  summary = next;
  render();
}

export default {
  name: "validation_summary",
  label: "Validation Summary",
  accepts: [], // owns its data: GET /api/validation for the active config

  init(container, app, headerSlot) {
    appRef = app;

    const refreshBtn = document.createElement("button");
    refreshBtn.textContent = "Refresh";
    refreshBtn.addEventListener("click", refresh);
    headerSlot.append(refreshBtn);

    diagEl = document.createElement("div");
    diagEl.className = "val-diagnosis";

    modelsEl = document.createElement("div");
    modelsEl.className = "panel-caption";

    echoEl = document.createElement("div");
    echoEl.className = "val-echo";
    echoEl.style.display = "none";

    hintEl = document.createElement("div");
    hintEl.className = "empty-hint";
    hintEl.textContent = "No comparisons yet — run 2+ models (or a validate job) for this config.";

    matrixOuter = document.createElement("div");
    matrixOuter.className = "scroll-shadow";
    matrixWrap = document.createElement("div");
    matrixWrap.className = "val-matrix-wrap";
    matrixWrap.addEventListener("scroll", updateScrollHint);
    matrixOuter.append(matrixWrap);

    legendEl = document.createElement("div");
    legendEl.className = "val-legend";
    for (const [toneKey, text] of LEGEND) {
      const item = document.createElement("span");
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.dataset.tone = toneKey;
      item.append(chip, text);
      legendEl.append(item);
    }

    detailEl = document.createElement("div");
    detailEl.className = "panel-caption";

    container.append(diagEl, modelsEl, echoEl, hintEl,
                     matrixOuter, legendEl, detailEl);
    render();

    app.onConfigChange(() => refresh());
    app.onJobComplete(() => refresh());
  },

  update() {},

  destroy() {
    summary = null;
    appRef = diagEl = modelsEl = echoEl = hintEl = matrixWrap = matrixOuter = null;
    legendEl = detailEl = null;
  },
};
