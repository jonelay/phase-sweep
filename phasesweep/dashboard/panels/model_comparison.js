// Model comparison panel: every model run on the active
// motor, one crossval quantity at a time. Computed values draw as wide
// line markers, measured/published as diamonds with error bars from the
// dataset's _uncertainty. Delta vs the reference model in the legend.
// The validation summary panel jumps here via app.selectQuantity.

import { PLOT_CONFIG, SERIES, axisStyle, baseLayout } from "./plot_theme.js";

let appRef = null;
let plotEl = null;
let hintEl = null;
let selectEl = null;
const byModel = new Map(); // model -> full result (latest OK per model)
const modelSlots = new Map(); // model -> SERIES index, stable per panel life
let selectedQuantity = null;
let refreshTok = 0;

function slotFor(model) {
  if (!modelSlots.has(model)) modelSlots.set(model, modelSlots.size % SERIES.length);
  return modelSlots.get(model);
}

function scalarKeys(result) {
  return Object.entries(result.metrics ?? {})
    .filter(([k, v]) => typeof v === "number" && !k.startsWith("_"))
    .map(([k]) => k);
}

// The crossval surface: scalar quantities present in >= 2 results.
function quantityList() {
  const counts = new Map();
  for (const r of byModel.values()) {
    for (const k of scalarKeys(r)) counts.set(k, (counts.get(k) ?? 0) + 1);
  }
  return [...counts].filter(([, n]) => n >= 2).map(([k]) => k).sort();
}

function referenceModel(q) {
  const candidates = [...byModel.values()]
    .filter((r) => r.source === "computed" && typeof r.metrics?.[q] === "number")
    .map((r) => r.model)
    .sort();
  return candidates.includes("analytical") ? "analytical" : candidates[0] ?? null;
}

function tracesFor(q) {
  const ref = referenceModel(q);
  const refVal = ref ? byModel.get(ref).metrics[q] : null;
  const traces = [];
  for (const model of [...byModel.keys()].sort()) {
    const r = byModel.get(model);
    const v = r.metrics?.[q];
    if (typeof v !== "number") continue;
    const color = SERIES[slotFor(model)];
    const measured = r.source !== "computed";
    let name = model + (measured ? ` (${r.source})` : "");
    if (ref && model !== ref && typeof refVal === "number" && Math.abs(refVal) > 1e-12) {
      const pct = ((v - refVal) / Math.abs(refVal)) * 100;
      name += ` · ${pct >= 0 ? "+" : ""}${pct.toFixed(1)}% vs ${ref}`;
    }
    const unc = r.metrics?._uncertainty?.[q];
    traces.push({
      x: [model],
      y: [v],
      mode: "markers",
      marker: measured
        ? { symbol: "diamond", size: 11, color }
        : { symbol: "line-ew-open", size: 24, color, line: { width: 3, color } },
      error_y: typeof unc === "number"
        ? { type: "data", array: [unc], visible: true, color } : undefined,
      name,
      hovertemplate: `%{y:.5g}<extra>${name}</extra>`,
    });
  }
  return traces;
}

function redraw() {
  const quantities = quantityList();
  if (!quantities.includes(selectedQuantity)) selectedQuantity = quantities[0] ?? null;
  selectEl.replaceChildren(...quantities.map((q) => {
    const opt = document.createElement("option");
    opt.value = opt.textContent = q;
    opt.selected = q === selectedQuantity;
    return opt;
  }));
  selectEl.style.display = quantities.length ? "" : "none";

  const traces = selectedQuantity ? tracesFor(selectedQuantity) : [];
  hintEl.style.display = traces.length ? "none" : "";
  plotEl.style.display = traces.length ? "" : "none";
  if (traces.length) {
    const layout = baseLayout({
      xaxis: axisStyle("model"),
      yaxis: axisStyle(selectedQuantity),
      hovermode: "closest",
      // this panel keeps its legend (color=model in a single-config,
      // multi-model context — deliberate split from the Results tab's
      // color=config decision note)
      showlegend: true,
      legend: { orientation: "h", y: -0.25 },
    });
    Plotly.react(plotEl, traces, layout, PLOT_CONFIG);
  }
}

async function refresh() {
  const motor = appRef.getActiveConfig();
  const tok = ++refreshTok;
  const latest = new Map(); // model -> result_id
  if (motor) {
    try {
      const rows = await appRef.listResults(motor);
      for (const row of rows) {
        if (row.status === "OK") latest.set(row.model, row.result_id); // chronological: last wins
      }
    } catch (e) {
      console.warn("comparison refresh failed", e);
    }
  }
  const full = await Promise.all([...latest.values()].map((id) =>
    appRef.fetchResult(id).catch(() => null)));
  if (tok !== refreshTok) return; // a newer refresh superseded this one
  byModel.clear();
  for (const r of full) if (r) byModel.set(r.model, r);
  redraw();
}

export default {
  name: "model_comparison",
  label: "Model Comparison",
  accepts: [], // owns its data: latest result per model for the active config

  init(container, app, headerSlot) {
    appRef = app;

    selectEl = document.createElement("select");
    selectEl.title = "comparison quantity";
    selectEl.addEventListener("change", () => {
      selectedQuantity = selectEl.value;
      redraw();
    });
    const refreshBtn = document.createElement("button");
    refreshBtn.textContent = "Refresh";
    refreshBtn.addEventListener("click", refresh);
    headerSlot.append(selectEl, refreshBtn);

    hintEl = document.createElement("div");
    hintEl.className = "empty-hint";
    hintEl.textContent = "No comparable results — run 2+ models (or import measured data) for this config.";

    plotEl = document.createElement("div");
    plotEl.style.height = "300px";

    container.append(hintEl, plotEl);
    redraw();

    app.onConfigChange(() => refresh());
    app.onJobComplete(() => refresh());
    app.onQuantitySelect((q) => {
      selectedQuantity = q;
      redraw();
      container.closest(".panel-card")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  },

  update() {},

  retheme: () => redraw(),

  destroy() {
    if (plotEl) Plotly.purge(plotEl);
    byModel.clear();
    modelSlots.clear();
    appRef = plotEl = hintEl = selectEl = null;
  },
};
