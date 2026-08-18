// B_r(theta) waveform panel: plotly line plot, overlay mode.
// Color = config (compare-set slot from app.js), dash = model
// (MODEL_DASH); identity lives in chips + hover, no legend.

import { MODEL_DASH, SERIES, axisStyle, baseLayout, makeOverlay, redrawOverlay, traceName } from "./plot_theme.js";
import { provenanceText } from "./widgets.js";
import { makeCursors, cursorToggleButton } from "./cursors.js";

// built per redraw so a theme switch picks up the current --chart-* colors
const layout = () => baseLayout({
  xaxis: axisStyle("θ [deg]"),
  yaxis: axisStyle("B_r [T]"),
});

let plotEl = null;
let hintEl = null;
let provEl = null;
let appCtx = null;
let hoverBound = false;
let cursors = null;

const overlay = makeOverlay();

function redraw() {
  // re-read the slot's hue so a palette switch (SERIES mutated in place,
  // plot_theme.js) re-hues traces that were colored at route time
  for (const t of overlay.traces) t.line.color = SERIES[t._slot];
  const l = layout();
  l.shapes = cursors?.shapes() ?? []; // [] when off, so Plotly clears them
  const p = redrawOverlay(plotEl, hintEl, overlay.traces, l);
  cursors?.refresh(); // rebuilds when on (tracks current traces), clears when off
  return p;
}

// linked highlight: fatten the announced trace, dim the
// rest; null restores. Transient hover state — any redraw resets it.
function applyHighlight(id) {
  if (!plotEl?.data?.length) return;
  Plotly.restyle(plotEl, {
    "line.width": overlay.traces.map((t) => (t._key === id ? 3.5 : 2)),
    opacity: overlay.traces.map((t) => (id && t._key !== id ? 0.35 : 1)),
  });
}

// plot hover announces the trace's result id; bound once the div is a
// live plotly plot (first non-empty redraw)
function bindHover() {
  if (hoverBound || typeof plotEl?.on !== "function") return;
  hoverBound = true;
  plotEl.on("plotly_hover", (ev) => {
    const key = overlay.traces[ev.points?.[0]?.curveNumber]?._key;
    if (key) appCtx.highlightResult(key, "br_waveform");
  });
  plotEl.on("plotly_unhover", () => appCtx.highlightResult(null, "br_waveform"));
  plotEl.on("plotly_relayout", (ev) => cursors.handleRelayout(ev));
}

// Overlay pool shows many results at once — the stamp names the latest
// routed one (per-trace identity stays in hover), with the trace count.
function stampLatest(result) {
  const n = overlay.traces.length;
  provEl.textContent = n > 1
    ? `${n} traces · latest: ${provenanceText(result)}` : provenanceText(result);
}

export default {
  name: "br_waveform",
  label: "B_r Waveform",
  accepts: ["analytical", "fem"],
  overlayPool: true, // routed results register compare-set membership

  init(container, app, headerSlot) {
    appCtx = app;

    hintEl = document.createElement("div");
    hintEl.className = "empty-hint";
    hintEl.textContent = "No waveform yet — select a config or submit an analytical/fem job.";

    plotEl = document.createElement("div");
    plotEl.style.height = "max(340px, 52vh)"; // Results-tab hero row

    const statsEl = document.createElement("div");
    statsEl.className = "cursor-stats-host";

    provEl = document.createElement("div");
    provEl.className = "provenance";

    // A/B cursors: θ is the shared axis, so peak-to-peak
    // and mean over a window read across all overlaid traces
    cursors = makeCursors({
      statsHost: statsEl,
      seriesFn: () => overlay.traces.map((t) => ({
        name: t.name, x: t.x, y: t.y, color: t.line?.color })),
      fmtX: (v) => v.toFixed(1),
      fmtY: (v) => (typeof v === "number" ? v.toFixed(4) : "—"),
      unit: "deg",
      sourceLabel: "B_r samples (full resolution)",
    });
    const clearBtn = document.createElement("button");
    clearBtn.textContent = "Clear overlay";
    // drops compare-set overlays app-wide (chips + all panels together)
    clearBtn.addEventListener("click", () => app.compare.clearOverlays());
    headerSlot.append(cursorToggleButton(cursors, redraw), clearBtn);

    container.append(hintEl, plotEl, statsEl, provEl);
    app.onHighlight((id, origin) => {
      if (origin !== "br_waveform") applyHighlight(id);
    });
    redraw();
  },

  // plain config click = single-config mode: the app clears plot panels
  // before routing the new config's results
  clear() {
    overlay.clear();
    provEl.textContent = "";
    redraw();
  },

  retheme: () => redraw(),

  // data: a full result payload from GET /api/results/{id}
  update(result) {
    const m = result.metrics;
    if (!m || !m.B_r_list || !m.theta_list) return;
    if (overlay.has(result.result_id)) return;
    const slot = appCtx.compare.slotFor(result.config?.motor?.name);
    if (slot === undefined) return;

    overlay.add(result.result_id, {
      _slot: slot,
      x: m.theta_list.map((t) => (t * 180) / Math.PI),
      y: m.B_r_list,
      mode: "lines",
      line: { color: SERIES[slot], width: 2, dash: MODEL_DASH[result.model] ?? "solid" },
      name: traceName(result),
      hovertemplate: "%{y:.4f} T<extra>%{fullData.name}</extra>",
    });
    stampLatest(result);
    redraw()?.then(bindHover);
  },

  destroy() {
    if (plotEl) Plotly.purge(plotEl);
    overlay.clear();
    plotEl = hintEl = provEl = appCtx = cursors = null;
    hoverBound = false;
  },
};
