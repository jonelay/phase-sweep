// Harmonic spectrum panel: bar chart of B_r harmonic
// magnitudes, overlay mode. Amplitudes are computed client-side by DFT of
// B_r_list (theta is uniform over [0, 2π)), matching harmonics_1sided.
// x-axis is the ELECTRICAL order h = n / n_p so overlays of motors with
// different pole counts align at h = 1, 3, 5, ...
// Color = config (compare-set slot); bars can't dash, so model identity
// is a pattern fill on fem bars (analytical plain).

import { SERIES, axisStyle, baseLayout, makeOverlay, redrawOverlay, traceName } from "./plot_theme.js";
import { provenanceText } from "./widgets.js";

// bar-chart rendering of MODEL_DASH's identity split
const MODEL_BAR_PATTERN = {
  fem: { shape: "/", fillmode: "overlay", fgcolor: "#ffffff", fgopacity: 0.45 },
};

const SCALE_KEY = "ps-spectrum-scale";
const SCALES = { linear: "linear", log: "log", pct: "% of fund." };
let scaleMode = localStorage.getItem(SCALE_KEY);
if (!Object.hasOwn(SCALES, scaleMode ?? "")) scaleMode = "linear";

// built per redraw so a theme switch picks up the current --chart-* colors
const layout = () => baseLayout({
  hovermode: "closest",
  barmode: "group",
  xaxis: axisStyle("electrical order h = n / n_p"),
  yaxis: {
    ...axisStyle(scaleMode === "pct"
      ? "|B_r| [% of fundamental]" : "|B_r| harmonic amplitude [T]"),
    type: scaleMode === "log" ? "log" : "linear",
    // tie the axis revision to the scale so a toggle resets the
    // y-view (a linear range preserved under type=log is garbage);
    // zoom otherwise survives via baseLayout's constant uirevision
    uirevision: scaleMode,
  },
});

// %-mode normalizes each trace to its own fundamental so overlays of
// different motors stay comparable
function displayTraces() {
  return overlay.traces.map((t) => {
    // re-read the slot's hue so a palette switch (SERIES mutated in place,
    // plot_theme.js) re-hues bars colored at route time
    const marker = { ...t.marker, color: SERIES[t._slot] };
    if (scaleMode !== "pct" || !(t._fund > 0)) {
      return { ...t, marker, y: t._rawY,
        hovertemplate: "n=%{customdata} (h=%{x:.2f}) · %{y:.4f} T<extra>%{fullData.name}</extra>" };
    }
    return { ...t, marker, y: t._rawY.map((v) => (100 * v) / t._fund),
      hovertemplate: "n=%{customdata} (h=%{x:.2f}) · %{y:.2f} %<extra>%{fullData.name}</extra>" };
  });
}

function harmonicAmps(B, maxOrder) {
  const N = B.length;
  const amps = [];
  for (let k = 1; k <= maxOrder; k++) {
    let re = 0;
    let im = 0;
    for (let n = 0; n < N; n++) {
      const a = (2 * Math.PI * k * n) / N;
      re += B[n] * Math.cos(a);
      im -= B[n] * Math.sin(a);
    }
    amps.push((2 / N) * Math.hypot(re, im));
  }
  return amps; // amps[k-1] = mechanical order k
}

let plotEl = null;
let hintEl = null;
let provEl = null;
let appCtx = null;
let hoverBound = false;

const overlay = makeOverlay();

function redraw() {
  return redrawOverlay(plotEl, hintEl, displayTraces(), layout());
}

// linked highlight: bars can't fatten, so dim the
// non-matching traces instead; null restores. Transient hover state.
function applyHighlight(id) {
  if (!plotEl?.data?.length) return;
  Plotly.restyle(plotEl, {
    opacity: overlay.traces.map((t) => (id && t._key !== id ? 0.3 : 1)),
  });
}

// bar hover announces the trace's result id; bound once the div is a
// live plotly plot (first non-empty redraw)
function bindHover() {
  if (hoverBound || typeof plotEl?.on !== "function") return;
  hoverBound = true;
  plotEl.on("plotly_hover", (ev) => {
    const key = overlay.traces[ev.points?.[0]?.curveNumber]?._key;
    if (key) appCtx.highlightResult(key, "harmonics");
  });
  plotEl.on("plotly_unhover", () => appCtx.highlightResult(null, "harmonics"));
}

// Overlay pool shows many results at once — the stamp names the latest
// routed one (per-trace identity stays in hover), with the trace count.
function stampLatest(result) {
  const n = overlay.traces.length;
  provEl.textContent = n > 1
    ? `${n} traces · latest: ${provenanceText(result)}` : provenanceText(result);
}

export default {
  name: "harmonics",
  label: "Harmonic Spectrum",
  accepts: ["analytical", "fem"],
  overlayPool: true, // routed results register compare-set membership

  init(container, app, headerSlot) {
    appCtx = app;
    const scaleSel = document.createElement("select");
    scaleSel.title = "spectrum y scale";
    for (const [value, label] of Object.entries(SCALES)) {
      scaleSel.add(new Option(label, value));
    }
    scaleSel.value = scaleMode;
    scaleSel.addEventListener("change", () => {
      scaleMode = scaleSel.value;
      localStorage.setItem(SCALE_KEY, scaleMode);
      redraw();
    });
    const clearBtn = document.createElement("button");
    clearBtn.textContent = "Clear overlay";
    // drops compare-set overlays app-wide (chips + all panels together)
    clearBtn.addEventListener("click", () => app.compare.clearOverlays());
    headerSlot.append(scaleSel, clearBtn);

    hintEl = document.createElement("div");
    hintEl.className = "empty-hint";
    hintEl.textContent = "No spectrum yet — select a config or submit an analytical/fem job.";

    plotEl = document.createElement("div");
    plotEl.style.height = "340px";

    provEl = document.createElement("div");
    provEl.className = "provenance";

    container.append(hintEl, plotEl, provEl);
    app.onHighlight((id, origin) => {
      if (origin !== "harmonics") applyHighlight(id);
    });
    redraw();
  },

  clear() {
    overlay.clear();
    provEl.textContent = "";
    redraw();
  },

  retheme: () => redraw(),

  update(result) {
    const m = result.metrics;
    const nP = result.config?.motor?.n_p;
    if (!m || !m.B_r_list || !nP) return;
    if (overlay.has(result.result_id)) return;
    const slot = appCtx.compare.slotFor(result.config?.motor?.name);
    if (slot === undefined) return; // routeResult registers before dispatch

    // Cover the odd electrical harmonics AND the slot sidebands Q ± n_p.
    const nSlots = result.config?.motor?.geometry?.n_slots ?? 0;
    const maxOrder = Math.min(
      Math.floor(m.B_r_list.length / 2) - 1,
      Math.max(5 * nP, nSlots + 2 * nP),
    );
    const amps = harmonicAmps(m.B_r_list, maxOrder);

    const marker = { color: SERIES[slot] };
    if (MODEL_BAR_PATTERN[result.model]) marker.pattern = MODEL_BAR_PATTERN[result.model];
    overlay.add(result.result_id, {
      _slot: slot,
      type: "bar",
      x: amps.map((_, i) => (i + 1) / nP),
      y: amps,
      _rawY: amps,
      _fund: amps[nP - 1], // electrical h=1 is mechanical order n_p
      width: 0.55 / nP,
      customdata: amps.map((_, i) => i + 1),
      marker,
      name: traceName(result),
    });
    stampLatest(result);
    redraw()?.then(bindHover);
  },

  destroy() {
    if (plotEl) Plotly.purge(plotEl);
    overlay.clear();
    plotEl = hintEl = provEl = appCtx = null;
    hoverBound = false;
  },
};
