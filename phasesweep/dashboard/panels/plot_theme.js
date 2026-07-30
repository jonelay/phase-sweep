// Shared plotly styling for result panels. Plotly needs concrete colors,
// so axisStyle/baseLayout/sequential read the --chart-* custom properties
// (style.css, both themes) at layout-build time — panels re-render on a
// theme switch via their retheme() hook. Color slots are assigned per
// CONFIG at compare-set membership time (app.js): a config holds its SLOT
// for its whole membership, so its dot always matches its traces. The slot
// stays put; the HUE at that slot is the user's selected palette. Switching
// palettes re-hues every slot at once (a deliberate re-hue event, app.js),
// but is otherwise theme-invariant — light/dark never re-hue a slot.

// Categorical palettes for the SERIES slots. All CVD-safe. Tol's sets are
// 7 colors; the getter cycles them to fill the 8-slot config cap — an 8th
// config repeats slot 0's hue (distinguished by chip label).
// Okabe-Ito is the 8-color Wong set with its pure black swapped for grey so
// slot 8 stays legible on the dark theme; strong hues front-loaded.
// Slot order is contrast-aware: slot 0 (the default single-config colour)
// leads with a blue/teal in every set (a warm orange/red as the default
// reads poorly), and the early slots — the most-compared configs — clear
// ~3:1 against BOTH plot backgrounds so no early trace is faint on either
// theme; the palettes' pale colours (cyan/yellow/grey, weak on the light
// background) sit in the last slots. The colour SET is unchanged from each
// published palette, only the slot order.
export const CATEGORICAL_PALETTES = {
  okabe_ito: {
    label: "Okabe-Ito",
    colors: ["#0072b2", "#009e73", "#d55e00", "#cc79a7",
             "#56b4e9", "#e69f00", "#999999", "#f0e442"],
  },
  tol_vibrant: {
    label: "Tol vibrant",
    colors: ["#009988", "#0077bb", "#cc3311", "#ee3377",
             "#ee7733", "#33bbee", "#bbbbbb"],
  },
  tol_bright: {
    label: "Tol bright",
    colors: ["#4477aa", "#228833", "#ee6677", "#aa3377",
             "#66ccee", "#ccbb44", "#bbbbbb"],
  },
  seaborn_colorblind: {
    label: "Seaborn colorblind",
    colors: ["#0173b2", "#029e73", "#d55e00", "#cc78bc", "#de8f05",
             "#56b4e9", "#ca9161", "#949494", "#fbafe4", "#ece133"],
  },
  tableau10: {
    label: "Tableau 10",
    colors: ["#4e79a7", "#59a14f", "#e15759", "#b07aa1", "#f28e2b",
             "#76b7b2", "#9c755f", "#bab0ac", "#ff9da7", "#edc948"],
  },
};
export const DEFAULT_PALETTE = "okabe_ito";
const SLOTS = 8; // compare-set config cap

// SERIES is a LIVE 8-slot array — applyPalette mutates it in place so every
// panel's SERIES[slot] read picks up new hues on its next retheme(), and
// app.js's [...SERIES.keys()] slot pool stays a stable 0..7.
export const SERIES = [];

export function paletteId() {
  const id = localStorage.getItem("ps-palette");
  return CATEGORICAL_PALETTES[id] ? id : DEFAULT_PALETTE;
}

export function applyPalette(id) {
  const chosen = CATEGORICAL_PALETTES[id] ? id : DEFAULT_PALETTE;
  localStorage.setItem("ps-palette", chosen);
  const { colors } = CATEGORICAL_PALETTES[chosen];
  SERIES.length = 0;
  for (let i = 0; i < SLOTS; i++) SERIES.push(colors[i % colors.length]);
  return chosen;
}
applyPalette(paletteId()); // populate SERIES before any importer reads it

// Model identity within a config's color: fixed map, not
// an arrival-order counter, so the same model always gets the same dash
// across configs. Same-model sweep points within a config render alike —
// hover's r_o distinguishes them.
export const MODEL_DASH = { analytical: "solid", fem: "dash" };

export const chartVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(`--chart-${name}`).trim();

// |B| heatmap ramp selector. "accent" is the theme-aware
// single-hue ramp (dark variant runs dark→bright so high field still pops);
// "viridis" is the perceptually-uniform map — but both are theme-aware, so
// low field never sinks into the panel background.
export const SEQ_RAMPS = {
  accent: { label: "Accent" }, // reads theme --chart-seq-* vars (RAMPS below)
  viridis: { label: "Viridis" },
  cividis: { label: "Cividis" },
  magma: { label: "Magma" },
};
export const DEFAULT_SEQ = "accent";

// Perceptual ramps carry the full colormap on the light panel. On the dark
// panel their darkest stops fall below ~2.5:1 against the near-black surface
// and sink into it, so each dark variant lifts the floor to the lowest stop
// that stays legible — keeping as much of the ramp's hue as possible. (Accent
// is theme-aware via the --chart-seq-* vars; its low end fades into the panel
// by design, so it isn't listed here.)
const RAMPS = {
  viridis: {
    light: [[0, "#440154"], [0.13, "#472d7b"], [0.25, "#3b528b"], [0.38, "#2c728e"],
      [0.5, "#21918c"], [0.63, "#28ae80"], [0.75, "#5ec962"], [0.88, "#addc30"], [1, "#fde725"]],
    dark: [[0, "#2c728e"], [0.25, "#21918c"], [0.5, "#28ae80"],
      [0.7, "#5ec962"], [0.85, "#addc30"], [1, "#fde725"]],
  },
  cividis: {
    light: [[0, "#00224e"], [0.11, "#123570"], [0.22, "#3b496c"], [0.33, "#575d6e"],
      [0.44, "#707173"], [0.56, "#8a8779"], [0.67, "#a69d75"], [0.78, "#c3b56c"],
      [0.89, "#e1cd60"], [1, "#ffea46"]],
    dark: [[0, "#575d6e"], [0.2, "#707173"], [0.4, "#8a8779"],
      [0.6, "#a69d75"], [0.8, "#c3b56c"], [0.9, "#e1cd60"], [1, "#ffea46"]],
  },
  magma: {
    light: [[0, "#000004"], [0.11, "#180f3e"], [0.22, "#451077"], [0.33, "#721f81"],
      [0.44, "#9f2f7f"], [0.56, "#cd4071"], [0.67, "#f1605d"], [0.78, "#fd9567"],
      [0.89, "#feca8d"], [1, "#fcfdbf"]],
    dark: [[0, "#9f2f7f"], [0.25, "#cd4071"], [0.5, "#f1605d"],
      [0.7, "#fd9567"], [0.85, "#feca8d"], [1, "#fcfdbf"]],
  },
};

export function seqId() {
  const id = localStorage.getItem("ps-seq");
  return SEQ_RAMPS[id] ? id : DEFAULT_SEQ;
}

export function applySeq(id) {
  const chosen = SEQ_RAMPS[id] ? id : DEFAULT_SEQ;
  localStorage.setItem("ps-seq", chosen);
  return chosen;
}

const isDark = () => document.documentElement.dataset.theme === "dark";

export const sequential = () => {
  const ramp = RAMPS[seqId()];
  if (ramp) return isDark() ? ramp.dark : ramp.light;
  return [[0, chartVar("seq-lo")], [0.5, chartVar("seq-mid")], [1, chartVar("seq-hi")]];
};

// scrollZoom: wheel zoom on all plot panels; zoom persists across
// redraws via the constant uirevision below
export const PLOT_CONFIG = { responsive: true, displaylogo: false, scrollZoom: true };

// Overlay trace pool shared by the waveform/spectrum panels: a plain
// dedup'd trace list. Color/eviction moved to app.js compare-set state
// (color per config, cap per config) — panels pass finished traces with
// externally assigned colors.
export function makeOverlay() {
  const traces = []; // insertion order = overlay order
  const byKey = new Map(); // result_id -> trace
  return {
    traces,
    has: (key) => byKey.has(key),
    add(key, trace) {
      const t = { _key: key, ...trace };
      traces.push(t);
      byKey.set(key, t);
    },
    clear() {
      traces.length = 0;
      byKey.clear();
    },
  };
}

// returns the Plotly.react promise (null when empty) so callers can
// bind plot event listeners once the div is a live plotly plot
export function redrawOverlay(plotEl, hintEl, traces, layout) {
  hintEl.style.display = traces.length ? "none" : "";
  plotEl.style.display = traces.length ? "" : "none";
  return traces.length ? Plotly.react(plotEl, traces, layout, PLOT_CONFIG) : null;
}

// Overlay legend identity: motor + model, plus r_outer so sweep points
// (same motor name, same model) stay distinguishable.
export function traceName(result) {
  const motor = result.config?.motor ?? {};
  const base = `${motor.name ?? "?"} · ${result.model}`;
  const rOuter = motor.geometry?.r_outer;
  return typeof rOuter === "number"
    ? `${base} · r_o ${(rOuter * 1000).toFixed(2)}` : base;
}

export function axisStyle(title) {
  return {
    title: { text: title },
    gridcolor: chartVar("grid"),
    zerolinecolor: chartVar("zeroline"),
    color: chartVar("muted"),
  };
}

export function baseLayout(overrides = {}) {
  return {
    margin: { l: 55, r: 12, t: 8, b: 42 },
    paper_bgcolor: chartVar("bg"),
    plot_bgcolor: chartVar("bg"),
    font: {
      family: 'system-ui, -apple-system, "Segoe UI", sans-serif',
      size: 12,
      color: chartVar("text"),
    },
    hoverlabel: {
      bgcolor: chartVar("bg"),
      bordercolor: chartVar("grid"),
      font: { color: chartVar("text") },
    },
    modebar: { color: chartVar("muted"), activecolor: chartVar("text") },
    hovermode: "x unified",
    // overlay panels run legend-free (chips + hover carry identity);
    // model_comparison overrides with its own legend
    showlegend: false,
    // constant revision: user zoom/pan survives data redraws (trace
    // arrays still swap — uirevision preserves view state, not traces)
    uirevision: "keep",
    ...overrides,
  };
}
