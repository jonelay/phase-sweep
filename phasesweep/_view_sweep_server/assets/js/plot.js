// Plotly layout builder driven by spec/plot.json.
//
// Plotly needs concrete colours, so the builder resolves the spec's
// [bindings] table against live --vs-* tokens at layout-build time.
// Panels rebuild on vs:theme-change. The same table generates the static
// templates in spec/plotly/, so both paths agree by construction.
//
// Plotly is a peer dependency: this module assumes a global `Plotly`
// where it renders and never imports one.

import { token as readToken } from "./theme.js";

// Pure: apply bindings onto a deep copy of spec.layout using `resolve(name)`.
// Exported for tests and for WASM/Node callers that supply their own
// token source.
export function resolveTemplate(spec, resolve, series = []) {
  const layout = structuredClone(spec.layout);
  for (const [path, name] of Object.entries(spec.bindings)) {
    const keys = path.split(".");
    let node = layout;
    for (const k of keys.slice(0, -1)) node = node[k] ??= {};
    node[keys.at(-1)] = resolve(name);
  }
  const n = spec.colorway?.from_series ?? 0;
  if (n && series.length) layout.colorway = Array.from({ length: n }, (_, i) => series[i % series.length]);
  return layout;
}

export async function loadPlotSpec(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`plot spec ${url}: ${r.status}`);
  return r.json();
}

// Deep-merge `override` onto a clone of `base`: plain objects merge
// key-by-key, arrays and scalars replace wholesale.
export function deepMerge(base, override) {
  const isObj = (v) => v != null && typeof v === "object" && !Array.isArray(v);
  if (!isObj(base) || !isObj(override)) return structuredClone(override);
  const out = structuredClone(base);
  for (const [k, v] of Object.entries(override)) out[k] = k in out ? deepMerge(out[k], v) : structuredClone(v);
  return out;
}

export function createPlot({ spec, theme, root = document.documentElement }) {
  if (!spec?.bindings) throw new Error("createPlot: plot spec JSON required");
  const tok = (name) => readToken(name, root);
  const config = { ...spec.config };

  const base = () => resolveTemplate(spec, tok, theme?.series ?? []);

  const api = {
    spec,
    config,
    token: tok,

    axisStyle(title) {
      return {
        title: { text: title },
        gridcolor: tok("chart-grid"),
        zerolinecolor: tok("chart-zeroline"),
        color: tok("chart-muted"),
      };
    },

    // Full layout: spec defaults + bindings + caller overrides.
    layout(overrides = {}) {
      return deepMerge(base(), overrides);
    },

    // Plotly `template` object for consumers that prefer layout.template.
    template() {
      return { layout: base() };
    },

    sequential() {
      return theme ? theme.sequentialStops() : [[0, tok("seq-lo")], [0.5, tok("seq-mid")], [1, tok("seq-hi")]];
    },

    // Plotly.react with the shared config; toggles a hint element when empty.
    redraw(plotEl, traces, overrides = {}, hintEl = null) {
      const has = traces.length > 0;
      if (hintEl) hintEl.style.display = has ? "none" : "";
      plotEl.style.display = has ? "" : "none";
      return has ? Plotly.react(plotEl, traces, api.layout(overrides), config) : null;
    },

    // Entry point for WASM callers: typed arrays in, Plotly promise out.
    renderFromWasm(divId, traces, overrides = {}) {
      const el = typeof divId === "string" ? document.getElementById(divId) : divId;
      if (!el) throw new Error(`renderFromWasm: no element ${divId}`);
      const plain = traces.map((t) => Object.fromEntries(
        Object.entries(t).map(([k, v]) => [k, ArrayBuffer.isView(v) ? Array.from(v) : v]),
      ));
      return Plotly.react(el, plain, api.layout(overrides), config);
    },
  };
  return api;
}

// Overlay trace pool: insertion-ordered, keyed, no colour or eviction
// policy (that belongs to the consumer's compare-set logic).
export function makeOverlay() {
  const traces = [];
  const byKey = new Map();
  return {
    traces,
    has: (key) => byKey.has(key),
    add(key, trace) {
      const t = { _key: key, ...trace };
      traces.push(t);
      byKey.set(key, t);
    },
    remove(key) {
      const t = byKey.get(key);
      if (!t) return false;
      traces.splice(traces.indexOf(t), 1);
      byKey.delete(key);
      return true;
    },
    clear() {
      traces.length = 0;
      byKey.clear();
    },
  };
}
