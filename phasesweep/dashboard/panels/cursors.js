// A/B measurement cursors + per-panel stats readout.
// Per panel, same-axis traces only — our result panels have heterogeneous
// x-axes, so cursors are local to a panel and there is no global cursor.
// Two editable vertical Plotly shapes pinned at indices 0 and 1
// (handleRelayout maps a drag to A/B by shape index, so cursors MUST stay
// first in the shapes array), plus a stats strip: A, B, Δ, and per-series
// min/max/peak-to-peak/mean/σ over the [A,B] window. Transient hover-grade
// state — never persisted. Donor: toggleCursors/onCursorDrag/refreshStats.

import { chartVar } from "./plot_theme.js";

const COLS = ["A", "B", "Δ", "min", "max", "p-p", "mean", "σ"];

// opts: statsHost element, seriesFn() -> [{ name, x, y, color }] over the
// panel's same-axis traces, fmtX/fmtY formatters, x unit, honesty label.
export function makeCursors({ statsHost, seriesFn, fmtX, fmtY, unit = "", sourceLabel = "" }) {
  let on = false;
  let a = null;
  let b = null;

  function span() {
    let lo = Infinity;
    let hi = -Infinity;
    for (const s of seriesFn()) {
      for (const x of s.x) {
        if (x < lo) lo = x;
        if (x > hi) hi = x;
      }
    }
    return Number.isFinite(lo) && hi > lo ? [lo, hi] : null;
  }

  // cursors are shapes[0] and [1] — handleRelayout reads the drag by index
  function shapes() {
    if (!on || a === null) return [];
    return [a, b].map((x) => ({
      type: "line", x0: x, x1: x, yref: "paper", y0: 0, y1: 1,
      editable: true, line: { color: chartVar("zeroline"), width: 1.6, dash: "dot" },
    }));
  }

  // one pass per series: the nearest-sample values at A and B (the cursor
  // readout) alongside the window min/max/mean/σ, so a full-resolution
  // trace is scanned once, not three times
  function seriesStats(x, y, a0, b0) {
    const lo = Math.min(a0, b0);
    const hi = Math.max(a0, b0);
    let min = Infinity;
    let max = -Infinity;
    let sum = 0;
    let sumSq = 0;
    let n = 0;
    let va = null;
    let vb = null;
    let dA = Infinity;
    let dB = Infinity;
    for (let i = 0; i < x.length; i += 1) {
      const xi = x[i];
      const yi = y[i];
      // FiniteJSONResponse serializes NaN as null (drive_sim produces
      // NaN legitimately) — a null sample must not zero the mean or
      // become the min
      if (!Number.isFinite(yi)) continue;
      if (Math.abs(xi - a0) < dA) { dA = Math.abs(xi - a0); va = yi; }
      if (Math.abs(xi - b0) < dB) { dB = Math.abs(xi - b0); vb = yi; }
      if (xi < lo || xi > hi) continue;
      if (yi < min) min = yi;
      if (yi > max) max = yi;
      sum += yi;
      sumSq += yi * yi;
      n += 1;
    }
    if (!n) return null;
    const mean = sum / n;
    return { va, vb, min, max, pp: max - min, mean, std: Math.sqrt(Math.max(0, sumSq / n - mean * mean)) };
  }

  function refresh() {
    statsHost.replaceChildren();
    if (!on) return;
    const sp = span();
    if (!sp) {
      // toggled on with no finite data — no fabricated A/B positions
      a = null;
      b = null;
      const note = document.createElement("div");
      note.className = "cursor-head";
      note.textContent = "cursors: no data";
      statsHost.append(note);
      return;
    }
    if (a === null) {
      // data arrived after an empty toggle — seat the cursors now
      a = sp[0] + 0.35 * (sp[1] - sp[0]);
      b = sp[0] + 0.65 * (sp[1] - sp[0]);
    } else {
      // re-clamp on new data: a narrower x-range must not strand the
      // cursors outside every series' window (stats would vanish)
      a = Math.min(Math.max(a, sp[0]), sp[1]);
      b = Math.min(Math.max(b, sp[0]), sp[1]);
    }
    const lo = Math.min(a, b);
    const hi = Math.max(a, b);

    const head = document.createElement("div");
    head.className = "cursor-head";
    const strong = document.createElement("b");
    strong.textContent = `Δ = ${fmtX(hi - lo)}${unit ? ` ${unit}` : ""}`;
    head.append(`A = ${fmtX(a)}   B = ${fmtX(b)}   `, strong);
    if (sourceLabel) {
      const src = document.createElement("span");
      src.className = "cursor-src";
      src.textContent = `stats: ${sourceLabel}`;
      head.append("   ", src);
    }

    const table = document.createElement("table");
    table.className = "cursor-stats";
    const hr = document.createElement("tr");
    for (const c of ["signal", ...COLS]) {
      const th = document.createElement("th");
      th.textContent = c;
      hr.append(th);
    }
    table.append(hr);
    for (const s of seriesFn()) {
      const w = seriesStats(s.x, s.y, a, b);
      if (!w) continue;
      const tr = document.createElement("tr");
      const name = document.createElement("td");
      const sw = document.createElement("span");
      sw.className = "cursor-swatch";
      if (s.color) sw.style.background = s.color;
      name.append(sw, s.name);
      tr.append(name);
      for (const v of [w.va, w.vb, w.vb - w.va, w.min, w.max, w.pp, w.mean, w.std]) {
        const td = document.createElement("td");
        td.textContent = fmtY(v);
        tr.append(td);
      }
      table.append(tr);
    }
    statsHost.append(head, table);
  }

  return {
    active: () => on,
    shapes,
    refresh,
    toggle() {
      on = !on;
      if (on) {
        const sp = span();
        if (sp) {
          a = sp[0] + 0.35 * (sp[1] - sp[0]);
          b = sp[0] + 0.65 * (sp[1] - sp[0]);
        } else {
          a = null;
          b = null;
        }
      }
      return on;
    },
    // Plotly emits shapes[i].x0/.x1 on an editable-shape drag; cursors are
    // shapes 0 and 1. Returns true when it consumed the event so the panel
    // can skip a redundant redraw (Plotly already moved the line).
    handleRelayout(ev) {
      if (!on) return false;
      let hit = false;
      for (const k of Object.keys(ev)) {
        const m = /^shapes\[(\d+)\]\.x0$/.exec(k);
        if (!m) continue;
        const idx = Number(m[1]);
        if (idx === 0) { a = ev[k]; hit = true; }
        else if (idx === 1) { b = ev[k]; hit = true; }
      }
      if (hit) refresh();
      return hit;
    },
  };
}

// Header-slot toggle button shared by the cursor panels: owns the label,
// tooltip, and .active-class convention; the panel passes only its
// re-render callback (which differs — overlay redraw vs single-result
// react). Returns the button for the caller to append.
export function cursorToggleButton(cursors, onToggle) {
  const btn = document.createElement("button");
  btn.textContent = "Cursors";
  btn.title = "toggle A/B measurement cursors";
  btn.addEventListener("click", () => {
    btn.classList.toggle("active", cursors.toggle());
    onToggle();
  });
  return btn;
}
