// Sim waveforms panel: time-domain torque / speed / current
// from a drive_sim result, three stacked subplots on a shared time axis.
// Dashed markers: speed step and load step from the sim plan; w_ref on the
// speed subplot.

import { PLOT_CONFIG, SERIES, axisStyle, baseLayout, chartVar } from "./plot_theme.js";
import { provenanceText, statStrip } from "./widgets.js";
import { makeCursors, cursorToggleButton } from "./cursors.js";

const HINT = "No simulation yet — submit a drive_sim job.";

let plotEl = null;
let hintEl = null;
let captionEl = null;
let statsEl = null;
let cursorStatsEl = null;
let provEl = null;
let appCtx = null;
let cursors = null;
let cursorBound = false;
let shownKey = null;
let shownResult = null; // kept so retheme() can rebuild with fresh colors
let shownMotor = "";

// bound once the div is a live plotly plot: an editable-shape drag routes
// to the cursor A/B state
function bindCursorRelayout() {
  if (cursorBound || typeof plotEl?.on !== "function") return;
  cursorBound = true;
  plotEl.on("plotly_relayout", (ev) => cursors.handleRelayout(ev));
}

const fmt = (v, unit) => (typeof v === "number" ? `${v.toPrecision(3)} ${unit}` : "—");

// Single-result panel in a multi-config compare session shows only the
// latest routed result — badge names it, with the config's compare-set
// dot color where it maps to a live member. Refreshed
// via app.compare.onChange.
function renderCaption() {
  captionEl.replaceChildren();
  if (!shownMotor) return;
  const slot = appCtx.compare.slotFor(shownMotor);
  if (slot !== undefined) {
    const dot = document.createElement("span");
    dot.className = "chip-dot";
    dot.style.background = SERIES[slot];
    captionEl.append(dot);
  }
  const overlaid = appCtx.compare.overlays().length;
  captionEl.append(
    overlaid > 0 ? `showing: ${shownMotor} (latest routed)` : shownMotor,
  );
}

export default {
  name: "sim_waveforms",
  label: "Sim Waveforms",
  accepts: ["drive_sim"],

  init(container, app, headerSlot) {
    appCtx = app;
    const self = this;
    hintEl = document.createElement("div");
    hintEl.className = "empty-hint";
    hintEl.textContent = HINT;

    captionEl = document.createElement("div");
    captionEl.className = "panel-caption";

    statsEl = document.createElement("div");
    statsEl.className = "stat-strip";

    plotEl = document.createElement("div");
    plotEl.style.height = "460px";
    plotEl.style.display = "none";

    cursorStatsEl = document.createElement("div");
    cursorStatsEl.className = "cursor-stats-host";

    provEl = document.createElement("div");
    provEl.className = "provenance";

    // A/B cursors: all three subplots share the time axis,
    // so a window yields settle deltas (ω p-p) and steady means per signal
    cursors = makeCursors({
      statsHost: cursorStatsEl,
      seriesFn: () => {
        const m = shownResult?.metrics;
        if (!m?.t_list) return [];
        return [
          { name: "τ_M", x: m.t_list, y: m.tau_M_list, color: SERIES[0] },
          { name: "ω_M", x: m.t_list, y: m.w_M_list, color: SERIES[1] },
          { name: "|i_s|", x: m.t_list, y: m.i_s_abs_list, color: SERIES[2] },
        ];
      },
      fmtX: (v) => v.toPrecision(3),
      fmtY: (v) => (typeof v === "number" ? v.toPrecision(3) : "—"),
      unit: "s",
      sourceLabel: "drive_sim samples (full resolution)",
    });
    // re-render owns the refresh: with data, update()'s react.then rebuilds
    // the stats (or clears them when off); with none, refresh directly
    headerSlot?.append(cursorToggleButton(cursors, () => {
      if (shownResult) { shownKey = null; self.update(shownResult); }
      else cursors.refresh();
    }));

    container.append(hintEl, captionEl, statsEl, plotEl, cursorStatsEl, provEl);
    app.compare.onChange(renderCaption);
  },

  clear() {
    shownKey = null;
    shownResult = null;
    shownMotor = "";
    hintEl.textContent = HINT;
    hintEl.style.display = "";
    captionEl.textContent = "";
    statsEl.replaceChildren();
    cursorStatsEl.replaceChildren();
    provEl.textContent = "";
    plotEl.style.display = "none";
  },

  retheme() {
    if (!shownResult) return;
    shownKey = null; // defeat the same-id dedup; colors changed
    this.update(shownResult);
  },

  update(result) {
    const m = result.metrics;
    if (!m) return;
    if (!m.t_list) {
      if (shownKey === null) {
        hintEl.textContent = "This drive_sim result predates waveform capture — re-run the drive_sim job.";
      }
      return;
    }
    if (shownKey === result.result_id) return;
    shownKey = result.result_id;
    shownResult = result;

    const traces = [
      { x: m.t_list, y: m.tau_M_list, yaxis: "y", name: "τ_M",
        mode: "lines", line: { color: SERIES[0], width: 2 } },
      { x: m.t_list, y: m.w_M_list, yaxis: "y2", name: "ω_M",
        mode: "lines", line: { color: SERIES[1], width: 2 } },
      { x: m.t_list, y: m.i_s_abs_list, yaxis: "y3", name: "|i_s|",
        mode: "lines", line: { color: SERIES[2], width: 2 } },
    ];

    const plan = result.config?.sim_plan;
    const shapes = [];
    for (const t of [plan?.speed_step_time, plan?.load_time]) {
      if (typeof t !== "number") continue;
      shapes.push({
        type: "line", xref: "x", yref: "paper",
        x0: t, x1: t, y0: 0, y1: 1,
        line: { color: chartVar("zeroline"), width: 1, dash: "dot" },
      });
    }
    const wRef = result.config?.motor?.drive?.W_REF;
    if (typeof wRef === "number") {
      shapes.push({
        type: "line", xref: "paper", yref: "y2",
        x0: 0, x1: 1, y0: wRef, y1: wRef,
        line: { color: SERIES[1], width: 1, dash: "dash" },
      });
    }

    const layout = baseLayout({
      margin: { l: 65, r: 12, t: 8, b: 42 },
      showlegend: false,
      xaxis: { ...axisStyle("t [s]"), anchor: "y3" },
      yaxis: { ...axisStyle("τ_M [Nm]"), domain: [0.72, 1] },
      yaxis2: { ...axisStyle("ω_M [rad/s]"), domain: [0.38, 0.66] },
      yaxis3: { ...axisStyle("|i_s| [A]"), domain: [0.04, 0.32] },
      // cursors first: handleRelayout maps drags by index
      shapes: [...cursors.shapes(), ...shapes],
    });

    shownMotor = result.config?.motor?.name ?? "?";
    renderCaption();
    statStrip(statsEl, [
      { label: "settle time", value: fmt(m.t_settle, "s") },
      { label: "steady current", value: fmt(m.i_ss, "A") },
      { label: "speed droop",
        value: typeof m.speed_droop === "number"
          ? `${(m.speed_droop * 100).toPrecision(3)} %` : "—" },
      { label: "peak torque", value: fmt(m.tau_peak, "Nm") },
    ]);
    provEl.textContent = provenanceText(result);

    hintEl.style.display = "none";
    plotEl.style.display = "";
    Plotly.react(plotEl, traces, layout, PLOT_CONFIG).then(() => {
      bindCursorRelayout();
      cursors.refresh(); // builds when on (tracks the new result), clears when off
    });
  },

  destroy() {
    if (plotEl) Plotly.purge(plotEl);
    plotEl = hintEl = captionEl = statsEl = cursorStatsEl = provEl = appCtx = cursors = null;
    shownKey = shownResult = null;
    shownMotor = "";
    cursorBound = false;
  },
};
