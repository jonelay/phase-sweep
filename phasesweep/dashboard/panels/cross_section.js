// Cross-section heatmap panel: |B| over the motor
// cross-section from the FEM raster (B_mag_grid), material region
// boundaries drawn as circles from the config geometry. Plotly heatmap
// (decision 2026-07-09: no Three.js).

import { PLOT_CONFIG, SERIES, axisStyle, baseLayout, chartVar, sequential } from "./plot_theme.js";
import { provenanceText } from "./widgets.js";

// |B| above ~2.2 T is corner/singularity overshoot in linear solves;
// clamp the color ramp so it doesn't wash out the field structure
// (hover still shows the true value).
const ZMAX_CAP = 2.5;

const HINT = "No cross-section yet — submit a fem job.";

let plotEl = null;
let hintEl = null;
let captionEl = null;
let provEl = null;
let appCtx = null;
let shownKey = null;
let shownResult = null; // kept so retheme() can rebuild with fresh colors
let shownMotor = "";
let captionRest = "";

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
    (overlaid > 0 ? `showing: ${shownMotor} (latest routed)` : shownMotor) + captionRest,
  );
}

function boundaryRadii(geometry) {
  if (!geometry) return [];
  const radii = ["r_inner", "r_rotor", "r_magnet", "r_stator", "r_outer"]
    .map((k) => geometry[k])
    .filter((r) => typeof r === "number" && r > 0);
  if (geometry.back_iron_thickness != null) {
    radii.push(geometry.r_rotor + geometry.back_iron_thickness);
  }
  return radii;
}

export default {
  name: "cross_section",
  label: "Cross-Section |B|",
  accepts: ["fem"],

  init(container, app) {
    appCtx = app;
    hintEl = document.createElement("div");
    hintEl.className = "empty-hint";
    hintEl.textContent = HINT;

    captionEl = document.createElement("div");
    captionEl.className = "panel-caption";

    plotEl = document.createElement("div");
    plotEl.style.height = "420px";
    plotEl.style.display = "none";

    provEl = document.createElement("div");
    provEl.className = "provenance";

    container.append(hintEl, captionEl, plotEl, provEl);
    app.compare.onChange(renderCaption);
  },

  clear() {
    shownKey = null;
    shownResult = null;
    shownMotor = "";
    captionRest = "";
    hintEl.textContent = HINT;
    hintEl.style.display = "";
    captionEl.textContent = "";
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
    if (!m.B_mag_grid || !m.grid_coords_list) {
      if (shownKey === null) {
        hintEl.textContent = "This fem result predates cross-section capture — re-run the fem job.";
      }
      return;
    }
    if (shownKey === result.result_id) return;
    shownKey = result.result_id;
    shownResult = result;

    const mm = m.grid_coords_list.map((c) => c * 1000);
    let zmax = 0;
    for (const row of m.B_mag_grid) {
      for (const v of row) if (v !== null && v > zmax) zmax = v;
    }

    const geometry = result.config?.motor?.geometry;
    const shapes = boundaryRadii(geometry).map((r) => ({
      type: "circle",
      xref: "x", yref: "y",
      x0: -r * 1000, y0: -r * 1000, x1: r * 1000, y1: r * 1000,
      line: { color: chartVar("ink"), width: 1 },
    }));

    // clipped ramp: pin the ticks so the top one can say "≥CAP"
    const clipped = zmax > ZMAX_CAP;
    const colorbar = { title: { text: "|B| [T]" }, thickness: 12, outlinewidth: 0 };
    if (clipped) {
      colorbar.tickvals = [];
      colorbar.ticktext = [];
      for (let v = 0; v < ZMAX_CAP; v += 0.5) {
        colorbar.tickvals.push(v);
        colorbar.ticktext.push(String(v));
      }
      colorbar.tickvals.push(ZMAX_CAP);
      colorbar.ticktext.push(`≥${ZMAX_CAP}`);
    }
    const trace = {
      type: "heatmap",
      x: mm,
      y: mm,
      z: m.B_mag_grid,
      colorscale: sequential(),
      zmin: 0,
      zmax: Math.min(zmax, ZMAX_CAP),
      hoverongaps: false,
      hovertemplate: "x=%{x:.2f}, y=%{y:.2f} mm · |B|=%{z:.3f} T<extra></extra>",
      colorbar,
    };
    const layout = baseLayout({
      hovermode: "closest",
      showlegend: false,
      xaxis: axisStyle("x [mm]"),
      yaxis: { ...axisStyle("y [mm]"), scaleanchor: "x", scaleratio: 1 },
      shapes,
    });

    const ironNote = typeof m.b_iron_max === "number"
      ? ` · peak iron |B| ${m.b_iron_max.toFixed(2)} T` : "";
    shownMotor = result.config?.motor?.name ?? "?";
    captionRest = ironNote
      + (clipped ? ` · peak |B| ${zmax.toFixed(2)} T (ramp clipped)` : "");
    renderCaption();
    provEl.textContent = provenanceText(result);

    hintEl.style.display = "none";
    plotEl.style.display = "";
    Plotly.react(plotEl, [trace], layout, PLOT_CONFIG);
  },

  destroy() {
    if (plotEl) Plotly.purge(plotEl);
    plotEl = hintEl = captionEl = provEl = appCtx = null;
    shownKey = shownResult = null;
    shownMotor = "";
    captionRest = "";
  },
};
