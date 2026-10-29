// Cross-section heatmap panel: |B| over the motor
// cross-section from the FEM raster (B_mag_grid), material region
// boundaries drawn as circles from the config geometry. Plotly heatmap
// (no Three.js).
//
// Multi-config: when 2+ configs have FEM results in the compare set, the
// panel tiles them side-by-side (internal columns) rather
// than overlaying — heatmaps don't overlay. Each tile gets its own Plotly
// div, colorbar, boundary circles, and a label with the compare-set dot.

import { provenanceText } from "/view-sweep/js/widgets.js";

// |B| above ~2.2 T is corner/singularity overshoot in linear solves;
// clamp the color ramp so it doesn't wash out the field structure
// (hover still shows the true value).
const ZMAX_CAP = 2.5;

const HINT = "No cross-section yet — submit a fem job.";

const slots = new Map(); // motor display name → {tileEl, plotEl, labelEl, provEl, result, key, captionRest}
let containerEl = null;
let hintEl = null;
let appCtx = null;

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

function renderLabel(motorName, slot) {
  slot.labelEl.replaceChildren();
  const cs = appCtx.compare.slotFor(motorName);
  if (cs !== undefined) {
    const dot = document.createElement("span");
    dot.className = "vs-chip-dot";
    dot.style.background = appCtx.theme.series[cs];
    slot.labelEl.append(dot);
  }
  slot.labelEl.append(motorName + slot.captionRest);
}

function renderTile(motorName, slot) {
  const result = slot.result;
  const m = result.metrics;
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
    line: { color: appCtx.plot.token("chart-ink"), width: 1 },
  }));

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
    colorscale: appCtx.plot.sequential(),
    zmin: 0,
    zmax: Math.min(zmax, ZMAX_CAP),
    hoverongaps: false,
    hovertemplate: "x=%{x:.2f}, y=%{y:.2f} mm · |B|=%{z:.3f} T<extra></extra>",
    colorbar,
  };
  const { plot } = appCtx;
  const layout = plot.layout({
    hovermode: "closest",
    showlegend: false,
    xaxis: plot.axisStyle("x [mm]"),
    yaxis: { ...plot.axisStyle("y [mm]"), scaleanchor: "x", scaleratio: 1 },
    shapes,
  });

  const ironNote = typeof m.b_iron_max === "number"
    ? ` · peak iron |B| ${m.b_iron_max.toFixed(2)} T` : "";
  slot.captionRest = ironNote
    + (clipped ? ` · peak |B| ${zmax.toFixed(2)} T (ramp clipped)` : "");
  renderLabel(motorName, slot);
  slot.provEl.textContent = provenanceText(result);

  Plotly.react(slot.plotEl, [trace], layout, plot.config);
}

function syncTileLayout() {
  for (const slot of slots.values()) {
    slot.tileEl.style.flex = slots.size > 1 ? "1 1 0" : "1 1 100%";
  }
  requestAnimationFrame(() => {
    for (const slot of slots.values()) {
      if (slot.plotEl) Plotly.Plots.resize(slot.plotEl);
    }
  });
}

function pruneSlots() {
  let removed = false;
  for (const [name, slot] of slots) {
    const inSet = appCtx.compare.slotFor(name) !== undefined;
    const isActive = name === appCtx.getActiveMotor();
    if (!inSet && !isActive) {
      Plotly.purge(slot.plotEl);
      slot.tileEl.remove();
      slots.delete(name);
      removed = true;
    }
  }
  for (const [name, slot] of slots) renderLabel(name, slot);
  if (removed) syncTileLayout();
  if (!slots.size) {
    hintEl.textContent = HINT;
    hintEl.style.display = "";
    containerEl.style.display = "none";
  }
}

export default {
  name: "cross_section",
  label: "Cross-Section |B|",
  accepts: ["fem"],

  init(container, app) {
    appCtx = app;
    hintEl = document.createElement("div");
    hintEl.className = "vs-empty-hint";
    hintEl.textContent = HINT;

    containerEl = document.createElement("div");
    containerEl.className = "cs-tiles";
    containerEl.style.display = "none";

    container.append(hintEl, containerEl);
    app.compare.onChange(() => pruneSlots());
  },

  clear() {
    for (const slot of slots.values()) {
      Plotly.purge(slot.plotEl);
      slot.tileEl.remove();
    }
    slots.clear();
    hintEl.textContent = HINT;
    hintEl.style.display = "";
    containerEl.style.display = "none";
  },

  retheme() {
    for (const [name, slot] of slots) {
      slot.key = null;
      renderTile(name, slot);
    }
  },

  update(result) {
    const m = result.metrics;
    if (!m) return;
    if (!m.B_mag_grid || !m.grid_coords_list) {
      if (!slots.size) {
        hintEl.textContent = "This fem result predates cross-section capture — re-run the fem job.";
      }
      return;
    }
    const motorName = result.config?.motor?.name ?? "?";

    let slot = slots.get(motorName);
    if (slot && slot.key === result.result_id) return;

    if (!slot) {
      const tileEl = document.createElement("div");
      tileEl.className = "cs-tile";

      const labelEl = document.createElement("div");
      labelEl.className = "vs-caption cs-tile-label";

      const plotEl = document.createElement("div");
      plotEl.style.height = "420px";

      const provEl = document.createElement("div");
      provEl.className = "vs-provenance";

      tileEl.append(labelEl, plotEl, provEl);
      containerEl.append(tileEl);

      slot = { tileEl, plotEl, labelEl, provEl, result: null, key: null, captionRest: "" };
      slots.set(motorName, slot);
    }

    slot.result = result;
    slot.key = result.result_id;
    renderTile(motorName, slot);

    hintEl.style.display = "none";
    containerEl.style.display = "";
    syncTileLayout();
  },

  destroy() {
    for (const slot of slots.values()) Plotly.purge(slot.plotEl);
    slots.clear();
    containerEl = hintEl = appCtx = null;
  },
};
