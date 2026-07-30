// Config nameplate strip: identity row under the topbar
// for the ACTIVE config — pole pairs / topology from the config summary,
// field metrics from the latest routed FULL results (no new endpoint).
// Prefers fem when both models are present (analytical v4 genuinely
// differs on peak_Br/thd_pct); the source tile names which model the
// numbers came from. Subsumes the old topbar #active-config span.

import { statStrip } from "./widgets.js";

let nameEl = null;
let statsEl = null;
let appCtx = null;
const latest = new Map(); // model -> full result, active motor only

const fmtT = (v) => (typeof v === "number" ? `${v.toPrecision(3)} T` : "—");

function render() {
  const info = appCtx.activeConfigInfo();
  if (!info) {
    nameEl.textContent = "";
    statsEl.replaceChildren();
    return;
  }
  nameEl.textContent = info.name;
  nameEl.title = `active config · motor: ${info.motor_name ?? info.name}`;
  const r = latest.get("fem") ?? latest.get("analytical");
  const m = r?.metrics ?? {};
  statStrip(statsEl, [
    { label: "pole pairs", value: info.n_p != null ? String(info.n_p) : "—" },
    { label: "topology", value: info.topology ?? "—" },
    { label: "peak Bᵣ", value: fmtT(m.peak_Br) },
    { label: "fundamental", value: fmtT(m.fundamental) },
    { label: "THD",
      value: typeof m.thd_pct === "number" ? `${m.thd_pct.toPrecision(3)} %` : "—" },
    { label: "source", value: r ? r.model : "—" },
  ]);
}

export default {
  name: "nameplate",
  accepts: ["analytical", "fem"],

  init(container, app) {
    appCtx = app;
    nameEl = document.createElement("span");
    nameEl.className = "nameplate-name";
    statsEl = document.createElement("div");
    statsEl.className = "stat-strip";
    container.append(nameEl, statsEl);
    render();
  },

  // config switch / overlay rebuild: metrics reset, statics re-render —
  // clearPanels runs after the app updates the active config, so the
  // name row is already the new config's
  clear() {
    latest.clear();
    render();
  },

  update(result) {
    // overlay routes feed all panels — the nameplate is active-config only
    if (result.config?.motor?.name !== appCtx.getActiveMotor()) return;
    latest.set(result.model, result);
    render();
  },

  destroy() {
    latest.clear();
    nameEl = statsEl = appCtx = null;
  },
};
