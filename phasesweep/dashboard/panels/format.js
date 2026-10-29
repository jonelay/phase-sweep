// Small shared formatters for the list panels (run history, jobs).

// "4d ago" — coarse relative age; pair with absTs() on hover for the exact
// value. Returns the raw string when the timestamp won't parse.
export function relTime(ts) {
  if (!ts) return "—";
  const then = new Date(ts).getTime();
  if (!Number.isFinite(then)) return ts;
  const s = Math.max(0, (Date.now() - then) / 1000);
  if (s < 45) return "just now";
  const m = s / 60;
  if (m < 60) return `${Math.round(m)}m ago`;
  const h = m / 60;
  if (h < 24) return `${Math.round(h)}h ago`;
  const d = h / 24;
  if (d < 30) return `${Math.round(d)}d ago`;
  const mo = d / 30;
  if (mo < 12) return `${Math.round(mo)}mo ago`;
  return `${Math.round(mo / 12)}y ago`;
}

// "2026-07-16 18:20" — the stored ISO string, trimmed to the minute.
export const absTs = (ts) => (ts ? ts.slice(0, 16).replace("T", " ") : "—");

// -- trace identity ------------------------------------------------------------

// Model identity within a config's color: fixed map, not an arrival-order
// counter, so the same model always gets the same dash across configs.
// Same-model sweep points within a config render alike — hover's r_o
// distinguishes them.
export const MODEL_DASH = { analytical: "solid", fem: "dash" };

// Overlay legend identity: motor + model, plus r_outer so sweep points
// (same motor name, same model) stay distinguishable.
export function traceName(result) {
  const motor = result.config?.motor ?? {};
  const base = `${motor.name ?? "?"} · ${result.model}`;
  const rOuter = motor.geometry?.r_outer;
  return typeof rOuter === "number"
    ? `${base} · r_o ${(rOuter * 1000).toFixed(2)}` : base;
}
