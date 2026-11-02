// Design-space plot: every design an MCP agent's jobs have
// produced, on two chosen quantities — swept dimensions or any model's
// headline metric — joined in the order the agent reached them. Polls
// GET /api/mcp/designs with a version token, so points arrive while a
// sweep runs. Hover joins the linked highlight bus; click routes the
// y-metric's result to the other panels.

const POLL_MS = 1500;
const GEO = new Set(["r_outer", "L_stk", "airgap", "back_iron_thickness"]);

let app = null;
let plotEl = null;
let hintEl = null;
let xSel = null;
let ySel = null;
let timer = null;
let version = null;
let data = { units: {}, designs: [] };
let picked = { x: null, y: null }; // user choices survive refreshes
let linked = null; // design id spotlighted from another panel
let bound = false;

const isNum = (v) => typeof v === "number" && Number.isFinite(v);
// geometry is stored in m; shown in mm like the sweep table
const scale = (col) => (GEO.has(col) ? 1000 : 1);
const unitOf = (col) => (GEO.has(col) ? "mm" : data.units[col] ?? "");
const label = (col) => {
  const u = unitOf(col);
  return u && u !== "1" ? `${col} [${u}]` : col;
};

// columns with at least two distinct finite values across designs:
// dimensions first, most varied first (the swept axis with the most
// steps is the natural x); then metrics in the server's headline order
function columns() {
  const seen = new Map();
  for (const d of data.designs) {
    for (const [k, v] of Object.entries(d.values)) {
      if (!isNum(v)) continue;
      if (!seen.has(k)) seen.set(k, new Set());
      seen.get(k).add(v);
    }
  }
  const varied = [...seen].filter(([, s]) => s.size >= 2);
  const geo = varied.filter(([k]) => GEO.has(k)).sort((a, b) => b[1].size - a[1].size);
  return [...geo, ...varied.filter(([k]) => !GEO.has(k))].map(([k]) => k);
}

function syncSelects(cols) {
  const metrics = cols.filter((c) => !GEO.has(c));
  const x = cols.includes(picked.x) ? picked.x : (cols.find((c) => GEO.has(c)) ?? metrics[0]);
  const y = cols.includes(picked.y) ? picked.y : (metrics.find((c) => c !== x) ?? cols.find((c) => c !== x));
  for (const [sel, val] of [[xSel, x], [ySel, y]]) {
    sel.replaceChildren(...cols.map((c) => new Option(label(c), c)));
    sel.value = val ?? "";
  }
}

// the result behind a point: the y metric's model, else any
function resultFor(d, ycol) {
  const model = ycol?.includes(".") ? ycol.split(".")[0] : null;
  return d.result_ids[model] ?? Object.values(d.result_ids)[0] ?? null;
}

function render() {
  if (!plotEl) return;
  const cols = columns();
  syncSelects(cols);
  const xc = xSel.value;
  const yc = ySel.value;
  const pts = data.designs.filter((d) => isNum(d.values[xc]) && isNum(d.values[yc]));
  hintEl.style.display = pts.length >= 2 ? "none" : "";
  plotEl.style.display = pts.length >= 2 ? "" : "none";
  if (pts.length < 2) return;

  const { plot, theme } = app;
  const xs = pts.map((d) => d.values[xc] * scale(xc));
  const ys = pts.map((d) => d.values[yc] * scale(yc));
  const text = pts.map((d) => [
    `#${d.order + 1} ${d.motor}${d.point_idx != null ? ` · point ${d.point_idx}` : ""}`,
    `job ${d.job_id}`,
    ...(d.caveats.length ? [`⚠ ${[...new Set(d.caveats)].join(", ")}`] : []),
  ].join("<br>"));
  const css = getComputedStyle(document.documentElement);
  const warnColor = css.getPropertyValue("--vs-status-warning").trim() || "#d08c00";
  const pathColor = css.getPropertyValue("--vs-text-muted").trim() || "#888";
  const traces = [
    { // the agent's path, in exploration order
      x: xs, y: ys, mode: "lines", hoverinfo: "skip", showlegend: false,
      line: { color: pathColor, width: 1, dash: "dot" },
    },
    {
      x: xs, y: ys, mode: "markers", text, hovertemplate: "%{text}<extra></extra>",
      customdata: pts.map((d) => resultFor(d, yc)),
      showlegend: false,
      marker: {
        size: pts.map((_, i) => (i === pts.length - 1 ? 13 : 9)),
        color: pts.map((d) => d.order),
        colorscale: theme.sequentialStops(),
        colorbar: { title: { text: "order", side: "right" }, thickness: 10 },
        symbol: pts.map((d) => (d.warn ? "diamond" : "circle")),
        line: { width: pts.map((d) => (d.warn ? 2 : 0)), color: warnColor },
      },
    },
  ];
  const hi = pts.find((d) => d.id === linked);
  if (hi) {
    traces.push({
      x: [hi.values[xc] * scale(xc)], y: [hi.values[yc] * scale(yc)],
      mode: "markers", hoverinfo: "skip", showlegend: false,
      marker: { size: 20, color: "rgba(0,0,0,0)", line: { width: 2, color: theme.series[0] } },
    });
  }
  Plotly.react(plotEl, traces, plot.layout({
    height: 320,
    hovermode: "closest", // the shared layout unifies on x; a scatter needs the point
    xaxis: plot.axisStyle(label(xc)),
    yaxis: plot.axisStyle(label(yc)),
    margin: { l: 60, r: 12, t: 12, b: 45 },
  }), plot.config);
  if (!bound) bindPlotEvents();
}

// Plotly attaches .on() to the element at its first plot
function bindPlotEvents() {
  bound = true;
  plotEl.on("plotly_hover", (ev) => {
    const rid = ev.points?.[0]?.customdata;
    if (rid) app.highlightResult(rid, "design_space");
  });
  plotEl.on("plotly_unhover", () => app?.highlightResult(null, "design_space"));
  plotEl.on("plotly_click", (ev) => {
    const rid = ev.points?.[0]?.customdata;
    if (rid) app.routeResult(rid);
  });
}

async function poll() {
  let res;
  try {
    res = await app.api.get(`/api/mcp/designs${version ? `?version=${encodeURIComponent(version)}` : ""}`);
  } catch {
    return; // the activity panel reports reachability
  }
  if (res.designs === null) return;
  version = res.version;
  data = { units: res.units, designs: res.designs };
  render();
}

function schedule() {
  clearInterval(timer);
  timer = setInterval(() => {
    if (document.visibilityState === "visible") poll();
  }, POLL_MS);
}

export default {
  name: "design_space",
  label: "Design Space",
  accepts: [],

  init(container, appCtx) {
    app = appCtx;
    const bar = document.createElement("div");
    bar.className = "design-space-bar";
    xSel = document.createElement("select");
    ySel = document.createElement("select");
    xSel.title = "x axis";
    ySel.title = "y axis";
    for (const [sel, key] of [[xSel, "x"], [ySel, "y"]]) {
      sel.className = "vs-select";
      sel.addEventListener("change", () => {
        picked[key] = sel.value;
        render();
      });
    }
    const xl = document.createElement("label");
    xl.append("x ", xSel);
    const yl = document.createElement("label");
    yl.append("y ", ySel);
    bar.append(xl, yl);

    hintEl = document.createElement("div");
    hintEl.className = "vs-empty-hint";
    hintEl.textContent = "Designs an agent explores appear here once two of them share a varying quantity.";
    plotEl = document.createElement("div");
    plotEl.style.display = "none";
    container.append(bar, hintEl, plotEl);

    app.onHighlight((id, origin) => {
      if (origin === "design_space") return;
      const d = id ? data.designs.find((x) => Object.values(x.result_ids).includes(id)) : null;
      const next = d?.id ?? null;
      if (next !== linked) {
        linked = next;
        render();
      }
    });
    poll();
    schedule();
  },

  update() {},
  retheme: () => render(),

  destroy() {
    clearInterval(timer);
    timer = null;
    version = null;
    linked = null;
    bound = false;
    data = { units: {}, designs: [] };
    picked = { x: null, y: null };
    app = plotEl = hintEl = xSel = ySel = null;
  },
};
