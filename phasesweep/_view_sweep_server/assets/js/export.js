// Composite PNG export: every visible Plotly plot in a container stacked
// onto one canvas with a header and per-panel provenance, so the export
// stays attributable. Selectors are configurable; defaults follow
// spec/widgets.toml. Assumes a global Plotly.

const loadImage = (src) => new Promise((res, rej) => {
  const im = new Image();
  im.onload = () => res(im);
  im.onerror = rej;
  im.src = src;
});

const cssVar = (name) =>
  getComputedStyle(document.documentElement).getPropertyValue(`--vs-${name}`).trim();

export const DEFAULT_SELECTORS = {
  panel: ".vs-panel",
  collapsedClass: "collapsed",
  plot: ".js-plotly-plot",
  title: "h2",
  provenance: ".vs-provenance",
};

// Collect exportable plots under `container`. Exported for tests.
export function collectPlots(container, selectors = DEFAULT_SELECTORS) {
  const s = { ...DEFAULT_SELECTORS, ...selectors };
  const items = [];
  for (const card of container.querySelectorAll(s.panel)) {
    if (card.classList.contains(s.collapsedClass)) continue;
    const el = card.querySelector(s.plot);
    if (!el || el.style.display === "none" || !el.clientWidth) continue;
    items.push({
      el,
      title: card.querySelector(s.title)?.textContent ?? "",
      prov: card.querySelector(s.provenance)?.textContent ?? "",
    });
  }
  return items;
}

export function exportFilename(brand, scope, stamp) {
  const slug = (s) => (s || "view").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return `${slug(brand)}_${slug(scope)}_${stamp.replace(/[: ]/g, "-")}.png`;
}

// Returns the Blob (and triggers a download unless download: false).
export async function exportComposite({
  container,
  brand = "dashboard",
  scope = "",
  subtitle = "",
  selectors = {},
  scale = 2,
  notify = () => {},
  download = true,
} = {}) {
  const items = collectPlots(container, selectors);
  if (!items.length) { notify("no plots to export"); return null; }
  notify("rendering PNG…");
  let imgs;
  try {
    const urls = await Promise.all(items.map((it) => Plotly.toImage(it.el, {
      format: "png", scale, width: it.el.clientWidth, height: it.el.clientHeight,
    })));
    imgs = await Promise.all(urls.map(loadImage));
  } catch (e) {
    console.warn("export failed", e);
    notify("export failed");
    return null;
  }

  const ink = cssVar("text");
  const muted = cssVar("text-muted");
  const fontUi = cssVar("font-ui") || "system-ui, sans-serif";
  const fontMono = cssVar("font-mono") || "ui-monospace, monospace";
  const pad = 16 * scale;
  const headH = 48 * scale;
  const titleH = 18 * scale;
  const provH = 15 * scale;
  const gap = 12 * scale;

  const canvas = document.createElement("canvas");
  canvas.width = Math.max(...imgs.map((im) => im.width)) + 2 * pad;
  canvas.height = headH + imgs.reduce((s, im, i) =>
    s + titleH + (items[i].prov ? provH : 0) + im.height + gap, 0) + pad;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = getComputedStyle(document.body).backgroundColor;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.textBaseline = "top";

  const stamp = new Date().toISOString().slice(0, 19).replace("T", " ");
  ctx.fillStyle = ink;
  ctx.font = `600 ${14 * scale}px ${fontUi}`;
  ctx.fillText([brand, scope, subtitle].filter(Boolean).join(" · "), pad, pad);
  ctx.fillStyle = muted;
  ctx.font = `${11 * scale}px ${fontUi}`;
  ctx.fillText(stamp, pad, pad + 21 * scale);

  let y = headH;
  items.forEach((it, i) => {
    ctx.fillStyle = ink;
    ctx.font = `600 ${12 * scale}px ${fontUi}`;
    ctx.fillText(it.title, pad, y);
    y += titleH;
    if (it.prov) {
      ctx.fillStyle = muted;
      ctx.font = `${10 * scale}px ${fontMono}`;
      ctx.fillText(it.prov, pad, y);
      y += provH;
    }
    ctx.drawImage(imgs[i], pad, y);
    y += imgs[i].height + gap;
  });

  const blob = await new Promise((res) => canvas.toBlob(res, "image/png"));
  if (!blob) { notify("export failed"); return null; }
  if (download) {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = exportFilename(brand, scope, stamp);
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    notify("PNG downloaded");
  }
  return blob;
}
