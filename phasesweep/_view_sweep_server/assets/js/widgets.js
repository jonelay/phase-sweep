// Plain-DOM widgets (spec/widgets.toml). Pure builders: data in, nodes
// out; theming rides the CSS classes. No Plotly here.

const el = (tag, className, text) => {
  const n = document.createElement(tag);
  if (className) n.className = className;
  if (text != null) n.textContent = text;
  return n;
};

export const NULL_VALUE = "—";

// Compact KPI row. items: [{ label, value }] with pre-formatted strings;
// a null/undefined value renders the null glyph. Rebuilds host in place.
export function statStrip(host, items, { loading = false } = {}) {
  host.classList.add("vs-stat-strip");
  host.replaceChildren();
  for (const { label, value } of items) {
    const tile = el("div", "vs-stat-tile");
    if (loading) tile.classList.add("is-loading");
    tile.append(
      el("span", "vs-stat-label", label),
      el("span", "vs-stat-value", value ?? NULL_VALUE),
    );
    host.append(tile);
  }
  return host;
}

// `model vN · YYYY-MM-DD HH:MM · id[:8]`; "v?" when the record predates
// version stamping. Missing fields collapse rather than print "undefined".
export function provenanceText({ model, model_version, timestamp, result_id } = {}) {
  const ts = (timestamp ?? "").slice(0, 16).replace("T", " ");
  const v = model_version != null ? `v${model_version}` : "v?";
  return [model ? `${model} ${v}` : null, ts || null, (result_id ?? "").slice(0, 8) || null]
    .filter(Boolean).join(" · ");
}

export function provenance(host, result) {
  host.classList.add("vs-provenance");
  host.textContent = result ? provenanceText(result) : "";
  return host;
}

export function badge(status, text = status) {
  const b = el("span", "vs-badge", text);
  b.dataset.status = status;
  return b;
}

export function progress(fraction) {
  const track = el("div", "vs-progress-track");
  const fill = el("div", "vs-progress-fill");
  fill.style.width = `${Math.max(0, Math.min(1, fraction)) * 100}%`;
  track.append(fill);
  return track;
}

// Dot + label; returns the node with a `setState` method.
export function connectionIndicator(state = "connecting") {
  const node = el("span", "vs-conn");
  const dot = el("span", "vs-conn-dot");
  const label = el("span", "vs-conn-label", state);
  node.append(dot, label);
  node.dataset.state = state;
  node.setState = (s) => {
    node.dataset.state = s;
    label.textContent = s;
  };
  return node;
}

export function emptyHint(text) {
  return el("p", "vs-empty-hint", text);
}

export function chip({ label, color, onRemove }) {
  const c = el("span", "vs-chip");
  if (color) {
    const dot = el("span", "vs-chip-dot");
    dot.style.background = color;
    c.append(dot);
  }
  c.append(document.createTextNode(label));
  if (onRemove) {
    const x = el("button", "vs-chip-x", "×");
    x.title = `remove ${label}`;
    x.addEventListener("click", onRemove);
    c.append(x);
  }
  return c;
}
