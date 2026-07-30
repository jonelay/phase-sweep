// Shared plain-DOM widgets for result panels: stat tiles
// and provenance stamps. No plotly — theming rides the CSS tokens.

// Compact KPI row: small uppercase label over tabular numeral.
// items: [{ label, value }] strings — formatting stays at the
// call site. Rebuilds host in place so panels can re-render per result.
export function statStrip(host, items) {
  host.replaceChildren();
  for (const { label, value } of items) {
    const tile = document.createElement("div");
    tile.className = "stat-tile";
    const l = document.createElement("span");
    l.className = "stat-label";
    l.textContent = label;
    const v = document.createElement("span");
    v.className = "stat-value";
    v.textContent = value;
    tile.append(l, v);
    host.append(tile);
  }
}

// Provenance stamp for a routed full result: `model vN · timestamp ·
// result_id[:8]`. model_version is the save-time registry stamp;
// the read side filters stale records, so a visible
// result's stamp always matches the live registry — "v?" means a
// record served by a server predating the stamp.
export function provenanceText(result) {
  const ts = (result.timestamp ?? "").slice(0, 16).replace("T", " ");
  const v = result.model_version != null ? `v${result.model_version}` : "v?";
  return `${result.model} ${v} · ${ts} · ${(result.result_id ?? "").slice(0, 8)}`;
}
