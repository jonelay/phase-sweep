// Static-file provider for sites served without a backend (Caddy, GitHub
// Pages, Tauri asset dir). Resolves endpoints through an optional
// manifest, otherwise as paths relative to `base`.
//
//   manifest: { "<endpoint>": "<relative file>", ... }  or a URL to one
//   fetch(endpoint, params) -> parsed JSON (params are ignored; a static
//                              file has one shape)
//   subscribe -> no-op; connected -> true

export function resolveEndpoint(base, endpoint, manifest = null) {
  const rel = manifest?.[endpoint] ?? endpoint;
  return new URL(rel, base).toString();
}

export class StaticProvider {
  constructor(base = globalThis.location?.href ?? "http://localhost/", { manifest = null } = {}) {
    this.base = base;
    this.manifestSource = manifest;
    this.manifest = typeof manifest === "object" ? manifest : null;
    this.cache = new Map();
  }

  get connected() { return true; }

  async #manifest() {
    if (this.manifest || typeof this.manifestSource !== "string") return this.manifest;
    const r = await fetch(new URL(this.manifestSource, this.base));
    if (!r.ok) throw new Error(`manifest ${this.manifestSource}: ${r.status}`);
    this.manifest = await r.json();
    return this.manifest;
  }

  async fetch(endpoint, _params, { cache = true } = {}) {
    const url = resolveEndpoint(this.base, endpoint, await this.#manifest());
    if (cache && this.cache.has(url)) return this.cache.get(url);
    const p = fetch(url).then((r) => {
      if (!r.ok) throw new Error(`${r.status}: ${url}`);
      return r.json();
    });
    if (cache) this.cache.set(url, p);
    return p;
  }

  subscribe() { return () => {}; }
}
