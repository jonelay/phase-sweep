// HTTP data provider. Implements the DataProvider interface:
//   fetch(endpoint, params) -> data      GET with query params
//   subscribe(channel, cb) -> unsubscribe (no push channel; returns no-op)
//   connected -> boolean                 (true; HTTP is stateless)
// Plus post/put/del for mutation endpoints.

async function request(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail ?? detail; } catch { /* not JSON */ }
    throw new Error(`${r.status}: ${detail}`);
  }
  return r.status === 204 ? null : r.json();
}

export function buildUrl(base, endpoint, params) {
  const url = new URL(endpoint, base);
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v != null) url.searchParams.set(k, String(v));
    }
  }
  return url.toString();
}

export class FetchProvider {
  constructor(base = globalThis.location?.origin ?? "http://localhost/", { headers = {} } = {}) {
    this.base = base.endsWith("/") ? base : `${base}/`;
    this.headers = headers;
  }

  get connected() { return true; }

  fetch(endpoint, params) {
    return request(buildUrl(this.base, endpoint, params), { headers: this.headers });
  }

  #body(method, endpoint, body) {
    return request(buildUrl(this.base, endpoint), {
      method,
      headers: { "Content-Type": "application/json", ...this.headers },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
  }

  post(endpoint, body) { return this.#body("POST", endpoint, body); }
  put(endpoint, body) { return this.#body("PUT", endpoint, body); }
  del(endpoint) { return this.#body("DELETE", endpoint); }

  subscribe() { return () => {}; }
}
