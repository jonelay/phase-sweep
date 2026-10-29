// Tauri provider (interface stub; Phase 5 fills it in with a consumer).
//
//   fetch(endpoint, params) -> invoke(command, { endpoint, params })
//   subscribe(event, cb)    -> Tauri event listener; returns unlisten
//   connected               -> true once window.__TAURI__ is present
//
// Command name is configurable so the Rust side can route however it
// likes; default routes every fetch through one `vs_fetch` command.

export function tauriApi() {
  return globalThis.window?.__TAURI__ ?? null;
}

export class TauriProvider {
  constructor({ command = "vs_fetch", api = tauriApi() } = {}) {
    this.command = command;
    this.api = api;
  }

  get connected() { return !!this.api; }

  #require() {
    if (!this.api) throw new Error("TauriProvider: window.__TAURI__ not present");
    return this.api;
  }

  fetch(endpoint, params = {}) {
    return this.#require().core.invoke(this.command, { endpoint, params });
  }

  subscribe(event, cb) {
    const p = this.#require().event.listen(event, (e) => cb(e.payload));
    let unlisten = null;
    p.then((fn) => { unlisten = fn; });
    return () => p.then((fn) => (unlisten ?? fn)());
  }
}
