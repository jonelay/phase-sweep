// App chrome: tab router, panel cards, sidebar rail, status bar, toast,
// connection banner. Everything takes explicit elements or selectors;
// nothing here knows a consumer's DOM ids. Events per spec/events.toml.

const emit = (name, detail) =>
  document.dispatchEvent(new CustomEvent(`vs:${name}`, { detail }));

const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
};

const resize = () => window.dispatchEvent(new Event("resize"));

// -- hash routing (pure) ------------------------------------------------------------
// `#/<tab>` or `#/<tab>?k=v`. Unknown or empty hash -> fallback.

export function parseHash(hash, tabIds, fallback = tabIds[0]) {
  const body = (hash ?? "").startsWith("#/") ? hash.slice(2) : "";
  const [tab, query = ""] = body.split("?");
  return {
    tab: tabIds.includes(tab) ? tab : fallback,
    params: Object.fromEntries(new URLSearchParams(query)),
  };
}

export function formatHash(tab, params = {}) {
  const q = new URLSearchParams(params).toString();
  return `#/${tab}${q ? `?${q}` : ""}`;
}

// -- tabs -------------------------------------------------------------------------

// tabs: [{ id, label }]. Builds buttons into `tabBar` and panes into
// `paneHost`; returns { panes, buttons, activate, active }. Roving
// tabindex, arrow keys, optional hash routing, persisted last tab.
//
// adopt: true — discover existing [role=tab] buttons in `tabBar` and
// [role=tabpanel] children in `paneHost` instead of creating them.
// Each tab button must have `data-tab-id` or `aria-controls` pointing
// at the pane id (vs-pane-<id>). The `tabs` array is still required
// for ordering; `label` is ignored in adopt mode.
export function mountTabs({
  tabBar, paneHost, tabs,
  storageKey = "vs-active-tab",
  hashRouting = false,
  paneClass = "vs-tab-pane",
  adopt = false,
}) {
  const ids = tabs.map((t) => t.id);
  const buttons = new Map();
  const panes = new Map();
  let active = null;

  const hashParams = () => hashRouting ? parseHash(location.hash, ids, active).params : {};

  const activate = (id, { silent = false } = {}) => {
    if (!panes.has(id)) return active;
    for (const tid of ids) {
      panes.get(tid).hidden = tid !== id;
      const b = buttons.get(tid);
      b.setAttribute("aria-selected", String(tid === id));
      b.tabIndex = tid === id ? 0 : -1;
    }
    const changed = id !== active;
    active = id;
    buttons.get(id).classList.remove("attention");
    storage.set(storageKey, id);
    if (hashRouting) {
      const params = hashParams();
      const target = formatHash(id, params);
      if (location.hash !== target) history.replaceState(null, "", target);
    }
    resize();
    if (changed && !silent) emit("tab-change", { tab_id: id, params: hashParams() });
    return active;
  };

  tabBar.classList.add("vs-tabs");
  tabBar.setAttribute("role", "tablist");

  if (adopt) {
    const existingButtons = tabBar.querySelectorAll('[role="tab"]');
    const btnById = new Map();
    for (const btn of existingButtons) {
      const tid = btn.dataset.tabId ?? btn.getAttribute("aria-controls")?.replace(/^vs-pane-/, "");
      if (tid) btnById.set(tid, btn);
    }
    for (const [i, { id }] of tabs.entries()) {
      const btn = btnById.get(id);
      if (!btn) throw new Error(`mountTabs adopt: no button for tab "${id}"`);
      btn.id ||= `vs-tab-${id}`;
      btn.setAttribute("aria-controls", `vs-pane-${id}`);
      btn.title ||= `${btn.textContent.trim()} — press ${i + 1}`;
      btn.addEventListener("click", () => activate(id));
      buttons.set(id, btn);
      const pane = paneHost.querySelector(`#vs-pane-${id}`)
        ?? paneHost.querySelector(`[data-tab-id="${id}"]`);
      if (!pane) throw new Error(`mountTabs adopt: no pane for tab "${id}"`);
      pane.id ||= `vs-pane-${id}`;
      pane.classList.add(paneClass);
      pane.setAttribute("role", "tabpanel");
      pane.setAttribute("aria-labelledby", btn.id);
      btn.setAttribute("aria-controls", pane.id);
      panes.set(id, pane);
    }
  } else {
    for (const [i, { id, label }] of tabs.entries()) {
      const btn = document.createElement("button");
      btn.setAttribute("role", "tab");
      btn.id = `vs-tab-${id}`;
      btn.setAttribute("aria-controls", `vs-pane-${id}`);
      btn.textContent = label;
      btn.title = `${label} — press ${i + 1}`;
      btn.addEventListener("click", () => activate(id));
      tabBar.append(btn);
      buttons.set(id, btn);
      const pane = document.createElement("div");
      pane.className = paneClass;
      pane.id = `vs-pane-${id}`;
      pane.setAttribute("role", "tabpanel");
      pane.setAttribute("aria-labelledby", btn.id);
      pane.hidden = true;
      paneHost.append(pane);
      panes.set(id, pane);
    }
  }

  tabBar.addEventListener("keydown", (ev) => {
    let i = ids.indexOf(active);
    if (ev.key === "ArrowRight") i = (i + 1) % ids.length;
    else if (ev.key === "ArrowLeft") i = (i + ids.length - 1) % ids.length;
    else if (ev.key === "Home") i = 0;
    else if (ev.key === "End") i = ids.length - 1;
    else return;
    ev.preventDefault();
    activate(ids[i]);
    buttons.get(ids[i]).focus();
  });

  if (hashRouting) {
    window.addEventListener("hashchange", () => activate(parseHash(location.hash, ids, active).tab));
  }

  const initial = hashRouting && location.hash.startsWith("#/")
    ? parseHash(location.hash, ids, storage.get(storageKey) ?? ids[0]).tab
    : (panes.has(storage.get(storageKey)) ? storage.get(storageKey) : ids[0]);
  activate(initial, { silent: true });

  return {
    panes, buttons, activate,
    get active() { return active; },
    ids,
    markAttention(id) {
      if (id && id !== active) buttons.get(id)?.classList.add("attention");
    },
  };
}

// -- panel cards -----------------------------------------------------------------

// Returns { card, body, headerSlot, setCollapsed, isCollapsed }. The
// caller appends `card` to a pane or sidebar slot.
export function mountPanel({
  id, title,
  collapsible = true, startCollapsed = false,
  storageKey = `vs-collapsed:${id}`,
  draggable = false,
  onReveal = null,
}) {
  const card = document.createElement("section");
  card.className = "vs-panel";
  card.dataset.panel = id;
  const header = document.createElement("header");
  const h2 = document.createElement("h2");
  h2.textContent = title;
  const headerSlot = document.createElement("div");
  headerSlot.className = "vs-panel-slot";
  const body = document.createElement("div");
  body.className = "vs-panel-body";

  if (draggable) {
    const handle = document.createElement("span");
    handle.className = "vs-drag-handle";
    handle.textContent = "⋮⋮";
    handle.title = "drag to reorder";
    header.append(handle);
  }
  header.append(h2, headerSlot);

  const isCollapsed = () => card.classList.contains("collapsed");
  let btn = null;
  const setCollapsed = (collapsed, { persist = false } = {}) => {
    const was = isCollapsed();
    card.classList.toggle("collapsed", collapsed);
    if (btn) {
      btn.textContent = collapsed ? "+" : "−";
      btn.title = collapsed ? "expand" : "collapse";
    }
    if (persist) storage.set(storageKey, collapsed ? "1" : "0");
    if (was && !collapsed) {
      resize();
      onReveal?.();
    }
    if (was !== collapsed) emit("panel-toggle", { panel_id: id, collapsed });
  };

  if (collapsible) {
    btn = document.createElement("button");
    btn.className = "vs-panel-collapse";
    btn.addEventListener("click", () => setCollapsed(!isCollapsed(), { persist: true }));
    headerSlot.after(btn);
    const stored = storage.get(storageKey);
    setCollapsed(stored === null ? startCollapsed : stored === "1");
  }

  card.append(header, body);
  return { card, body, headerSlot, setCollapsed, isCollapsed };
}

// One maximized card per pane; Escape restores (bind via bindEscapeRestore).
export function toggleMaximize(card, pane = card.parentElement) {
  const on = !card.classList.contains("maximized");
  if (on) pane.querySelector(".vs-panel.maximized")?.classList.remove("maximized");
  card.classList.toggle("maximized", on);
  pane.classList.toggle("has-max", on);
  resize();
  emit("panel-maximize", { panel_id: card.dataset.panel ?? "", maximized: on });
  return on;
}

export function bindEscapeRestore({ isTextTarget = defaultIsTextTarget, guard = () => false } = {}) {
  document.addEventListener("keydown", (ev) => {
    if (ev.key !== "Escape" || guard() || isTextTarget(ev.target)) return;
    const card = document.querySelector(".vs-tab-pane:not([hidden]) .vs-panel.maximized");
    if (card) toggleMaximize(card, card.parentElement);
  });
}

export function defaultIsTextTarget(t) {
  return t instanceof HTMLElement
    && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(t.tagName));
}

// -- sidebar rail ---------------------------------------------------------------------

export function initRailToggle({ layout, button, storageKey = "vs-rail" }) {
  const apply = (collapsed, { persist = false } = {}) => {
    layout.classList.toggle("rail-collapsed", collapsed);
    button.textContent = collapsed ? "»" : "«";
    button.title = collapsed ? "expand sidebar" : "collapse sidebar";
    if (persist) storage.set(storageKey, collapsed ? "1" : "0");
    resize();
    emit("sidebar-toggle", { collapsed });
  };
  apply(storage.get(storageKey) === "1");
  button.addEventListener("click", () =>
    apply(!layout.classList.contains("rail-collapsed"), { persist: true }));
  return apply;
}

// -- status bar -------------------------------------------------------------------------

export function mountStatusBar(container) {
  container.classList.add("vs-statusbar");
  const conn = document.createElement("span");
  conn.className = "vs-conn";
  conn.dataset.state = "connecting";
  const dot = document.createElement("span");
  dot.className = "vs-conn-dot";
  const label = document.createElement("span");
  label.className = "vs-conn-label";
  label.textContent = "connecting";
  conn.append(dot, label);
  const jobs = document.createElement("span");
  jobs.textContent = "0 jobs running";
  const last = document.createElement("span");
  last.textContent = "last result —";
  container.append(conn, jobs, last);

  const bar = {
    setConnection(state) {
      conn.dataset.state = state;
      label.textContent = state;
    },
    setJobCount(n) {
      jobs.textContent = `${n} job${n === 1 ? "" : "s"} running`;
    },
    stampResult(date = new Date()) {
      last.textContent = `last result ${date.toLocaleTimeString([], { hour12: false })}`;
    },
  };
  document.addEventListener("vs:connection-change", (e) => bar.setConnection(e.detail.state));
  document.addEventListener("vs:result-routed", () => bar.stampResult());
  return bar;
}

// Latching banner: shown on a real drop (after a first connect), held
// through reconnect attempts, cleared on connect.
export function bindConnectionBanner(banner) {
  let everConnected = false;
  document.addEventListener("vs:connection-change", (e) => {
    const s = e.detail.state;
    if (s === "connected") { everConnected = true; banner.hidden = true; }
    else if (s === "disconnected" && everConnected) banner.hidden = false;
  });
}

// -- toast --------------------------------------------------------------------------------

let toastEl = null;
let toastTimer = 0;
export function toast(text, { duration = 2500 } = {}) {
  if (!toastEl) {
    toastEl = document.createElement("div");
    toastEl.className = "vs-toast";
    toastEl.setAttribute("role", "status");
    document.body.append(toastEl);
  }
  toastEl.textContent = text;
  toastEl.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toastEl.classList.remove("show"), duration);
  return toastEl;
}

// Title badge counting terminal jobs while the tab is hidden.
export function bindHiddenCounter(baseTitle = document.title) {
  let n = 0;
  document.addEventListener("visibilitychange", () => {
    if (!document.hidden) { n = 0; document.title = baseTitle; }
  });
  const bump = () => {
    if (document.hidden) document.title = `(${++n}) ${baseTitle}`;
  };
  document.addEventListener("vs:job-complete", bump);
  document.addEventListener("vs:job-failed", bump);
}
