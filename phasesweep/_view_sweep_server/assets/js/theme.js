// Palette loader and light/dark controller.
//
// Two ways to get token values onto the page:
//   1. Pure CSS: load spec/palettes/<name>.css and only `data-theme` needs
//      setting. Call initTheme({ palette }) with the parsed palette JSON
//      anyway so series slots, sequential ramps, and events work.
//   2. JS-applied: pass { palette } (parsed JSON) with applyInline: true
//      and every token is set as an inline custom property on the root.
//
// Series slots (--vs-series-1..8) are always applied inline because the
// user can switch the categorical set at runtime; inline wins over the
// palette stylesheet.
//
// Events (spec/events.toml): vs:theme-change, vs:palette-change,
// vs:sequential-change, dispatched on `document`.

export const SERIES_SLOTS = 8;

const emit = (name, detail) =>
  document.dispatchEvent(new CustomEvent(`vs:${name}`, { detail }));

// Inline <script> for <head>, before the stylesheet, so the first paint
// already has the stored variant. Pass the same storage key and default
// variant you give initTheme.
export function antiFlashSnippet(storageKey = "vs-theme", defaultVariant = "light") {
  return `document.documentElement.dataset.theme=localStorage.getItem(${JSON.stringify(storageKey)})||${JSON.stringify(defaultVariant)};`;
}

export async function loadPalette(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`palette ${url}: ${r.status}`);
  return r.json();
}

// Read a resolved token from the root element (after CSS cascade).
export const token = (name, root = document.documentElement) =>
  getComputedStyle(root).getPropertyValue(`--vs-${name}`).trim();

const storage = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* private mode */ } },
};

export function initTheme({
  palette,
  root = document.documentElement,
  applyInline = false,
  storageKey = "vs-theme",
  categoricalKey = "vs-categorical",
  sequentialKey = "vs-sequential",
  prefix = "--vs-",
} = {}) {
  if (!palette?.meta) throw new Error("initTheme: palette JSON required");
  const meta = palette.meta;

  const pick = (stored, allowed, fallback) => (allowed.includes(stored) ? stored : fallback);
  let variant = pick(storage.get(storageKey), meta.variants, meta.default_variant);
  let categorical = pick(storage.get(categoricalKey), Object.keys(palette.categorical), meta.default_categorical);
  let sequential = pick(storage.get(sequentialKey), Object.keys(palette.sequential), meta.default_sequential);

  // live array: consumers read series[slot] on every retheme
  const series = [];

  const applySeries = () => {
    const colors = palette.categorical[categorical].colors;
    series.length = 0;
    for (let i = 0; i < SERIES_SLOTS; i++) {
      series.push(colors[i % colors.length]);
      root.style.setProperty(`${prefix}series-${i + 1}`, series[i]);
    }
  };

  const applyVariant = () => {
    root.dataset.theme = variant;
    if (applyInline) {
      for (const [k, v] of Object.entries(palette[variant])) {
        if (!k.startsWith("series-")) root.style.setProperty(`${prefix}${k}`, v);
      }
    }
  };

  applyVariant();
  applySeries();

  const ctl = {
    palette,
    series,
    get variant() { return variant; },
    get categorical() { return categorical; },
    get sequential() { return sequential; },
    get isDark() { return variant === "dark"; },
    token: (name) => token(name, root),

    setVariant(next) {
      if (!meta.variants.includes(next) || next === variant) return variant;
      variant = next;
      storage.set(storageKey, variant);
      applyVariant();
      emit("theme-change", { theme: variant });
      return variant;
    },

    toggle() {
      const i = meta.variants.indexOf(variant);
      return ctl.setVariant(meta.variants[(i + 1) % meta.variants.length]);
    },

    setCategorical(id) {
      if (!palette.categorical[id] || id === categorical) return categorical;
      categorical = id;
      storage.set(categoricalKey, id);
      applySeries();
      emit("palette-change", { palette: meta.name, categorical: id });
      return categorical;
    },

    setSequential(id) {
      if (!palette.sequential[id] || id === sequential) return sequential;
      sequential = id;
      storage.set(sequentialKey, id);
      emit("sequential-change", { sequential: id });
      return sequential;
    },

    // Plotly colorscale for the current ramp and variant. `accent` and
    // any other from_vars ramp read live tokens so a theme switch re-hues.
    sequentialStops() {
      const ramp = palette.sequential[sequential];
      if (ramp.from_vars) {
        const n = ramp.from_vars.length;
        return ramp.from_vars.map((v, i) => [n === 1 ? 0 : i / (n - 1), token(v, root)]);
      }
      return ramp[variant] ?? ramp[meta.default_variant];
    },

    // [{ id, label }] for building selectors
    categoricalOptions: () =>
      Object.entries(palette.categorical).map(([id, c]) => ({ id, label: c.label })),
    sequentialOptions: () =>
      Object.entries(palette.sequential).map(([id, s]) => ({ id, label: s.label })),
  };
  return ctl;
}

// Wire a button to ctl.toggle() with a label reflecting the current variant.
export function bindThemeToggle(button, ctl, labels = { light: "Light", dark: "Dark" }) {
  const render = () => {
    button.textContent = labels[ctl.variant] ?? ctl.variant;
    button.title = `switch theme (now ${ctl.variant})`;
  };
  render();
  button.addEventListener("click", () => ctl.toggle());
  document.addEventListener("vs:theme-change", render);
  return render;
}

// Fill a <select> from [{id,label}] and wire it to a setter.
export function bindSelect(select, options, current, onChange) {
  select.replaceChildren();
  for (const { id, label } of options) {
    const o = document.createElement("option");
    o.value = id;
    o.textContent = label;
    o.selected = id === current;
    select.append(o);
  }
  select.addEventListener("change", () => onChange(select.value));
}
