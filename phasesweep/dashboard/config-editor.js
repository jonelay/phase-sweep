// Config editor panel: parameter form over the raw TOML
// structure, model selector, sweep builder. Saves go to the server's
// user-configs directory (PUT /api/configs/{name}); anchor configs in
// motors/ are read-only and save under a new name.

// Geometry lengths display in mm (matching the plots) but the TOML stays
// in metres: `scale` converts raw -> display, collect() divides back.
const MM = { unit: "mm", scale: 1000 };

const SECTION_FIELDS = {
  motor: {
    name: { str: true },
    topology: { options: ["inrunner", "outrunner"] },
  },
  circuit: {
    n_p: { int: true },
    R_s: { unit: "Ω" },
    L_d: { unit: "H" },
    L_q: { unit: "H" },
    psi_f: { unit: "Wb" },
    J: { unit: "kg·m²" },
    I_rated: { unit: "A pk" },
    I_rated_rms: { unit: "A rms" },
  },
  winding: {
    N: { int: true, unit: "turns" },
    k_w: {},
    coils_series: { int: true },
    n_slots: { int: true },
  },
  geometry: {
    r_outer: MM,
    r_stator: MM,
    r_magnet: MM,
    r_rotor: MM,
    r_inner: MM,
    r_ag: MM,
    L_stk: MM,
    n_slots: { int: true },
    slot_depth: MM,
    slot_width_ratio: {},
    slot_opening_width: MM,
    slot_opening_ratio: {},
    alpha_p: {},
    back_iron_thickness: MM,
  },
  materials: {
    B_rem: { unit: "T" },
    mu_r_fe: {},
    mu_r_pm: {},
    alpha_Br: { unit: "1/K" },
    B_knee: { unit: "T" },
    alpha_B_knee: { unit: "1/K" },
  },
  drive: {
    U_DC: { unit: "V" },
    MAX_I_S: { unit: "A" },
    W_REF: { unit: "rad/s" },
    I_LIMIT: { unit: "A" },
  },
  thermal: {
    winding_temp_limit: { unit: "°C" },
    ambient_temp: { unit: "°C" },
    r_th: { unit: "K/W" },
    thermal_time_constant: { unit: "s" },
    magnet_temp: { unit: "°C" },
    insulation_class: { str: true },
  },
  iron: {
    k_h: {},
    k_e: {},
    alpha_fe: {},
    m_core: { unit: "kg" },
    B_core: { unit: "T" },
  },
};

// all sweepable fields are lengths — entered in mm like the form above,
// converted to the metres SweepAxis expects on submit
const SWEEP_FIELDS = ["r_outer", "L_stk", "r_ag", "back_iron_thickness"];
const SWEEP_STRATEGIES = ["proportional", "fixed_gap"];

let app = null;
let rawConfig = null; // parsed TOML of the loaded config, section layout intact
let formEl, saveNameEl, saveMsgEl, modelListEl, axesEl, runMsgEl;
const modelMeta = {}; // keyed by model key, holds needs_params from /api/models

// -- parameter form --------------------------------------------

// trims binary-float noise from scaled values (0.0301*1000 -> "30.1")
const fmt = (v) => String(Number(v.toPrecision(12)));

function fieldRow(section, field, desc, value) {
  const row = document.createElement("label");
  row.className = "cfg-row";
  const name = document.createElement("span");
  name.className = "cfg-name";
  name.textContent = desc.unit ? `${field} [${desc.unit}]` : field;
  name.title = name.textContent; // labels ellipsize in narrow grids
  let input;
  if (desc.options) {
    input = document.createElement("select");
    for (const o of desc.options) input.add(new Option(o, o));
    input.value = value ?? desc.options[0];
  } else {
    input = document.createElement("input");
    input.type = "text";
    input.value = typeof value === "number" && desc.scale
      ? fmt(value * desc.scale) : (value ?? "");
    if (!desc.str) input.placeholder = "—";
  }
  input.dataset.section = section;
  input.dataset.field = field;
  input.dataset.kind = desc.str || desc.options ? "str" : desc.int ? "int" : "num";
  if (desc.scale) input.dataset.scale = desc.scale;
  // collect() only writes back fields the user actually edited, so
  // display rounding can never drift an untouched value in the TOML
  input.dataset.orig = input.value;
  row.append(name, input);
  return row;
}

// the OD/ID geometry spelling (CREATOR) isn't in the catalog — its length
// fields still get the mm treatment instead of falling back to raw metres
const ALT_LENGTHS = new Set([
  "stator_od", "stator_id", "rotor_od", "air_gap", "magnet_width",
  "magnet_thickness", "slot_height", "slot_width", "slot_opening_height",
  "tooth_width", "tooth_height",
]);

function inferDesc(section, field, value) {
  if (section === "geometry" && ALT_LENGTHS.has(field)) return MM;
  if (field.endsWith("_deg")) return { unit: "deg" };
  return typeof value === "string" ? { str: true } : {};
}

function renderForm() {
  formEl.replaceChildren();
  const raw = rawConfig;
  for (const [section, catalog] of Object.entries(SECTION_FIELDS)) {
    const present = raw[section] ?? {};
    const details = document.createElement("details");
    details.className = "cfg-section";
    details.open = Object.keys(present).length > 0;
    const summary = document.createElement("summary");
    summary.textContent = `[${section}]`;
    details.append(summary);
    const grid = document.createElement("div");
    grid.className = "cfg-grid";
    for (const [field, desc] of Object.entries(catalog)) {
      grid.append(fieldRow(section, field, desc, present[field]));
    }
    // fields in the TOML the catalog doesn't know — still editable
    for (const [field, value] of Object.entries(present)) {
      if (!(field in catalog) && typeof value !== "object") {
        grid.append(fieldRow(section, field, inferDesc(section, field, value), value));
      }
    }
    details.append(grid);
    formEl.append(details);
  }
  // sections outside the catalog (e.g. [validation]) pass through untouched
  const passthrough = Object.keys(raw).filter((s) => !(s in SECTION_FIELDS));
  if (passthrough.length) {
    const note = document.createElement("div");
    note.className = "cfg-note";
    note.textContent = `preserved as-is: ${passthrough.map((s) => `[${s}]`).join(" ")}`;
    formEl.append(note);
  }
}

function collect() {
  const out = structuredClone(rawConfig);
  for (const input of formEl.querySelectorAll("[data-section]")) {
    const { section, field, kind, scale, orig } = input.dataset;
    if (input.value === orig) continue; // untouched: keep the raw TOML value
    const text = input.value.trim();
    if (text === "") {
      if (out[section]) delete out[section][field];
      continue;
    }
    let value = text;
    if (kind !== "str") {
      value = Number(text);
      if (Number.isNaN(value)) throw new Error(`[${section}] ${field}: not a number`);
      if (kind === "int" && !Number.isInteger(value)) {
        throw new Error(`[${section}] ${field}: must be an integer`);
      }
      if (scale) value = Number((value / Number(scale)).toPrecision(12)); // 9.7/1000 -> 0.0097 exactly
    }
    (out[section] ??= {})[field] = value;
  }
  for (const section of Object.keys(out)) {
    if (typeof out[section] === "object" && !Object.keys(out[section]).length) {
      delete out[section];
    }
  }
  return out;
}

async function save() {
  saveMsgEl.textContent = "";
  saveMsgEl.classList.remove("ok");
  const name = saveNameEl.value.trim();
  if (!name) {
    saveMsgEl.textContent = "pick a config name";
    return;
  }
  let body;
  try {
    body = collect();
  } catch (e) {
    saveMsgEl.textContent = e.message;
    return;
  }
  try {
    await app.api.put(`/api/configs/${encodeURIComponent(name)}`, body);
  } catch (e) {
    saveMsgEl.textContent = String(e.message ?? e);
    return;
  }
  // reload first — selecting the saved config re-runs load(), which
  // clears the message line; confirm after it settles
  try {
    await app.reloadConfigs(name);
  } catch (e) {
    console.warn("config list reload failed", e);
  }
  saveMsgEl.textContent = `saved ${name}`;
  saveMsgEl.classList.add("ok");
}

// fast config switches can resolve out of order — only the latest load applies
let loadSeq = 0;

async function load(name) {
  const seq = ++loadSeq;
  saveMsgEl.textContent = "";
  saveMsgEl.classList.remove("ok");
  let d;
  try {
    d = await app.api.get(`/api/configs/${encodeURIComponent(name)}/raw`);
  } catch (e) {
    if (seq !== loadSeq) return;
    formEl.replaceChildren();
    saveMsgEl.textContent = `load failed — ${e.message ?? e}`;
    return;
  }
  if (seq !== loadSeq) return;
  rawConfig = d.raw;
  saveNameEl.value = d.editable ? name : `${name}_edit`;
  renderForm();
}

// -- model selector + sweep builder --------------------------------------------

function checkedModels() {
  return [...modelListEl.querySelectorAll("input:checked")].map((i) => i.value);
}

function modelNeedsParams(key) {
  const meta = modelMeta[key];
  return meta && meta.needs_params && meta.needs_params.length > 0;
}

async function loadModels() {
  const models = (await app.api.get("/api/models")).filter((m) => m.kind === "single");
  for (const m of models) {
    modelMeta[m.key] = m;
    const label = document.createElement("label");
    label.className = "model-choice";
    const box = document.createElement("input");
    box.type = "checkbox";
    box.value = m.key;
    box.checked = m.key === "analytical"; // fast exploration default
    const cost = document.createElement("span");
    cost.className = "cost-badge";
    cost.dataset.cost = m.cost;
    cost.textContent = m.cost;
    label.append(box, ` ${m.key} `, cost);
    modelListEl.append(label);
  }
}

function axisRow() {
  const row = document.createElement("div");
  row.className = "axis-row";
  const field = document.createElement("select");
  for (const f of SWEEP_FIELDS) field.add(new Option(f, f));
  const mk = (ph) => {
    const i = document.createElement("input");
    i.type = "text";
    i.placeholder = ph;
    return i;
  };
  const start = mk("start [mm]");
  const stop = mk("stop [mm]");
  const steps = mk("steps");
  const strategy = document.createElement("select");
  for (const s of SWEEP_STRATEGIES) strategy.add(new Option(s, s));
  const rm = document.createElement("button");
  rm.type = "button";
  rm.textContent = "×";
  rm.title = "remove axis";
  rm.addEventListener("click", () => row.remove());
  row.append(field, start, stop, steps, strategy, rm);
  row.axis = () => {
    const num = (input, label) => {
      const v = Number(input.value);
      if (input.value.trim() === "" || Number.isNaN(v)) {
        throw new Error(`sweep ${field.value}: ${label} is not a number`);
      }
      return v;
    };
    const toM = (v) => Number((v / 1000).toPrecision(12));
    return {
      field: field.value,
      start: toM(num(start, "start")),
      stop: toM(num(stop, "stop")),
      steps: num(steps, "steps"),
      strategy: strategy.value,
    };
  };
  return row;
}

async function runJob(body) {
  runMsgEl.textContent = "";
  try {
    const job = await app.submitJob(body);
    runMsgEl.textContent = `job ${job.id} submitted`;
  } catch (e) {
    runMsgEl.textContent = String(e.message ?? e);
  }
}

async function runModels() {
  const models = checkedModels();
  const motor = app.getActiveConfig();
  if (!models.length || !motor) {
    runMsgEl.textContent = "check at least one model";
    return;
  }
  // fetch default params for models that need them (sim_plan, etc.)
  let params = {};
  const gated = models.filter(modelNeedsParams);
  for (const mk of gated) {
    try {
      const defs = await app.api.get(
        `/api/model-defaults/${encodeURIComponent(motor)}/${mk}`);
      Object.assign(params, defs);
    } catch (e) {
      if (models.length === 1) {
        runMsgEl.textContent = `${mk} defaults failed — ${e.message ?? e}`;
        return;
      }
      // multi-model: skip this model, server gates it out via _PARAM_GATED_FIELDS
    }
  }
  const body = models.length === 1
    ? { motor, model: models[0], params }
    : { motor, model: "validate", params: { models, ...params } };
  runJob(body);
}

function runSweep() {
  const motor = app.getActiveConfig();
  const models = checkedModels();
  const rows = [...axesEl.querySelectorAll(".axis-row")];
  if (!rows.length) {
    runMsgEl.textContent = "add a sweep axis";
    return;
  }
  let axes;
  try {
    axes = rows.map((r) => r.axis());
  } catch (e) {
    runMsgEl.textContent = e.message;
    return;
  }
  runJob({ motor, model: "sweep", params: { axes, model_keys: models } });
}

// -- panel contract ----------------------------------------------

export default {
  name: "config_editor",
  label: "Config Editor",
  accepts: [],
  // ~1,400-2,000 px tall when open — dominates the auto-fit grid, so it
  // starts collapsed until the user opens it (persisted per browser)

  init(container, appCtx) {
    app = appCtx;

    const saveRow = document.createElement("div");
    saveRow.className = "cfg-save-row";
    saveNameEl = document.createElement("input");
    saveNameEl.type = "text";
    saveNameEl.placeholder = "config name";
    const saveBtn = document.createElement("button");
    saveBtn.textContent = "Save";
    saveBtn.addEventListener("click", save);
    saveMsgEl = document.createElement("span");
    saveMsgEl.className = "cfg-msg";
    saveRow.append(saveNameEl, saveBtn, saveMsgEl);

    formEl = document.createElement("div");
    formEl.className = "cfg-form";

    const runSection = document.createElement("div");
    runSection.className = "cfg-run";
    const modelsHead = document.createElement("h3");
    modelsHead.textContent = "Models";
    modelListEl = document.createElement("div");
    modelListEl.className = "model-list";
    const runBtn = document.createElement("button");
    runBtn.textContent = "Run on active config";
    runBtn.addEventListener("click", runModels);

    const sweepHead = document.createElement("h3");
    sweepHead.textContent = "Sweep";
    axesEl = document.createElement("div");
    axesEl.className = "axes";
    const addAxis = document.createElement("button");
    addAxis.textContent = "Add axis";
    addAxis.className = "ghost-btn";
    addAxis.addEventListener("click", () => axesEl.append(axisRow()));
    const sweepBtn = document.createElement("button");
    sweepBtn.textContent = "Run sweep";
    sweepBtn.addEventListener("click", runSweep);
    runMsgEl = document.createElement("div");
    runMsgEl.className = "cfg-msg";

    runSection.append(modelsHead, modelListEl, runBtn,
                      sweepHead, axesEl, addAxis, sweepBtn, runMsgEl);
    container.append(saveRow, formEl, runSection);

    loadModels().catch((e) => { runMsgEl.textContent = `models load failed — ${e.message ?? e}`; });
    app.onConfigChange(load);
  },

  update() {},

  destroy() {
    app = formEl = saveNameEl = saveMsgEl = modelListEl = axesEl = runMsgEl = null;
    rawConfig = null;
  },
};
