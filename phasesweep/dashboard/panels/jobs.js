// Job queue panel: submit form, status badges, progress bars.

import { absTs, relTime } from "./format.js";

let app = null;
let listEl = null;
let formErrEl = null;
const rows = new Map(); // job_id -> { el, meta, progress, badge, fill, sub, ... }

// Sidebar row content mirrors run-history: "motor · type" title, a
// relative timestamp (absolute on hover) + progress in the sub, and the raw
// job id in the row tooltip rather than in the always-on line.
function composeSub(row) {
  const { motor, created_at } = row.meta;
  const { completed, total } = row.progress;
  row.sub.replaceChildren();
  const rel = document.createElement("span");
  rel.textContent = relTime(created_at);
  rel.title = absTs(created_at);
  row.sub.append(rel, document.createTextNode(` · ${motor} · ${completed}/${total}`));
}

function makeRow(job) {
  const el = document.createElement("li");
  el.className = "job-row";

  const head = document.createElement("div");
  head.className = "head";
  const title = document.createElement("span");
  title.className = "title";
  title.textContent = `${job.motor_name} · ${job.job_type}`;
  const badge = document.createElement("span");
  badge.className = "badge";
  const cancelBtn = document.createElement("button");
  cancelBtn.className = "cancel-btn";
  cancelBtn.textContent = "Cancel";
  cancelBtn.addEventListener("click", async () => {
    try {
      await app.api.del(`/api/jobs/${job.id}`);
    } catch (e) {
      setError(job.id, String(e.message ?? e));
    }
  });
  head.append(title, badge, cancelBtn);

  const sub = document.createElement("div");
  sub.className = "sub";

  const track = document.createElement("div");
  track.className = "progress-track";
  const fill = document.createElement("div");
  fill.className = "progress-fill";
  track.append(fill);

  const errEl = document.createElement("div");
  errEl.className = "error";

  // clicking a completed card routes its output to the plots;
  // the cancel button (shown only while active) keeps its own handler
  el.addEventListener("click", (ev) => {
    if (ev.target.closest(".cancel-btn")) return;
    const row = rows.get(job.id);
    if (row?.job?.status === "completed") app.showJobResults(row.job);
  });

  el.append(head, sub, track, errEl);
  return {
    el, badge, fill, track, sub, cancelBtn, errEl,
    meta: { motor: job.motor_name, type: job.job_type, created_at: job.created_at },
    progress: { completed: job.completed, total: job.total },
  };
}

function render(job) {
  let row = rows.get(job.id);
  if (!row) {
    row = makeRow(job);
    rows.set(job.id, row);
    listEl.prepend(row.el); // newest first
  }
  row.badge.textContent = job.status;
  row.badge.dataset.status = job.status;
  row.progress = { completed: job.completed, total: job.total };
  row.job = job; // click-to-plot reads the latest result_ids
  composeSub(row);
  row.el.title = `${job.motor_name} · ${job.job_type} · ${job.id} · ${absTs(job.created_at)}`;
  row.fill.style.width = job.total ? `${(100 * job.completed) / job.total}%` : "0%";
  const active = job.status === "pending" || job.status === "running";
  row.cancelBtn.style.display = active ? "" : "none";
  row.track.style.display = active ? "" : "none"; // done/failed: badge says it all
  // completed cards with results are click-to-plot; show the affordance
  row.el.classList.toggle("clickable", job.status === "completed" && job.result_ids?.length > 0);
  row.errEl.textContent = job.error ?? "";
}

function setProgress(jobId, completed, total) {
  const row = rows.get(jobId);
  if (!row) return;
  row.badge.textContent = "running";
  row.badge.dataset.status = "running";
  row.fill.style.width = total ? `${(100 * completed) / total}%` : "0%";
  row.progress = { completed, total };
  composeSub(row);
}

async function refetch(jobId) {
  try {
    render(await app.api.get(`/api/jobs/${jobId}`));
  } catch (e) {
    console.warn("job refetch failed", jobId, e);
  }
}

function setError(jobId, text) {
  const row = rows.get(jobId);
  if (row) row.errEl.textContent = text;
}

// -- submit form ----------------------------------------------------------------

async function buildForm(container) {
  const form = document.createElement("form");
  form.className = "job-form";

  const motorSel = document.createElement("select");
  motorSel.name = "motor";
  const modelSel = document.createElement("select");
  modelSel.name = "model";

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "params (JSON)";
  const paramsTa = document.createElement("textarea");
  paramsTa.placeholder = '{"axes": [{"field": "r_magnet", "start": 0.6, "stop": 0.68, "steps": 5}]}';
  details.append(summary, paramsTa);

  const submitBtn = document.createElement("button");
  submitBtn.type = "submit";
  submitBtn.textContent = "Submit job";
  formErrEl = document.createElement("div");
  formErrEl.className = "form-error";

  form.append(motorSel, modelSel, details, submitBtn, formErrEl);
  container.append(form);

  try {
    const [configs, models] = await Promise.all([
      app.api.get("/api/configs"),
      app.api.get("/api/models"),
    ]);
    for (const c of configs) {
      motorSel.add(new Option(c.name, c.name, false, c.name === app.getActiveConfig()));
    }
    app.onConfigChange((name) => {
      // configs saved from the editor arrive after init
      if (![...motorSel.options].some((o) => o.value === name)) {
        motorSel.add(new Option(name, name));
      }
      motorSel.value = name;
    });
    for (const m of models) {
      modelSel.add(new Option(`${m.key} (${m.cost})`, m.key, false, m.key === "analytical"));
    }
  } catch (e) {
    formErrEl.textContent = `form init failed — ${e.message ?? e}`;
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    formErrEl.textContent = "";
    let params = {};
    if (paramsTa.value.trim()) {
      try {
        params = JSON.parse(paramsTa.value);
      } catch {
        formErrEl.textContent = "params is not valid JSON";
        return;
      }
    }
    try {
      await app.submitJob({
        motor: motorSel.value, model: modelSel.value, params,
      });
    } catch (e) {
      formErrEl.textContent = String(e.message ?? e);
    }
  });
}

// -- panel contract ------------------------------------------------------------------

export default {
  name: "jobs",
  label: "Jobs",
  accepts: [],

  init(container, appCtx) {
    app = appCtx;
    buildForm(container);
    listEl = document.createElement("ul");
    listEl.className = "job-list";
    container.append(listEl);
  },

  // data: {kind: "jobs", jobs} full refresh | {kind: "job", job} one new
  // submission (from any panel via app.submitJob) | {kind: "ws", msg} live
  update(data) {
    if (data.kind === "job") {
      render(data.job);
      return;
    }
    if (data.kind === "jobs") {
      // Authoritative refresh: drop rows the server no longer knows
      // (in-memory registry, lost on restart).
      const known = new Set(data.jobs.map((j) => j.id));
      for (const [id, row] of rows) {
        if (!known.has(id)) {
          row.el.remove();
          rows.delete(id);
        }
      }
      const order = [...data.jobs].sort((a, b) => a.created_at.localeCompare(b.created_at));
      for (const job of order) render(job);
      return;
    }
    const msg = data.msg;
    if (msg.type === "job_progress") {
      setProgress(msg.job_id, msg.completed, msg.total);
    } else {
      refetch(msg.job_id); // terminal state: pick up final fields
    }
  },

  destroy() {
    rows.clear();
    app = listEl = formErrEl = null;
  },
};
