// Bottom status bar: connection state, running-job
// count, last-result receive time. A plain flex footer in the app frame —
// app.js drives it directly (setWsState / setJobCount / stampResult), no
// panel routing. The footer is also the future home of the cursor-stats
// table (H1), so it owns a stable slot at the bottom of the frame.

let stateEl = null;
let jobsEl = null;
let lastEl = null;

export default {
  name: "statusbar",

  init(container) {
    stateEl = document.createElement("span");
    stateEl.id = "ws-status"; // element identity survives the topbar->footer move
    stateEl.className = "ws-status";
    stateEl.dataset.state = "connecting";
    stateEl.innerHTML = '<span class="dot"></span><span class="label">connecting</span>';
    jobsEl = document.createElement("span");
    jobsEl.textContent = "0 jobs running";
    lastEl = document.createElement("span");
    lastEl.textContent = "last result —";
    container.append(stateEl, jobsEl, lastEl);
  },

  setWsState(state) {
    if (!stateEl) return;
    stateEl.dataset.state = state;
    stateEl.querySelector(".label").textContent = state;
  },

  setJobCount(n) {
    if (jobsEl) jobsEl.textContent = `${n} job${n === 1 ? "" : "s"} running`;
  },

  // stamped at result-routing time: "when did I last hear a result",
  // not the result's own compute timestamp (that's the provenance line)
  stampResult() {
    if (!lastEl) return;
    const hms = new Date().toLocaleTimeString([], { hour12: false });
    lastEl.textContent = `last result ${hms}`;
  },
};
