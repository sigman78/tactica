/* tactica war room — vanilla JS, no build step, no external deps. */
"use strict";

const $ = (sel) => document.querySelector(sel);

function el(tag, attrs = {}, ...children) {
  const ns = "http://www.w3.org/2000/svg";
  const svgTags = new Set(["svg", "rect", "line", "path", "text", "circle", "g", "polyline", "polygon", "image"]);
  const node = svgTags.has(tag)
    ? document.createElementNS(ns, tag)
    : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.setAttribute("class", v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    node.append(c.nodeType ? c : document.createTextNode(c));
  }
  return node;
}

function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._timer);
  t._timer = setTimeout(() => t.classList.add("hidden"), 5000);
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let detail = res.statusText;
    try { detail = (await res.json()).detail || detail; } catch { /* noop */ }
    throw new Error(detail);
  }
  return res.json();
}

const fmtScore = (m, ci) => `${m.toFixed(3)}±${Number.isFinite(ci) ? ci.toFixed(2) : "∞"}`;
const unitIcon = (unit) => `/static/units/${unit.toLowerCase()}.png`;
const fmtDur = (s) => s >= 90 ? `${(s / 60).toFixed(1)}m` : `${s.toFixed(1)}s`;

/* ================================================================== */
/* state */

const state = {
  meta: null,
  presets: [],
  jobs: [],
  currentJobId: null,
  es: null,            // active EventSource
  live: {},            // per-job live accumulators (llr series, matrix, ...)
  viewer: { frames: null, idx: 0, timer: null, source: null, gameIdx: null },
};

/* ================================================================== */
/* run-config forms */

const FORMS = {
  tournament: [
    ["agents", "text", "random,heuristic,weighted,mcts:32", "agent specs, comma list"],
    ["scenarios", "text", "all"],
    ["pairs", "number", 20],
    ["seed", "number", 1],
    ["workers", "number", 0, "0 = one per core"],
    ["out", "text", "", "optional JSONL path"],
  ],
  sprt: [
    ["candidate", "text", "weighted:weights/default.json"],
    ["baseline", "text", "weighted:weights/conservative.json"],
    ["elo0", "number", 0],
    ["elo1", "number", 10],
    ["alpha", "number", 0.05],
    ["beta", "number", 0.05],
    ["max_pairs", "number", 2000],
    ["scenarios", "text", "all"],
    ["seed", "number", 1],
  ],
  "skill-curve": [
    ["agent", "text", "heuristic"],
    ["epsilons", "text", "0,0.05,0.1,0.2,0.5"],
    ["pairs", "number", 100],
    ["scenarios", "text", "all"],
    ["seed", "number", 1],
    ["workers", "number", 0, "0 = one per core"],
  ],
  "noise-floor": [
    ["agent", "text", "heuristic"],
    ["pairs", "number", 100],
    ["scenarios", "text", "all"],
    ["seed", "number", 1],
    ["workers", "number", 0, "0 = one per core"],
  ],
  play: [
    ["p0", "text", "mcts:64"],
    ["p1", "text", "heuristic"],
    ["scenario", "scenario-select", "open_field"],
    ["seed", "number", 1],
    ["deterministic", "check", false],
  ],
};

const NUMERIC = new Set(["pairs", "seed", "workers", "elo0", "elo1", "alpha",
                         "beta", "max_pairs"]);

/* What each experiment measures and why it is designed that way. */
const KIND_INFO = {
  tournament: {
    title: "Tournament — who beats whom, and by how much",
    what: "A round-robin where every agent pair plays the same battles: " +
      "each (scenario, seed) is played twice with sides swapped (a mirrored " +
      "pair), and every matchup gets identical seeds (common random " +
      "numbers). Reports a pair-score matrix with 95% confidence intervals " +
      "plus OpenSkill ratings; 0.5 means even.",
    why: "Raw winrates are badly biased: asymmetric maps are one-sided and " +
      "even symmetric ones carry an initiative-driven side advantage. " +
      "Mirroring cancels the side bias inside each pair, and shared seeds " +
      "make the luck common to both agents — so what's left in the matrix " +
      "is decision quality, not dice.",
  },
  sprt: {
    title: "SPRT — is the candidate actually stronger?",
    what: "A sequential probability ratio test of candidate vs baseline. " +
      "Streams mirrored pairs and recomputes the log-likelihood ratio " +
      "after each one, stopping the moment the evidence crosses an accept " +
      "bound: H1 (candidate is at least elo1 stronger) or H0 (no " +
      "improvement beyond elo0).",
    why: "With a fixed number of games you either waste compute on clear " +
      "results or under-sample close ones. The sequential test spends " +
      "exactly as many games as the evidence requires, with controlled " +
      "false-positive (alpha) and false-negative (beta) rates. This is the " +
      "gate a weight change must pass before being promoted — the shipped " +
      "default weights earned their place through this exact test.",
  },
  "skill-curve": {
    title: "Skill curve — do decisions even matter here?",
    what: "Plays epsilon-blunder versions of an agent (random move with " +
      "probability eps, otherwise the clean agent) against the clean " +
      "agent, and plots pair score as a function of the blunder rate.",
    why: "A steep curve means the environment punishes mistakes — skill is " +
      "measurable and improvements can show up. A flat curve means " +
      "outcomes are luck-dominated and no amount of agent engineering " +
      "will register. The slope also calibrates effect sizes: it tells " +
      "you how many elo a given blunder rate costs.",
  },
  "noise-floor": {
    title: "Noise floor — the resolution limit of your experiments",
    what: "An agent plays itself on mirrored CRN pairs. By construction " +
      "the true strength difference is zero, so any deviation from 0.500 " +
      "is pure measurement noise: side advantage at the game level, " +
      "sampling noise at the pair level.",
    why: "Any measured 'improvement' smaller than this floor is " +
      "indistinguishable from luck. The split also shows what the paired " +
      "design buys: game-level scores drift far from 0.5 (the maps are " +
      "one-sided), while pair-level scores sit at exactly 0.500 — zero " +
      "noise for deterministic agents. For stochastic agents (mcts) the " +
      "pair-level band is the real resolution limit.",
  },
  play: {
    title: "Play — watch one battle unfold",
    what: "Runs a single seeded game between two agent specs and records " +
      "it. Open the result in the replay viewer to step through every " +
      "action — or fight an agent yourself in the Play tab.",
    why: "The statistics tell you who wins; watching tells you why. Use " +
      "it to eyeball behavior (does mcts dawdle? does the heuristic kite?), " +
      "debug pathologies the aggregate numbers hide, and sanity-check new " +
      "scenarios or weights before spending compute on a tournament.",
  },
};

function updateKindInfo(kind) {
  const info = KIND_INFO[kind];
  $("#kind-info-title").textContent = info ? info.title : "";
  $("#kind-info-what").textContent = info ? info.what : "";
  $("#kind-info-why").textContent = info ? `Why this design: ${info.why}` : "";
}

function renderForm(kind, config = null) {
  const form = $("#config-form");
  form.innerHTML = "";
  for (const [name, type, dflt, hint] of FORMS[kind]) {
    const value = config && config[name] !== undefined ? config[name] : dflt;
    let input;
    if (type === "check") {
      input = el("input", { type: "checkbox", id: `f-${name}`, name });
      input.checked = Boolean(value);
      form.append(el("div", { class: "field check" }, input,
        el("label", { for: `f-${name}` }, name)));
      continue;
    }
    if (type === "scenario-select") {
      input = el("select", { id: `f-${name}`, name });
      for (const sc of Object.keys(state.meta?.scenarios || {})) {
        input.append(el("option", sc === value ? { selected: "" } : {}, sc));
      }
    } else {
      input = el("input", {
        type: type === "number" ? "number" : "text",
        id: `f-${name}`, name, step: "any",
        value: Array.isArray(value) ? value.join(",") : String(value ?? ""),
      });
    }
    form.append(el("div", { class: "field" },
      el("label", { for: `f-${name}` }, hint ? `${name} · ${hint}` : name),
      input));
  }
}

function readForm() {
  const config = {};
  for (const input of $("#config-form").querySelectorAll("input,select")) {
    const name = input.name;
    if (input.type === "checkbox") { if (input.checked) config[name] = true; continue; }
    const raw = input.value.trim();
    if (raw === "") continue;
    config[name] = NUMERIC.has(name) ? Number(raw) : raw;
  }
  return config;
}

function currentConfig() {
  const jsonArea = $("#config-json");
  if (!jsonArea.classList.contains("hidden")) {
    return JSON.parse(jsonArea.value);
  }
  return readForm();
}

function setupConfigPanel() {
  const kindSel = $("#kind-select");
  for (const kind of Object.keys(FORMS)) kindSel.append(el("option", {}, kind));
  kindSel.addEventListener("change", () => {
    $("#config-title").textContent = "New experiment";
    $("#config-desc").textContent = "";
    renderForm(kindSel.value);
    syncJsonFromForm();
    updateKindInfo(kindSel.value);
  });
  renderForm(kindSel.value);
  updateKindInfo(kindSel.value);

  $("#btn-json-toggle").addEventListener("click", () => {
    const area = $("#config-json"), form = $("#config-form");
    if (area.classList.contains("hidden")) {
      syncJsonFromForm();
      area.classList.remove("hidden");
      form.classList.add("hidden");
      $("#btn-json-toggle").textContent = "form";
    } else {
      try {
        renderForm(kindSel.value, JSON.parse(area.value));
      } catch (e) { toast(`bad JSON: ${e.message}`); return; }
      area.classList.add("hidden");
      form.classList.remove("hidden");
      $("#btn-json-toggle").textContent = "raw json";
    }
  });

  $("#btn-run").addEventListener("click", async () => {
    let config;
    try { config = currentConfig(); } catch (e) { toast(`bad JSON: ${e.message}`); return; }
    try {
      const job = await api("/api/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ kind: kindSel.value, config }),
      });
      await refreshJobs();
      selectJob(job.id);
    } catch (e) { toast(`run failed: ${e.message}`); }
  });

  $("#btn-save-preset").addEventListener("click", async () => {
    const name = prompt("preset name:", $("#config-title").dataset.preset || "");
    if (!name) return;
    let config;
    try { config = currentConfig(); } catch (e) { toast(`bad JSON: ${e.message}`); return; }
    const description = prompt("description (optional):", $("#config-desc").textContent) || "";
    try {
      await api("/api/presets", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, kind: kindSel.value, config, description }),
      });
      await refreshPresets();
    } catch (e) { toast(`save failed: ${e.message}`); }
  });
}

function syncJsonFromForm() {
  $("#config-json").value = JSON.stringify(readForm(), null, 2);
}

/* ================================================================== */
/* presets sidebar */

async function refreshPresets() {
  state.presets = await api("/api/presets");
  const ul = $("#preset-list");
  ul.innerHTML = "";
  if (!state.presets.length) {
    ul.append(el("li", { class: "empty" }, "no presets yet — save one"));
  }
  for (const p of state.presets) {
    const li = el("li", {},
      el("div", { class: "row1" },
        el("span", { class: "name" }, p.name),
        el("span", { class: "hint" }, p.kind || "?")),
      el("div", { class: "meta" }, p.description || JSON.stringify(p.config || {})));
    li.addEventListener("click", () => loadPreset(p));
    ul.append(li);
  }
}

function loadPreset(p) {
  if (p.error) { toast(p.error); return; }
  switchTab("run");
  $("#kind-select").value = p.kind;
  renderForm(p.kind, p.config || {});
  syncJsonFromForm();
  $("#config-title").textContent = `Experiment: ${p.name}`;
  $("#config-title").dataset.preset = p.name;
  $("#config-desc").textContent = p.description || "";
  updateKindInfo(p.kind);
}

/* ================================================================== */
/* jobs sidebar */

async function refreshJobs() {
  state.jobs = await api("/api/jobs");
  const ul = $("#job-list");
  ul.innerHTML = "";
  if (!state.jobs.length) {
    ul.append(el("li", { class: "empty" }, "no runs this session"));
  }
  for (const j of state.jobs) {
    const li = el("li", j.id === state.currentJobId ? { class: "selected" } : {},
      el("div", { class: "row1" },
        el("span", { class: "name" }, j.kind),
        el("span", { class: `chip ${j.status}` }, j.status)),
      el("div", { class: "meta" }, jobLabel(j)));
    li.addEventListener("click", () => { switchTab("run"); selectJob(j.id); });
    ul.append(li);
  }
}

function jobLabel(j) {
  const c = j.config || {};
  switch (j.kind) {
    case "tournament": return String(c.agents || "");
    case "sprt": return `${c.candidate} vs ${c.baseline}`;
    case "skill-curve": case "noise-floor": return String(c.agent || "");
    case "play": return `${c.p0} vs ${c.p1} @ ${c.scenario}`;
    default: return "";
  }
}

/* ================================================================== */
/* live job view */

function selectJob(jobId) {
  if (state.es) { state.es.close(); state.es = null; }
  state.currentJobId = jobId;
  state.live = { llr: [], points: [], rows: [] };
  $("#job-panel").classList.remove("hidden");
  $("#job-charts").innerHTML = "";
  $("#job-result").innerHTML = "";
  $("#log-pane").textContent = "";
  $("#progress-fill").style.width = "0";
  $("#progress-text").textContent = "—";
  refreshJobs();

  const job = state.jobs.find((j) => j.id === jobId);
  $("#job-title").textContent = job ? `${job.kind} — ${jobLabel(job)}` : jobId;
  setStatus(job ? job.status : "queued");

  const es = new EventSource(`/api/jobs/${jobId}/events`);
  state.es = es;
  es.onmessage = (msg) => {
    const { type, data } = JSON.parse(msg.data);
    handleEvent(job ? job.kind : "?", type, data);
  };
  es.onerror = () => { /* terminal status closes server side; ignore */ };
}

function setStatus(status) {
  const chip = $("#job-status");
  chip.textContent = status;
  chip.className = `chip ${status}`;
  $("#btn-cancel").disabled = !(status === "running" || status === "queued");
}

function handleEvent(kind, type, data) {
  if (type === "ping") return;
  if (type === "log") { appendLog(data.line); return; }
  if (type === "status") {
    setStatus(data.status);
    if (["done", "failed", "cancelled"].includes(data.status)) {
      state.es?.close();
      state.es = null;
      refreshJobs();
      if (data.status === "failed") toast(data.error || "job failed");
      if (data.result) renderResult(kind, data.result);
    }
    return;
  }
  if (type === "progress") {
    const { completed, total, elapsed } = data;
    if (total) {
      $("#progress-fill").style.width = `${(100 * completed / total).toFixed(1)}%`;
      const rate = completed / (elapsed || 1);
      $("#progress-text").textContent =
        `${completed}/${total} pairs · ${fmtDur(elapsed || 0)} · ${rate.toFixed(1)}/s`;
    }
    if (kind === "tournament" && data.matrix) renderMatrix($("#job-charts"), data.matrix, true);
    if (kind === "sprt" && data.llr !== undefined) {
      state.live.llr.push(data.llr);
      state.live.wdl = [data.wins, data.draws, data.losses];
      renderSprtLive();
    }
    return;
  }
  if (type === "point") { state.live.points.push(data); renderSkillCurveLive(); return; }
  if (type === "row") { state.live.rows.push(data); renderNoiseFloorLive(); return; }
}

function appendLog(line) {
  const pane = $("#log-pane");
  pane.textContent += line + "\n";
  pane.scrollTop = pane.scrollHeight;
}

$("#btn-cancel")?.addEventListener("click", async () => {
  if (state.currentJobId) {
    await api(`/api/jobs/${state.currentJobId}/cancel`, { method: "POST" });
  }
});

/* ------------------------------------------------------------------ */
/* renderers per kind */

function box(host, id, title) {
  let b = host.querySelector(`#${id}`);
  if (!b) {
    b = el("div", { id });
    if (title) b.append(el("p", { class: "chart-title" }, title));
    b.append(el("div", { class: "content" }));
    host.append(b);
  }
  return b.querySelector(".content");
}

function heatColor(mean) {
  const dev = mean - 0.5;                       // -0.5 .. +0.5
  const a = Math.min(Math.abs(dev) * 1.6, 0.75);
  return dev >= 0
    ? `rgba(111, 191, 115, ${a})`               // row beats column
    : `rgba(211, 106, 95, ${a})`;
}

function renderMatrix(host, rows, live) {
  const content = box(host, "matrix-box",
    "PAIR-SCORE MATRIX · row vs column · 0.5 = even");
  const agents = [...new Set(rows.flatMap((r) => [r.a, r.b]))];
  const cell = new Map(rows.map((r) => [`${r.a}|${r.b}`, r]));
  const table = el("table", { class: "matrix" });
  table.append(el("tr", {}, el("th"), ...agents.map((a) => el("th", {}, a))));
  for (const a of agents) {
    const tr = el("tr", {}, el("th", { class: "rowh" }, a));
    for (const b of agents) {
      if (a === b) { tr.append(el("td", { class: "diag" }, "—")); continue; }
      const r = cell.get(`${a}|${b}`);
      const ci = Number.isFinite(r?.ci) ? r.ci.toFixed(2) : "∞";
      const td = r
        ? el("td", { style: `background:${heatColor(r.mean)}` },
            r.mean.toFixed(3), el("span", { class: "ci" }, `±${ci} n=${r.n}`))
        : el("td", {}, "…");
      tr.append(td);
    }
    table.append(tr);
  }
  content.replaceChildren(table);
}

function renderSprtLive() {
  const content = box($("#job-charts"), "llr-box", "LLR TRAJECTORY");
  const job = state.jobs.find((j) => j.id === state.currentJobId);
  const cfg = job?.config || {};
  // bounds from config (alpha/beta) — recompute like the backend
  const alpha = cfg.alpha ?? 0.05, beta = cfg.beta ?? 0.05;
  const lower = Math.log(beta / (1 - alpha)), upper = Math.log((1 - beta) / alpha);
  const series = state.live.llr.map((y, i) => [i + 1, y]);
  content.replaceChildren(lineChart({
    series: [{ points: series, color: "var(--gold)" }],
    hlines: [
      { y: upper, label: "accept H1", color: "var(--good)" },
      { y: lower, label: "accept H0", color: "var(--bad)" },
      { y: 0, color: "var(--hairline)" },
    ],
    xlabel: "mirrored pairs", ylabel: "LLR",
  }));
  const [w, d, l] = state.live.wdl || [0, 0, 0];
  const n = Math.max(w + d + l, 1);
  const wdl = box($("#job-charts"), "wdl-box", null);
  wdl.replaceChildren(el("span", { class: "mono hint" },
    `WDL ${w}/${d}/${l} · score ${((w + d / 2) / n).toFixed(4)}`));
}

function renderSkillCurveLive() {
  const content = box($("#job-charts"), "curve-box",
    "SKILL CURVE · pair score vs blunder rate");
  const pts = state.live.points.map((p) => [p.eps, p.score]);
  const errs = state.live.points.map((p) => [p.eps, p.score, p.ci]);
  content.replaceChildren(lineChart({
    series: [{ points: pts, color: "var(--blue)", markers: true, errors: errs }],
    hlines: [{ y: 0.5, label: "no skill cost", color: "var(--hairline)", dash: true }],
    xlabel: "epsilon", ylabel: "pair score", yDomain: [0, 0.6],
  }));
}

function renderNoiseFloorLive() {
  const content = box($("#job-charts"), "floor-box", "NOISE FLOOR · per scenario");
  const table = el("table", { class: "data" },
    el("tr", {},
      el("th", {}, "scenario"), el("th", {}, "game-level"), el("th", {}, "dev"),
      el("th", {}, "pair-level"), el("th", {}, "n")));
  for (const r of state.live.rows) {
    const dev = r.game_mean - 0.5;
    table.append(el("tr", {},
      el("td", {}, r.scenario),
      el("td", {}, fmtScore(r.game_mean, r.game_ci)),
      el("td", { style: `color:${Math.abs(dev) > 0.05 ? "var(--warn)" : "var(--text-dim)"}` },
        (dev >= 0 ? "+" : "") + dev.toFixed(4)),
      el("td", {}, fmtScore(r.pair_mean, r.pair_ci)),
      el("td", {}, r.n)));
  }
  content.replaceChildren(table);
}

function renderResult(kind, result) {
  const host = $("#job-result");
  host.innerHTML = "";
  if (kind === "tournament") {
    renderMatrix($("#job-charts"), result.matrix, false);
    if (result.ratings) {
      host.append(el("div", { class: "block-title" }, "OpenSkill ratings (PlackettLuce)"));
      const maxOrd = Math.max(...result.ratings.map((r) => r.ordinal), 1);
      for (const r of result.ratings) {
        host.append(el("div", { class: "rating-row" },
          el("span", { class: "mono" }, r.agent),
          el("div", { class: "bar-track" },
            el("div", { class: "bar", style: `width:${Math.max(0, 100 * r.ordinal / maxOrd)}%` })),
          el("span", { class: "nums" },
            `μ=${r.mu.toFixed(2)} σ=${r.sigma.toFixed(2)} ord=${r.ordinal.toFixed(2)}`)));
      }
    }
    host.append(el("div", { class: "block-title" }, "Per-scenario pair scores"));
    const t = el("table", { class: "data" }, el("tr", {},
      el("th", {}, "matchup"), el("th", {}, "scenario"), el("th", {}, "score"), el("th", {}, "n")));
    for (const r of result.by_scenario) {
      t.append(el("tr", {},
        el("td", {}, `${r.a} vs ${r.b}`), el("td", { style: "text-align:left" }, r.scenario),
        el("td", {}, fmtScore(r.mean, r.ci)), el("td", {}, r.n)));
    }
    host.append(t, el("p", { class: "hint" },
      `${result.n_games} games in ${fmtDur(result.duration)}`));
  } else if (kind === "sprt") {
    state.live.llr = result.trajectory;
    state.live.wdl = [result.wins, result.draws, result.losses];
    renderSprtLive();
    const cls = result.verdict.includes("H1") ? "h1" : result.verdict.includes("H0") ? "h0" : "";
    host.append(
      el("div", { class: `verdict ${cls}` }, `Verdict: ${result.verdict}`),
      el("p", { class: "mono hint" },
        `${result.pairs} pairs · WDL ${result.wins}/${result.draws}/${result.losses}` +
        ` · score ${result.score.toFixed(4)} (~${result.elo >= 0 ? "+" : ""}${result.elo.toFixed(1)} elo)` +
        ` · final LLR ${result.llr.toFixed(3)} in [${result.bounds[0].toFixed(2)}, ${result.bounds[1].toFixed(2)}]` +
        ` · ${fmtDur(result.duration)}`));
  } else if (kind === "skill-curve") {
    state.live.points = result.points;
    renderSkillCurveLive();
    host.append(el("p", { class: "hint" }, `done in ${fmtDur(result.duration)} · ` +
      "steep curve = environment rewards skill; flat = luck-dominated"));
  } else if (kind === "noise-floor") {
    state.live.rows = result.rows;
    renderNoiseFloorLive();
    const o = result.overall;
    host.append(el("p", { class: "mono hint" },
      `OVERALL game-level ${fmtScore(o.game_mean, o.game_ci)} (dev ${(o.game_mean - 0.5).toFixed(4)})` +
      ` · pair-level ${fmtScore(o.pair_mean, o.pair_ci)} · ${fmtDur(result.duration)}`));
  } else if (kind === "play") {
    const outcome = result.winner == null ? "draw"
      : `side ${result.winner} (${result.specs[result.winner]}) wins`;
    host.append(
      el("div", { class: "verdict" },
        `${result.scenario} seed=${result.seed}: ${outcome} after ${result.rounds} rounds`),
      el("button", {
        class: "primary",
        onclick: () => openJobReplay(state.currentJobId, 0),
      }, "▶ watch in replay viewer"));
  }
}

/* ================================================================== */
/* SVG line chart (hand rolled, no deps) */

function lineChart({ series, hlines = [], xlabel = "", ylabel = "", yDomain = null }) {
  const W = 720, H = 260, padL = 52, padR = 14, padT = 12, padB = 34;
  const allPts = series.flatMap((s) => s.points);
  const hys = hlines.map((h) => h.y);
  if (!allPts.length) return el("div", { class: "chart-box" }, el("span", { class: "hint" }, "waiting for data…"));
  let xs = allPts.map((p) => p[0]), ys = allPts.map((p) => p[1]).concat(hys);
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  let [y0, y1] = yDomain || [Math.min(...ys), Math.max(...ys)];
  if (x0 === x1) { x0 -= 1; x1 += 1; }
  const ypad = (y1 - y0) * 0.08 || 0.1;
  if (!yDomain) { y0 -= ypad; y1 += ypad; }
  const X = (x) => padL + (x - x0) / (x1 - x0) * (W - padL - padR);
  const Y = (y) => H - padB - (y - y0) / (y1 - y0) * (H - padT - padB);

  const svg = el("svg", { viewBox: `0 0 ${W} ${H}` });
  // axes + ticks
  svg.append(el("line", { x1: padL, y1: H - padB, x2: W - padR, y2: H - padB,
                          stroke: "var(--hairline)" }));
  svg.append(el("line", { x1: padL, y1: padT, x2: padL, y2: H - padB,
                          stroke: "var(--hairline)" }));
  for (let i = 0; i <= 4; i++) {
    const yv = y0 + (y1 - y0) * i / 4;
    svg.append(el("text", { x: padL - 6, y: Y(yv) + 4, "text-anchor": "end",
      "font-size": 10, fill: "var(--text-dim)" }, yv.toFixed(2)));
    svg.append(el("line", { x1: padL, y1: Y(yv), x2: W - padR, y2: Y(yv),
      stroke: "var(--hairline)", "stroke-opacity": 0.35 }));
  }
  for (let i = 0; i <= 5; i++) {
    const xv = x0 + (x1 - x0) * i / 5;
    svg.append(el("text", { x: X(xv), y: H - padB + 14, "text-anchor": "middle",
      "font-size": 10, fill: "var(--text-dim)" },
      Math.abs(xv) >= 100 ? Math.round(xv) : +xv.toFixed(2)));
  }
  if (xlabel) svg.append(el("text", { x: (padL + W - padR) / 2, y: H - 4,
    "text-anchor": "middle", "font-size": 10, fill: "var(--text-dim)" }, xlabel));
  if (ylabel) svg.append(el("text", { x: 12, y: padT + 8, "font-size": 10,
    fill: "var(--text-dim)" }, ylabel));

  for (const h of hlines) {
    svg.append(el("line", { x1: padL, y1: Y(h.y), x2: W - padR, y2: Y(h.y),
      stroke: h.color || "var(--text-dim)",
      "stroke-dasharray": h.dash === false ? "" : "5 4" }));
    if (h.label) svg.append(el("text", { x: W - padR - 4, y: Y(h.y) - 4,
      "text-anchor": "end", "font-size": 10, fill: h.color || "var(--text-dim)" }, h.label));
  }
  for (const s of series) {
    if (s.points.length > 1) {
      svg.append(el("polyline", {
        points: s.points.map(([x, y]) => `${X(x)},${Y(y)}`).join(" "),
        fill: "none", stroke: s.color, "stroke-width": 1.8 }));
    }
    if (s.errors) {
      for (const [x, y, e] of s.errors) {
        if (!Number.isFinite(e)) continue;
        svg.append(el("line", { x1: X(x), y1: Y(y - e), x2: X(x), y2: Y(y + e),
          stroke: s.color, "stroke-width": 1.2 }));
      }
    }
    if (s.markers || s.points.length === 1) {
      for (const [x, y] of s.points) {
        svg.append(el("circle", { cx: X(x), cy: Y(y), r: 3.2, fill: s.color }));
      }
    }
  }
  return el("div", { class: "chart-box" }, svg);
}

/* ================================================================== */
/* replay viewer */

const CELL = 54, BPAD = 0;

async function refreshReplaySources() {
  const list = $("#replay-source-list");
  list.innerHTML = "";
  const files = await api("/api/replays");
  const jobsWithGames = state.jobs.filter((j) => j.n_games > 0);
  for (const j of jobsWithGames) {
    const item = el("div", { class: "source-item" },
      el("span", {}, `run: ${j.kind} · ${jobLabel(j)}`),
      el("span", { class: "hint" }, `${j.n_games}`));
    item.addEventListener("click", () => selectSource(item, { job: j.id }));
    list.append(item);
  }
  for (const f of files) {
    const item = el("div", { class: "source-item" },
      el("span", {}, f.file), el("span", { class: "hint" }, `${f.games}`));
    item.addEventListener("click", () => selectSource(item, { file: f.file }));
    list.append(item);
  }
  if (!list.children.length) {
    list.append(el("div", { class: "hint" }, "no game records yet — run something"));
  }
}

async function selectSource(item, source) {
  document.querySelectorAll(".source-item").forEach((n) => n.classList.remove("selected"));
  item.classList.add("selected");
  state.viewer.source = source;
  const games = source.job
    ? await api(`/api/jobs/${source.job}/games`)
    : await api(`/api/replays/games?file=${encodeURIComponent(source.file)}`);
  const list = $("#replay-game-list");
  list.innerHTML = "";
  for (const g of games) {
    const wcls = g.winner == null ? "" : `w${g.winner}`;
    const item2 = el("div", { class: "game-item" },
      el("span", {}, `#${g.index} ${g.scenario}`),
      el("span", { class: wcls },
        g.winner == null ? "draw" : `${g.specs[g.winner]} (${g.winner})`),
      el("span", { class: "hint" }, `r${g.rounds}`));
    item2.addEventListener("click", () => {
      document.querySelectorAll(".game-item").forEach((n) => n.classList.remove("selected"));
      item2.classList.add("selected");
      loadGame(source, g.index);
    });
    list.append(item2);
  }
}

async function openJobReplay(jobId, index) {
  switchTab("replays");
  await refreshReplaySources();
  state.viewer.source = { job: jobId };
  loadGame({ job: jobId }, index);
}

async function loadGame(source, index) {
  stopPlayback();
  try {
    const data = source.job
      ? await api(`/api/jobs/${source.job}/frames/${index}`)
      : await api(`/api/replays/frames?file=${encodeURIComponent(source.file)}&index=${index}`);
    state.viewer.frames = data;
    state.viewer.idx = 0;
    $("#vc-slider").max = data.frames.length - 1;
    $("#vc-slider").value = 0;
    $("#viewer-title").textContent =
      `${data.scenario} · ${data.specs[0]} (gold) vs ${data.specs[1]} (blue)`;
    const outcome = data.winner == null ? "draw"
      : `side ${data.winner} (${data.specs[data.winner]}) wins`;
    $("#viewer-outcome").textContent = `${outcome} · ${data.rounds} rounds · seed ${data.seed}`;
    drawFrame();
  } catch (e) { toast(`replay failed: ${e.message}`); }
}

function drawFrame() {
  const v = state.viewer;
  if (!v.frames) return;
  const { board, obstacles, frames } = v.frames;
  const frame = frames[v.idx];
  const svg = $("#board");
  svg.setAttribute("viewBox", `0 0 ${board.w * CELL} ${board.h * CELL}`);
  svg.innerHTML = "";

  for (let y = 0; y < board.h; y++) {
    for (let x = 0; x < board.w; x++) {
      svg.append(el("rect", {
        x: x * CELL, y: y * CELL, width: CELL, height: CELL,
        class: `cell${(x + y) % 2 ? " dark" : ""}` }));
    }
  }
  for (const o of obstacles) {
    svg.append(el("rect", {
      x: o.x * CELL + 3, y: o.y * CELL + 3, width: CELL - 6, height: CELL - 6,
      class: "obstacle", rx: 4 }));
    svg.append(el("line", { x1: o.x * CELL + 8, y1: o.y * CELL + CELL - 8,
      x2: o.x * CELL + CELL - 8, y2: o.y * CELL + 8,
      stroke: "var(--gold-dim)", "stroke-opacity": 0.5 }));
  }
  for (const s of frame.stacks) {
    const gx = s.x * CELL, gy = s.y * CELL;
    const g = el("g", {});
    const active = s.uid === frame.active;
    g.append(el("rect", {
      x: gx + 4, y: gy + 4, width: CELL - 8, height: CELL - 8, rx: 6,
      class: `stack-tile side${s.side}${active ? " active" : ""}` }));
    g.append(el("image", { href: unitIcon(s.unit), x: gx + 12, y: gy + 11,
      width: 30, height: 30, class: "stack-icon" }));
    g.append(el("text", { x: gx + CELL / 2, y: gy + CELL - 9,
      class: "stack-count" }, `×${s.count}`));
    // top-creature HP bar
    const hpw = (CELL - 16) * (s.top_hp / s.max_hp);
    g.append(el("rect", { x: gx + 8, y: gy + 7, width: CELL - 16, height: 3,
      class: "hp-track", rx: 1.5 }));
    g.append(el("rect", { x: gx + 8, y: gy + 7, width: Math.max(hpw, 0), height: 3,
      class: "hp-fill", rx: 1.5 }));
    if (s.defending) {
      g.append(el("circle", { cx: gx + CELL - 11, cy: gy + 12, r: 4, class: "defend-mark" }));
    }
    if (s.impaired) impairedMark(g, gx, gy);
    g.append(el("title", {},
      `${s.unit} ×${s.count} · ${s.top_hp}/${s.max_hp}hp top · side ${s.side}` +
      (s.defending ? " · defending" : "") +
      (s.impaired ? " · shooter blocked" : "")));
    svg.append(g);
  }

  const actorName = frame.actor != null
    ? (frame.stacks.find((s) => s.uid === frame.actor)?.unit ||
       frames[v.idx - 1]?.stacks.find((s) => s.uid === frame.actor)?.unit || "?")
    : null;
  $("#viewer-caption").textContent = v.idx === 0
    ? `deployment · round ${frame.round}`
    : `step ${v.idx}/${frames.length - 1} · round ${frame.round} · ${actorName}: ${frame.action}`;
  $("#vc-slider").value = v.idx;
}

function stepFrame(d) {
  const v = state.viewer;
  if (!v.frames) return;
  v.idx = Math.max(0, Math.min(v.frames.frames.length - 1, v.idx + d));
  drawFrame();
}

function stopPlayback() {
  if (state.viewer.timer) { clearInterval(state.viewer.timer); state.viewer.timer = null; }
  $("#vc-play").innerHTML = "&#9654;";
}

function setupViewer() {
  $("#vc-first").addEventListener("click", () => { stopPlayback(); state.viewer.idx = 0; drawFrame(); });
  $("#vc-prev").addEventListener("click", () => { stopPlayback(); stepFrame(-1); });
  $("#vc-next").addEventListener("click", () => { stopPlayback(); stepFrame(1); });
  $("#vc-play").addEventListener("click", () => {
    const v = state.viewer;
    if (v.timer) { stopPlayback(); return; }
    if (!v.frames) return;
    if (v.idx >= v.frames.frames.length - 1) v.idx = 0;
    $("#vc-play").innerHTML = "&#10073;&#10073;";
    v.timer = setInterval(() => {
      if (v.idx >= v.frames.frames.length - 1) { stopPlayback(); return; }
      stepFrame(1);
    }, Number($("#vc-speed").value));
  });
  $("#vc-speed").addEventListener("change", () => {
    if (state.viewer.timer) { stopPlayback(); $("#vc-play").click(); }
  });
  $("#vc-slider").addEventListener("input", (e) => {
    stopPlayback();
    state.viewer.idx = Number(e.target.value);
    drawFrame();
  });
  $("#btn-replays-refresh").addEventListener("click", refreshReplaySources);
  document.addEventListener("keydown", (e) => {
    if (!$("#tab-replays").classList.contains("active")) return;
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.key === "ArrowLeft") { stopPlayback(); stepFrame(-1); }
    if (e.key === "ArrowRight") { stopPlayback(); stepFrame(1); }
    if (e.key === " ") { e.preventDefault(); $("#vc-play").click(); }
  });
}

/* ================================================================== */
/* play vs human */

let actPending = false;

async function startGame() {
  try {
    const st = await api("/api/games", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent: $("#pg-agent").value.trim() || "heuristic",
        scenario: $("#pg-scenario").value,
        seed: Number($("#pg-seed").value) || 1,
        human_side: Number($("#pg-side").value),
        deterministic: $("#pg-det").checked,
      }),
    });
    renderPlay(st);
  } catch (e) { toast(`could not start: ${e.message}`); }
}

async function playAct(actionId) {
  if (actPending || !state.play) return;
  actPending = true;
  const svg = $("#play-board");
  svg.classList.add("busy");
  try {
    const st = await api(`/api/games/${state.play.id}/act`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ action: actionId }),
    });
    const fresh = st.events.slice(state.playSeen || 0);
    if (state.playGroups && fresh.length) await animateEvents(fresh);
    renderPlay(st);
  } catch (e) { toast(e.message); }
  svg.classList.remove("busy");
  actPending = false;
}

/* ---- step animation: cosmetic cutscene over the stale board, then a
   truth re-render from the new server state ---- */

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function animateEvents(events) {
  const groups = state.playGroups;
  const svg = $("#play-board");
  svg.querySelectorAll(".move-overlay").forEach((n) => n.remove());
  for (const ev of events) {
    const g = groups.get(ev.uid);
    if (!g) continue;
    if (ev.t === "wait") {
      g.animate([{ opacity: 1 }, { opacity: 0.35 }, { opacity: 1 }], { duration: 280 });
      await sleep(240);
    } else if (ev.t === "defend") {
      const ring = el("circle", { cx: CELL / 2, cy: CELL / 2, r: 17, class: "defend-pulse" });
      g.append(ring);
      ring.animate([{ opacity: 0.9 }, { opacity: 0 }], { duration: 450, fill: "forwards" });
      setTimeout(() => ring.remove(), 480);
      await sleep(300);
    } else {
      // glide to the post-step cell (a plain move, or the melee approach)
      if (ev.to && (g._pos.x !== ev.to.x || g._pos.y !== ev.to.y)) {
        g.style.transform = `translate(${ev.to.x * CELL}px, ${ev.to.y * CELL}px)`;
        g._pos = { x: ev.to.x, y: ev.to.y };
        await sleep(310);
      }
      if (ev.t === "attack") {
        const tg = groups.get(ev.target);
        if (ev.melee && tg) {
          const dx = Math.sign(tg._pos.x - g._pos.x) * 10;
          const dy = Math.sign(tg._pos.y - g._pos.y) * 10;
          const base = `translate(${g._pos.x * CELL}px, ${g._pos.y * CELL}px)`;
          g.animate([
            { transform: base },
            { transform: `translate(${g._pos.x * CELL + dx}px, ${g._pos.y * CELL + dy}px)`, offset: 0.4 },
            { transform: base },
          ], { duration: 260, easing: "ease-out" });
        } else if (tg) {
          const line = el("line", {
            x1: g._pos.x * CELL + CELL / 2, y1: g._pos.y * CELL + CELL / 2,
            x2: tg._pos.x * CELL + CELL / 2, y2: tg._pos.y * CELL + CELL / 2,
            class: "tracer" });
          svg.append(line);
          line.animate([{ opacity: 0.9 }, { opacity: 0 }], { duration: 380, fill: "forwards" });
          setTimeout(() => line.remove(), 400);
        }
        await sleep(170);
        for (const ef of ev.effects || []) applyHit(groups.get(ef.uid), ef);
        await sleep(580);
      }
    }
  }
}

function applyHit(g, ef) {
  if (!g) return;
  g._count.textContent = `×${ef.count}`;
  g._hp.setAttribute("width", Math.max((CELL - 16) * ef.top_hp / g._maxhp, 0));
  g.animate([{ opacity: 1 }, { opacity: 0.45 }, { opacity: 1 }], { duration: 200 });
  const dmg = el("text", { x: CELL / 2, y: 8, class: "dmg-float" }, `−${ef.dmg}`);
  g.append(dmg);
  dmg.animate([{ transform: "translateY(0px)", opacity: 1 },
               { transform: "translateY(-18px)", opacity: 0 }],
              { duration: 850, easing: "ease-out", fill: "forwards" });
  setTimeout(() => dmg.remove(), 900);
  if (ef.killed > 0) {
    const note = el("text", { x: CELL / 2, y: 22, class: "dmg-float slain" },
                    ef.dead ? "destroyed!" : `${ef.killed} slain`);
    g.append(note);
    note.animate([{ transform: "translateY(0px)", opacity: 1 },
                  { transform: "translateY(-14px)", opacity: 0 }],
                 { duration: 1000, easing: "ease-out", fill: "forwards" });
    setTimeout(() => note.remove(), 1050);
  }
  if (ef.dead) {
    g.animate([{ opacity: 1 }, { opacity: 0 }],
              { duration: 450, delay: 380, fill: "forwards" });
  }
}

function renderPlay(st) {
  state.play = st;
  $("#play-setup").classList.add("hidden");
  $("#play-layout").classList.remove("hidden");

  const you = st.human_side === 0 ? "gold" : "blue";
  $("#play-title").textContent =
    `${st.scenario} · you are ${you} vs ${st.agent}`;
  $("#play-round").textContent = st.round;

  const banner = $("#play-turn-banner");
  if (st.status === "over") {
    banner.textContent = "battle over";
    banner.className = "chip";
  } else {
    banner.textContent = "your turn";
    banner.className = "chip yours";
  }

  // initiative queue
  const q = $("#play-queue");
  q.innerHTML = "";
  for (const [i, c] of st.queue.entries()) {
    q.append(el("span", {
      class: `q-chip s${c.side}${i === 0 ? " now" : ""}${c.waiting ? " waiting" : ""}`,
      title: `${c.unit} ×${c.count}${c.waiting ? " (waiting)" : ""}` },
      el("img", { class: "q-icon", src: unitIcon(c.unit), alt: c.glyph }),
      `×${c.count}`));
  }

  // log
  const log = $("#play-log");
  log.textContent = st.log.join("\n");
  log.scrollTop = log.scrollHeight;

  // wait / defend
  const waitAction = st.legal.find((a) => a.type === "WAIT");
  const defendAction = st.legal.find((a) => a.type === "DEFEND");
  $("#pa-wait").disabled = !waitAction;
  $("#pa-defend").disabled = !defendAction;
  $("#pa-wait").onclick = () => waitAction && playAct(waitAction.id);
  $("#pa-defend").onclick = () => defendAction && playAct(defendAction.id);

  // game over banner
  const over = $("#play-over");
  if (st.status === "over") {
    over.classList.remove("hidden");
    const ob = $("#play-over-banner");
    if (st.winner == null) {
      ob.textContent = `Draw after ${st.round} rounds.`;
      ob.className = "verdict";
    } else if (st.you_won) {
      ob.textContent = `Victory! Your ${you} army prevails after ${st.round} rounds.`;
      ob.className = "verdict h1";
    } else {
      ob.textContent = `Defeat — ${st.agent} wins after ${st.round} rounds.`;
      ob.className = "verdict h0";
    }
    $("#btn-save-game").disabled = Boolean(st.saved_to);
    $("#play-hint").textContent = st.saved_to ? `saved to ${st.saved_to}` : "";
  } else {
    over.classList.add("hidden");
    const active = st.stacks.find((s) => s.uid === st.active);
    $("#play-hint").textContent = active
      ? `to act: ${active.unit} ×${active.count} (speed ${active.speed})`
      : "";
  }

  drawPlayBoard(st);
  state.playSeen = st.events.length;
}

function impairedMark(parent, ox, oy) {
  parent.append(el("circle", { cx: ox + 13, cy: oy + CELL - 16,
    r: 5.5, class: "impaired-mark" }));
  parent.append(el("line", { x1: ox + 9.2, y1: oy + CELL - 12.2,
    x2: ox + 16.8, y2: oy + CELL - 19.8, class: "impaired-mark" }));
}

function stackGroup(s, { active = false, attack = null } = {}) {
  const gcls = ["stack-g"];
  if (attack) {
    gcls.push("target-group",
      attack.type === "RANGED_ATTACK" ? "target-ranged" : "target-melee");
  }
  const g = el("g", { class: gcls.join(" ") });
  g.style.transform = `translate(${s.x * CELL}px, ${s.y * CELL}px)`;
  g._pos = { x: s.x, y: s.y };
  g._maxhp = s.max_hp;

  const cls = ["stack-tile", `side${s.side}`];
  if (active) cls.push("active");
  if (attack) cls.push("targetable");
  g.append(el("rect", { x: 4, y: 4, width: CELL - 8, height: CELL - 8, rx: 6,
    class: cls.join(" ") }));
  g.append(el("image", { href: unitIcon(s.unit), x: 12, y: 11,
    width: 30, height: 30, class: "stack-icon" }));
  g._count = el("text", { x: CELL / 2, y: CELL - 9, class: "stack-count" },
    `×${s.count}`);
  g.append(g._count);
  g.append(el("rect", { x: 8, y: 7, width: CELL - 16, height: 3,
    class: "hp-track", rx: 1.5 }));
  g._hp = el("rect", { x: 8, y: 7,
    width: Math.max((CELL - 16) * (s.top_hp / s.max_hp), 0), height: 3,
    class: "hp-fill", rx: 1.5 });
  g.append(g._hp);
  if (s.defending) {
    g.append(el("circle", { cx: CELL - 11, cy: 12, r: 4, class: "defend-mark" }));
  }
  if (s.impaired) impairedMark(g, 0, 0);

  let tip = `${s.unit} ×${s.count} · ${s.top_hp}/${s.max_hp}hp top` +
    (s.defending ? " · defending" : "") +
    (s.impaired ? " · shooter blocked: melee only, half damage" : "");
  if (attack) {
    const verb = attack.type === "RANGED_ATTACK" ? "shoot" : "strike";
    const movesIn = attack.from &&
      (attack.from.x !== s.x || attack.from.y !== s.y);
    tip += `\n${verb}: ~${attack.est} dmg` +
      (attack.retaliates ? " · will retaliate" : "") +
      (movesIn ? ` · from (${attack.from.x},${attack.from.y})` : "");
    g.addEventListener("click", () => playAct(attack.id));
  }
  g.append(el("title", {}, tip));
  return g;
}

function drawPlayBoard(st) {
  const svg = $("#play-board");
  const { board, obstacles } = st;
  svg.setAttribute("viewBox", `0 0 ${board.w * CELL} ${board.h * CELL}`);
  svg.innerHTML = "";

  for (let y = 0; y < board.h; y++) {
    for (let x = 0; x < board.w; x++) {
      svg.append(el("rect", {
        x: x * CELL, y: y * CELL, width: CELL, height: CELL,
        class: `cell${(x + y) % 2 ? " dark" : ""}` }));
    }
  }
  for (const o of obstacles) {
    svg.append(el("rect", {
      x: o.x * CELL + 3, y: o.y * CELL + 3, width: CELL - 6, height: CELL - 6,
      class: "obstacle", rx: 4 }));
  }

  const isMeleeType = (t) => t.startsWith("MELEE_");
  // enemy-click default: ranged if present, else the is_default melee side
  const attacks = new Map();
  for (const a of st.legal) {
    if (a.type === "RANGED_ATTACK") { attacks.set(a.target_uid, a); }
  }
  for (const a of st.legal) {
    if (isMeleeType(a.type) && a.is_default && !attacks.has(a.target_uid)) {
      attacks.set(a.target_uid, a);
    }
  }

  // move overlays under the stacks
  for (const a of st.legal) {
    if (a.type !== "MOVE") continue;
    svg.append(el("rect", {
      x: a.x * CELL + 6, y: a.y * CELL + 6, width: CELL - 12, height: CELL - 12,
      rx: 5, class: "move-overlay",
      onclick: () => playAct(a.id) },
      el("title", {}, `move to (${a.x},${a.y})`)));
  }

  // directional melee: a clickable "strike from here" marker on each reachable
  // approach square, layered above the move overlays.
  for (const a of st.legal) {
    if (!isMeleeType(a.type) || !a.from) continue;
    const cx = a.from.x * CELL + CELL / 2;
    const cy = a.from.y * CELL + CELL / 2;
    svg.append(el("polygon", {
      points: `${cx},${cy - 9} ${cx + 9},${cy} ${cx},${cy + 9} ${cx - 9},${cy}`,
      class: "approach-marker" + (a.is_default ? " default" : ""),
      onclick: (ev) => { ev.stopPropagation(); playAct(a.id); } },
      el("title", {}, `strike from (${a.from.x},${a.from.y}) · ~${a.est} dmg`)));
  }

  state.playGroups = new Map();
  for (const s of st.stacks) {
    const g = stackGroup(s, {
      active: s.uid === st.active && st.status !== "over",
      attack: attacks.get(s.uid) || null,
    });
    state.playGroups.set(s.uid, g);
    svg.append(g);
  }
}

function setupPlay() {
  $("#btn-start-game").addEventListener("click", startGame);
  $("#btn-new-game").addEventListener("click", () => {
    state.play = null;
    state.playGroups = null;
    state.playSeen = 0;
    $("#play-layout").classList.add("hidden");
    $("#play-setup").classList.remove("hidden");
  });
  $("#btn-save-game").addEventListener("click", async () => {
    if (!state.play) return;
    try {
      const { file } = await api(`/api/games/${state.play.id}/save`, { method: "POST" });
      $("#play-hint").textContent = `saved to ${file}`;
      $("#btn-save-game").disabled = true;
    } catch (e) { toast(`save failed: ${e.message}`); }
  });
}

async function resumeLatestGame() {
  if (state.play) return;
  try {
    const open = (await api("/api/games")).find((g) => g.status === "your-turn");
    if (open) renderPlay(await api(`/api/games/${open.id}`));
  } catch { /* no resume, fine */ }
}

/* ================================================================== */
/* tabs + boot */

function switchTab(name) {
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.tab === name));
  document.querySelectorAll(".tab-pane").forEach((p) =>
    p.classList.toggle("active", p.id === `tab-${name}`));
  if (name === "replays") refreshReplaySources();
  if (name === "play") resumeLatestGame();
}

async function boot() {
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => switchTab(t.dataset.tab)));
  setupViewer();
  try {
    state.meta = await api("/api/meta");
    $("#brand-status").textContent = "ready";
  } catch (e) {
    $("#brand-status").textContent = "api unreachable";
    toast(`cannot reach the tactica server: ${e.message}`);
    return;
  }
  const legend = $("#unit-legend");
  for (const u of state.meta.units) {
    legend.append(el("span", {},
      el("img", { class: "q-icon", src: unitIcon(u.name), alt: u.glyph }),
      ` ${u.name}${u.ranged ? " ↑" : ""}${u.flyer ? " ✴" : ""}`));
  }
  const scSel = $("#pg-scenario");
  for (const sc of Object.keys(state.meta.scenarios)) {
    scSel.append(el("option", {}, sc));
  }
  const dl = $("#agent-specs");
  for (const spec of state.meta.agent_examples) {
    dl.append(el("option", { value: spec }));
  }
  setupConfigPanel();
  setupPlay();
  await refreshPresets();
  await refreshJobs();
  setInterval(refreshJobs, 4000);
}

boot();
