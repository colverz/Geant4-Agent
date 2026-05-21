const $ = (id) => document.getElementById(id);

const state = {
  sessionId: localStorage.getItem("g4_session_id") || "",
  lang: localStorage.getItem("g4_lang") || "zh",
  runtime: null,
  modelConfigPath: localStorage.getItem("g4_ollama_config_path") || "",
  lastDesign: null,
  lastRecommendedConfig: null,
  candidateStatus: null,
  lastRuntimeState: null,
  lastRuntimeReport: null,
  lastTrace: null,
  sending: false,
};

const copy = {
  zh: {
    welcome: "描述你的模拟目标。我会先整理方案并生成候选配置，确认后再进入 Geant4 运行。",
    sendFailed: "这一轮请求失败",
    noResult: "还没有模拟结果。你可以说“批准运行 1000 events”，我会先校验当前方案再运行。",
    noConfig: "还没有可运行方案。请先描述模拟目标，我会给出推荐配置。",
    generalQuestion: "我会先把它理解成模拟目标来整理方案；如果只是普通问题，我不会写入配置或启动运行。",
    runIntent: (events) => `收到运行请求。我会使用当前方案，先校验，再运行 ${events} 个事件。`,
    viewerIntent: "收到 viewer 请求。我会先校验配置，再打开 Geant4 viewer。",
    retained: "已确认当前方案，并写入当前配置。你可以继续修改，或直接运行模拟。",
    validateOk: "配置预检通过，可以运行。",
    validateFailed: "配置还不能运行",
    runDone: "Geant4 运行完成",
    viewerDone: "Viewer 请求已处理",
    resetDone: "会话已重置。",
    designTitle: "我先把需求整理成一个模拟方案：",
    referenceTags: "参考标签",
    nextAction: "下一步",
    runnable: "可直接运行",
    approval: "需要确认",
    observables: "观测量",
    model: "推荐模型",
    goal: "目标",
  },
  en: {
    welcome: "Describe the simulation goal. I will design an executable setup before Geant4 runtime.",
    sendFailed: "This turn failed",
    noResult: "No Geant4 runtime result is available yet. Run the simulation first, then ask about the result.",
    noConfig: "No current configuration is available yet. Please describe a simulation goal first.",
    generalQuestion: "This does not look like a config change, result question, or runtime request. I will not write config. Describe geometry, material, source, physics, or output changes directly.",
    runIntent: (events) => `Confirmed runtime request. I will commit the candidate, validate the configuration, then run ${events} event${events === 1 ? "" : "s"}.`,
    viewerIntent: "Confirmed viewer request. I will validate the configuration, then open the Geant4 viewer.",
    retained: "The current design has been accepted and committed to the session. You can modify it or run the simulation.",
    validateOk: "Runtime preflight passed.",
    validateFailed: "The configuration is not runnable yet",
    runDone: "Geant4 run completed",
    viewerDone: "Viewer request handled",
    resetDone: "Session reset.",
    designTitle: "I have converted the request into a simulation design:",
    referenceTags: "Reference tags",
    nextAction: "Next",
    runnable: "Directly runnable",
    approval: "Needs approval",
    observables: "Observables",
    model: "Recommended model",
    goal: "Goal",
  },
};

const activityLabels = {
  intent: { zh: "识别意图", en: "Classify intent" },
  design: { zh: "生成模拟方案", en: "Build simulation design" },
  capability: { zh: "校验能力边界", en: "Check capabilities" },
  config: { zh: "生成推荐配置", en: "Draft config" },
  accept: { zh: "写入当前配置", en: "Commit candidate" },
  validate: { zh: "Runtime 预检", en: "Runtime preflight" },
  run: { zh: "运行模拟", en: "Run simulation" },
  summary: { zh: "读取结果", en: "Read result" },
};

function text(key, ...args) {
  const value = copy[state.lang]?.[key] || copy.en[key] || key;
  return typeof value === "function" ? value(...args) : value;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function asList(value) {
  return Array.isArray(value) ? value.filter((item) => String(item || "").trim()) : [];
}

function boolText(value) {
  if (value === undefined || value === null) return "-";
  return state.lang === "zh" ? (value ? "是" : "否") : value ? "yes" : "no";
}

function nextActionText(action) {
  const zh = {
    build_candidate_config: "可以生成候选配置",
    ask_user_to_choose_approximation: "需要确认近似方案",
    unsupported_capability: "当前 runtime 不支持",
    needs_more_information: "需要补充信息",
  };
  const en = {
    build_candidate_config: "candidate config can be built",
    ask_user_to_choose_approximation: "approval is needed for approximation",
    unsupported_capability: "unsupported by current runtime",
    needs_more_information: "more information is required",
  };
  return (state.lang === "zh" ? zh : en)[action] || action || "-";
}

async function postJson(path, payload = {}) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const reason = data.message || data.error || data.errors?.join(", ") || `${response.status}`;
    throw new Error(reason);
  }
  return data;
}

async function getJson(path) {
  const response = await fetch(path);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(data.message || data.error || `${response.status}`);
  return data;
}

function setBusy(value) {
  state.sending = value;
  $("send-btn").disabled = value;
  $("activity-strip").hidden = !value;
  if (value) $("activity-strip").textContent = state.lang === "zh" ? "Agent 正在处理..." : "Agent is working...";
}

function ensureSession(id) {
  if (!id || id === state.sessionId) return;
  state.sessionId = id;
  localStorage.setItem("g4_session_id", id);
  renderHeader();
}

function clearWelcome() {
  const welcome = document.querySelector(".welcome");
  if (welcome) welcome.remove();
}

function appendMessage(role, html, options = {}) {
  clearWelcome();
  const node = document.createElement("article");
  node.className = `turn ${role}${options.variant ? ` ${options.variant}` : ""}`;
  const label = role === "user" ? (state.lang === "zh" ? "你" : "You") : "Agent";
  node.innerHTML = `
    <div class="turn-label">${escapeHtml(label)}</div>
    <div class="turn-body">${html}</div>
  `;
  $("timeline").appendChild(node);
  node.scrollIntoView({ block: "end", behavior: "smooth" });
  return node;
}

function activityHtml(items = []) {
  if (!items.length) return "";
  const rows = items
    .map((item) => {
      const label = activityLabels[item.stage]?.[state.lang] || item.stage;
      return `
        <div class="activity-step ${item.status || "done"}">
          <span class="activity-dot"></span>
          <div>
            <strong>${escapeHtml(label)}</strong>
            <p>${escapeHtml(item.detail || "")}</p>
          </div>
        </div>
      `;
    })
    .join("");
  return `<section class="activity-trace" data-agent-activity="visible"><div class="activity-title">Agent Activity</div>${rows}</section>`;
}

function appendAgent(markdownText, activity = [], variant = "") {
  const body = `<div class="agent-answer">${formatPlainText(markdownText)}</div>${activityHtml(activity)}`;
  return appendMessage("agent", body, { variant });
}

function formatPlainText(value) {
  return escapeHtml(value).replace(/\n/g, "<br />");
}

function renderHeader() {
  $("session-pill").textContent = state.sessionId ? `session: ${state.sessionId.slice(0, 12)}` : "new session";
  const metadata = state.lastRuntimeState?.metadata || {};
  $("adapter-pill").textContent = `adapter: ${metadata.adapter || "unknown"}`;
  $("phase-pill").textContent = state.lastRuntimeState?.runtime_phase || "idle";
  if (state.runtime?.current_model) {
    $("adapter-pill").textContent = `${state.runtime.current_provider || "llm"}: ${state.runtime.current_model}`;
  }
}

function renderDesignSummary() {
  const candidate = state.lastDesign || {};
  if (!state.lastDesign) {
    $("design-state").textContent = "none";
    $("design-summary").textContent = state.lang === "zh" ? "还没有方案。先描述一个模拟目标。" : "No design yet.";
    return;
  }
  const setup = candidate.recommended_setup || {};
  const check = candidate.capability_check || {};
  const status = state.candidateStatus?.status ? ` / ${state.candidateStatus.status}` : "";
  $("design-state").textContent = `${nextActionText(candidate.next_action)}${status}`;
  $("design-summary").innerHTML = `
    <div><span>${text("goal")}</span><strong>${escapeHtml(candidate.goal || "-")}</strong></div>
    <div><span>${text("model")}</span><strong>${escapeHtml([setup.geometry, setup.material, setup.source].filter(Boolean).join(" / ") || "-")}</strong></div>
    <div><span>${text("runnable")}</span><strong>${escapeHtml(boolText(check.supported))}</strong></div>
    <div><span>${text("approval")}</span><strong>${escapeHtml(boolText(check.requires_user_approval))}</strong></div>
  `;
}

function renderConfigReadiness(config = state.lastRecommendedConfig) {
  const checks = [
    ["geometry", !!config?.geometry?.structure],
    ["material", asList(config?.materials?.selected_materials).length > 0],
    ["source", !!config?.source?.type && !!config?.source?.particle],
    ["energy", config?.source?.energy !== undefined],
    ["physics", !!(config?.physics?.physics_list || config?.physics_list?.name || config?.physics_list)],
  ];
  $("config-state").textContent = checks.every(([, ok]) => ok) ? "ready" : "pending";
  $("config-readiness").innerHTML = checks
    .map(([name, ok]) => `<div class="readiness-item ${ok ? "ok" : "missing"}"><span>${name}</span><strong>${ok ? "ok" : "missing"}</strong></div>`)
    .join("");
}

function renderRuntimeResult(report = state.lastRuntimeReport) {
  if (!report) {
    $("result-state").textContent = "empty";
    $("result-summary").innerHTML = `<div class="empty-metric">${escapeHtml(text("noResult"))}</div>`;
    return;
  }
  const metrics = report.key_metrics || {};
  $("result-state").textContent = report.ok ? "ok" : "failed";
  $("result-summary").innerHTML = [
    ["events", `${report.events_completed ?? "-"} / ${report.events_requested ?? "-"}`],
    ["completion", report.completion_fraction === undefined ? "-" : Number(report.completion_fraction).toFixed(3)],
    ["target edep", metrics.target_edep_total_mev ?? "-"],
    ["detector crossing", metrics.detector_crossing_count ?? "-"],
    ["plane crossing", metrics.plane_crossing_count ?? "-"],
  ]
    .map(([label, value]) => `<div class="metric"><span>${label}</span><strong>${escapeHtml(value)}</strong></div>`)
    .join("");
}

function renderEvidence() {
  $("config-json").textContent = JSON.stringify(state.lastRecommendedConfig || {}, null, 2);
  $("runtime-json").textContent = JSON.stringify(state.lastRuntimeState || {}, null, 2);
  $("trace-json").textContent = JSON.stringify(state.lastTrace || {}, null, 2);
}

function updateAll() {
  renderHeader();
  renderDesignSummary();
  renderConfigReadiness();
  renderRuntimeResult();
  renderEvidence();
}

function designMessage(data) {
  const candidate = data.simulation_design || {};
  const setup = candidate.recommended_setup || {};
  const check = candidate.capability_check || {};
  const observables = asList(candidate.observables);
  const simplifications = asList(candidate.simplifications);
  const unsupported = asList(candidate.unsupported_capabilities);
  const decisions = asList(candidate.user_decisions_required);
  const refs = asList(candidate.knowledge_references);
  const config = data.recommended_config || {};
  const material = setup.material || asList(config.materials?.selected_materials).join(", ") || "-";
  const source = setup.source || [config.source?.particle, config.source?.energy ? `${config.source.energy} MeV` : ""].filter(Boolean).join(" ") || "-";
  const geometry = setup.geometry || config.geometry?.structure || "-";
  const rationale = setup.design_rationale ? String(setup.design_rationale) : "";
  const alternatives = asList(setup.alternatives_considered);
  const lines = [
    text("designTitle"),
    "",
    state.lang === "zh"
      ? `目标：${candidate.goal || "-"}`
      : `${text("goal")}: ${candidate.goal || "-"}`,
    state.lang === "zh"
      ? `推荐做法：用 ${geometry} 几何，主材料 ${material}，源模型 ${source}。`
      : `${text("model")}: ${geometry}; ${material}; ${source}`,
    state.lang === "zh"
      ? `要看的结果：${observables.join(", ") || "-"}`
      : `${text("observables")}: ${observables.join(", ") || "-"}`,
    state.lang === "zh"
      ? `能否直接形成可运行配置：${boolText(check.supported)}`
      : `${text("runnable")}: ${boolText(check.supported)}`,
    state.lang === "zh"
      ? `是否需要你先批准近似：${boolText(check.requires_user_approval)}`
      : `${text("approval")}: ${boolText(check.requires_user_approval)}`,
    state.lang === "zh"
      ? `下一步：${nextActionText(candidate.next_action)}`
      : `${text("nextAction")}: ${nextActionText(candidate.next_action)}`,
  ];
  if (simplifications.length) lines.push("", state.lang === "zh" ? "需要说明的近似：" : "Approximation:", ...simplifications.slice(0, 4).map((x) => `- ${x}`));
  if (rationale) lines.push("", state.lang === "zh" ? `为什么这样建模：${rationale}` : `Rationale: ${rationale}`);
  if (alternatives.length) lines.push("", state.lang === "zh" ? "我考虑过但没有优先采用的方案：" : "Alternatives considered:", ...alternatives.slice(0, 4).map((x) => `- ${x}`));
  if (unsupported.length) lines.push("", state.lang === "zh" ? "当前不支持：" : "Unsupported:", ...unsupported.slice(0, 4).map((x) => `- ${x}`));
  if (decisions.length) lines.push("", state.lang === "zh" ? "需要你裁定：" : "Decision required:", ...decisions.slice(0, 4).map((x) => `- ${x}`));
  if (refs.length) lines.push("", `${text("referenceTags")}: ${refs.slice(0, 6).join(", ")}`);
  if (state.lang === "zh") {
    lines.push("", "如果这个方案合理，直接回复“批准运行 1000 events”。如果不合理，告诉我要改材料、几何、源或观测量。");
  }
  return lines.join("\n");
}

function applyDesignResponse(data) {
  ensureSession(data.session_id);
  state.lastDesign = data.simulation_design || null;
  state.lastRecommendedConfig = data.recommended_config || data.config || state.lastRecommendedConfig;
  state.candidateStatus = data.candidate_status || state.candidateStatus;
  state.lastTrace = data.internal_trace || null;
  updateAll();
}

function runtimePatch() {
  return state.lastRecommendedConfig || {};
}

function actionToken(action, payload = {}) {
  const base = JSON.stringify({ action, payload, session: state.sessionId || "" });
  let hash = 0;
  for (let i = 0; i < base.length; i += 1) hash = (hash * 31 + base.charCodeAt(i)) >>> 0;
  return `${action}-${hash.toString(16)}`;
}

async function classifyIntent(inputText) {
  try {
    return await postJson("/api/geant4/intent", { text: inputText, lang: state.lang });
  } catch (_) {
    return { intent: "normal_chat", action_safety_class: "read_only" };
  }
}

function isAcceptCurrentDesignText(inputText) {
  const raw = String(inputText || "").trim().toLowerCase();
  return /^(就按这个|按这个继续|可以|确认|同意|没问题|生成配置|用这个方案)$/.test(raw)
    || /\b(use this|looks good|continue|confirm this|generate config|build config|keep this design)\b/.test(raw);
}

function parseRequestedEvents(inputText) {
  const match = String(inputText || "").match(/(\d+)\s*(events?|个事件|次)/i);
  if (!match) return 1000;
  return Math.max(1, Math.min(100000, Number.parseInt(match[1], 10)));
}

async function requestDesign(inputText) {
  return postJson("/api/simulation/design", {
    session_id: state.sessionId || null,
    text: inputText,
    strict_mode: true,
    lang: state.lang,
    llm_router: true,
    llm_question: false,
    geometry_pipeline: "v2",
    source_pipeline: "v2",
  });
}

async function acceptCandidate(options = {}) {
  if (!state.lastRecommendedConfig || !Object.keys(state.lastRecommendedConfig).length) return null;
  const data = await postJson("/api/simulation/accept", {
    session_id: state.sessionId || null,
    recommended_config: state.lastRecommendedConfig,
    source: options.source || "ui_accept_candidate",
  });
  ensureSession(data.session_id);
  state.lastRecommendedConfig = data.config || state.lastRecommendedConfig;
  state.candidateStatus = data.candidate_status || state.candidateStatus;
  renderConfigReadiness();
  renderDesignSummary();
  renderEvidence();
  if (!options.silent) {
    appendAgent(text("retained"), [{ stage: "accept", detail: "Candidate config committed to session.", status: "done" }]);
  }
  return data;
}

async function ensureCandidateCommitted() {
  if (!state.lastRecommendedConfig || !Object.keys(state.lastRecommendedConfig).length) return;
  if (state.candidateStatus?.committed) return;
  await acceptCandidate({ silent: true, source: "ui_auto_commit_before_runtime" });
}

async function answerRuntimeQuestion(inputText) {
  const data = await postJson("/api/geant4/summary", {
    lang: state.lang,
    question: inputText,
  });
  if (data.runtime_smoke_report) {
    state.lastRuntimeReport = data.runtime_smoke_report;
    renderRuntimeResult();
  }
  return data.runtime_result_explanation?.message || data.message || text("noResult");
}

async function answerConfigQuestion() {
  const data = await postJson("/api/config/summary", {
    session_id: state.sessionId || "",
    lang: state.lang,
  });
  if (data.config) {
    state.lastRecommendedConfig = data.config;
    renderConfigReadiness();
    renderEvidence();
  }
  return data.message || text("noConfig");
}

async function validateGeant4Config(events = 1, silent = false) {
  await ensureCandidateCommitted();
  const data = await postJson("/api/geant4/validate", {
    session_id: state.sessionId || null,
    patch: runtimePatch(),
    events,
  });
  const payload = data.payload || {};
  if (!payload.ok && !silent) {
    appendAgent(`${text("validateFailed")}: ${(payload.missing_paths || data.errors || []).join(", ") || "unknown"}`, [
      { stage: "validate", detail: "Runtime preflight rejected missing or invalid fields.", status: "warn" },
    ], "warning");
  }
  if (payload.ok && !silent) {
    appendAgent(text("validateOk"), [{ stage: "validate", detail: "All required runtime fields are present.", status: "done" }]);
  }
  return payload;
}

async function runGeant4(events = 1) {
  await ensureCandidateCommitted();
  const preflight = await validateGeant4Config(events, true);
  if (!preflight.ok) {
    appendAgent(`${text("validateFailed")}: ${(preflight.missing_paths || []).join(", ") || "unknown"}`, [
      { stage: "validate", detail: "Run stopped before subprocess/runtime execution.", status: "warn" },
    ], "warning");
    return;
  }
  const patch = runtimePatch();
  const data = await postJson("/api/geant4/run", {
    session_id: state.sessionId || null,
    patch,
    events,
    action_id: actionToken("run_beam", { events, patch }),
  });
  state.lastRuntimeReport = data.runtime_smoke_report || state.lastRuntimeReport;
  renderRuntimeResult();
  await refreshGeant4State();
  const message = data.runtime_result_explanation?.message || data.message || text("runDone");
  appendAgent(`${text("runDone")}\n${message}`, [
    { stage: "accept", detail: "Candidate config is synchronized with the session.", status: "done" },
    { stage: "validate", detail: "Runtime preflight passed.", status: "done" },
    { stage: "run", detail: `${events} event(s) requested.`, status: "done" },
    { stage: "summary", detail: "Structured runtime report returned.", status: "done" },
  ]);
}

async function openViewer() {
  await ensureCandidateCommitted();
  const patch = runtimePatch();
  const data = await postJson("/api/geant4/viewer/open", {
    session_id: state.sessionId || null,
    patch,
    events: 12,
    action_id: actionToken("viewer_open", { patch }),
  });
  state.lastRuntimeReport = data.runtime_smoke_report || state.lastRuntimeReport;
  renderRuntimeResult();
  appendAgent(`${text("viewerDone")}: ${data.message || "ok"}`, [
    { stage: "validate", detail: "Viewer preflight completed.", status: "done" },
    { stage: "run", detail: "Viewer action was explicitly requested.", status: "done" },
  ]);
}

async function refreshGeant4State() {
  const data = await getJson("/api/geant4/state");
  state.lastRuntimeState = data;
  updateAll();
}

async function refreshGeant4Log() {
  const data = await postJson("/api/geant4/log", {});
  $("log-json").textContent = JSON.stringify(data, null, 2);
}

async function loadRuntimeConfig() {
  const data = await getJson("/api/runtime");
  state.runtime = data;
  const select = $("model-config-select");
  select.innerHTML = "";
  const configs = Array.isArray(data.available) ? data.available : [];
  if (!configs.length) {
    const option = document.createElement("option");
    option.value = data.current_path || "";
    option.textContent = data.current_model || "default";
    select.appendChild(option);
  }
  for (const item of configs) {
    const path = item.path || "";
    const option = document.createElement("option");
    option.value = path;
    option.textContent = `${item.provider || "llm"} / ${item.model || path.split(/[\\/]/).pop() || path}`;
    option.selected = path === data.current_path || path === state.modelConfigPath;
    select.appendChild(option);
  }
  renderHeader();
}

async function setRuntimeConfig(path) {
  if (!path) return;
  const data = await postJson("/api/runtime", { config_path: path, ollama_config_path: path });
  state.modelConfigPath = path;
  localStorage.setItem("g4_ollama_config_path", path);
  state.runtime = data;
  await loadRuntimeConfig();
  appendAgent(state.lang === "zh" ? `模型配置已切换：${data.current_model || path}` : `Model config switched: ${data.current_model || path}`);
}

async function sendPrompt() {
  if (state.sending) return;
  const input = $("prompt-input").value.trim();
  if (!input) return;
  appendMessage("user", escapeHtml(input));
  $("prompt-input").value = "";
  setBusy(true);

  try {
    const intent = await classifyIntent(input);
    const baseActivity = [{ stage: "intent", detail: `${intent.intent || "normal_chat"} / ${intent.action_safety_class || "unknown"}` }];

    if (intent.intent === "read_summary") {
      const answer = await answerRuntimeQuestion(input);
      appendAgent(answer, [...baseActivity, { stage: "summary", detail: "Read-only runtime summary path.", status: "done" }]);
      return;
    }
    if (intent.intent === "read_config") {
      const answer = await answerConfigQuestion();
      appendAgent(answer, [...baseActivity, { stage: "summary", detail: "Read-only config summary path.", status: "done" }]);
      return;
    }
    if (intent.intent === "run_requested") {
      const events = parseRequestedEvents(input);
      if (!state.lastRecommendedConfig || !Object.keys(state.lastRecommendedConfig).length) {
        const designData = await requestDesign(input);
        applyDesignResponse(designData);
        appendAgent(`${designMessage(designData)}\n\n${text("runIntent", events)}`, [
          ...baseActivity,
          { stage: "design", detail: "No runnable candidate existed, so the agent designed one first.", status: "done" },
          { stage: "capability", detail: `next_action=${designData.simulation_design?.next_action || "unknown"}`, status: "done" },
          { stage: "config", detail: designData.recommended_config ? "Recommended config draft is available." : "No runnable config draft.", status: designData.recommended_config ? "done" : "warn" },
        ]);
        if (!state.lastRecommendedConfig || !Object.keys(state.lastRecommendedConfig).length) {
          appendAgent(
            state.lang === "zh"
              ? "这个目标现在还不能直接运行，因为方案需要你先批准近似或补充关键条件。我不会把不合理方案硬塞进 Geant4。"
              : "This goal is not directly runnable yet because the design needs approval or key missing information.",
            [{ stage: "capability", detail: "Runtime execution stopped before preflight.", status: "warn" }],
            "warning"
          );
          return;
        }
      } else {
        appendAgent(text("runIntent", events), baseActivity);
      }
      await runGeant4(events);
      return;
    }
    if (intent.intent === "viewer_requested") {
      appendAgent(text("viewerIntent"), baseActivity);
      await openViewer();
      return;
    }
    if (state.lastDesign && isAcceptCurrentDesignText(input)) {
      await acceptCandidate({ source: "ui_explicit_accept_candidate" });
      return;
    }
    if (intent.intent !== "config_mutation" && intent.intent !== "normal_chat") {
      appendAgent(text("generalQuestion"), baseActivity, "warning");
      return;
    }

    const data = await requestDesign(input);
    applyDesignResponse(data);
    appendAgent(designMessage(data), [
      ...baseActivity,
      { stage: "design", detail: data.simulation_design_source || "simulation_design stage returned a candidate.", status: "done" },
      { stage: "capability", detail: `next_action=${data.simulation_design?.next_action || "unknown"}`, status: "done" },
      { stage: "config", detail: data.recommended_config ? "Recommended config draft is available." : "No recommended config draft.", status: data.recommended_config ? "done" : "warn" },
    ]);
  } catch (error) {
    appendAgent(`${text("sendFailed")}: ${error.message}`, [{ stage: "summary", detail: "Request stopped with an error.", status: "warn" }], "error");
  } finally {
    setBusy(false);
    await refreshGeant4State().catch(() => {});
  }
}

async function resetSession() {
  if (state.sessionId) {
    await postJson("/api/reset", { session_id: state.sessionId }).catch(() => {});
  }
  localStorage.removeItem("g4_session_id");
  state.sessionId = "";
  state.lastDesign = null;
  state.lastRecommendedConfig = null;
  state.candidateStatus = null;
  state.lastRuntimeReport = null;
  state.lastTrace = null;
  $("timeline").innerHTML = `
    <article class="welcome">
      <p class="caption">Start here</p>
      <h2>${escapeHtml(text("welcome"))}</h2>
      <p>10 mm x 20 mm x 30 mm copper box target; gamma point source 1 MeV at (0,0,-20) mm along +z; physics FTFP_BERT.</p>
    </article>
  `;
  updateAll();
  appendAgent(text("resetDone"));
}

function bindEvents() {
  $("send-btn").addEventListener("click", sendPrompt);
  $("prompt-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendPrompt();
    }
  });
  $("lang-select").addEventListener("change", (event) => {
    state.lang = event.target.value;
    localStorage.setItem("g4_lang", state.lang);
    $("prompt-input").placeholder =
      state.lang === "zh"
        ? "输入模拟目标、修改意见或结果问题。Enter 发送，Shift+Enter 换行。"
        : "Enter a simulation goal, modification, or result question. Enter to send, Shift+Enter for newline.";
    updateAll();
  });
  $("model-config-select").addEventListener("change", (event) => setRuntimeConfig(event.target.value).catch((error) => appendAgent(error.message, [], "error")));
  $("refresh-btn").addEventListener("click", () => Promise.all([refreshGeant4State(), refreshGeant4Log()]).catch((error) => appendAgent(error.message, [], "error")));
  $("reset-btn").addEventListener("click", () => resetSession().catch((error) => appendAgent(error.message, [], "error")));
  document.querySelectorAll(".evidence-tab").forEach((button) => {
    button.addEventListener("click", () => {
      document.querySelectorAll(".evidence-tab").forEach((tab) => tab.classList.toggle("active", tab === button));
      document.querySelectorAll(".evidence-panel").forEach((panel) => {
        panel.hidden = panel.id !== button.dataset.evidence;
      });
    });
  });
  $("window-min-btn").addEventListener("click", () => window.geant4Desktop?.minimize?.());
  $("window-max-btn").addEventListener("click", () => window.geant4Desktop?.toggleMaximize?.());
  $("window-close-btn").addEventListener("click", () => {
    if (window.geant4Desktop?.close) window.geant4Desktop.close();
    else window.close();
  });
}

async function boot() {
  $("lang-select").value = state.lang;
  bindEvents();
  updateAll();
  await loadRuntimeConfig().catch((error) => appendAgent(`Runtime config load failed: ${error.message}`, [], "error"));
  await refreshGeant4State().catch(() => {});
  await refreshGeant4Log().catch(() => {});
}

boot();
