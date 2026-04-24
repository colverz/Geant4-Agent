const $ = (id) => document.getElementById(id);

const state = {
  sessionId: localStorage.getItem("g4_session_id") || "",
  lang: localStorage.getItem("g4_lang") || "zh",
  ollamaConfigPath: localStorage.getItem("g4_ollama_config_path") || "",
  llmProvider: "",
  ollamaModel: "",
  modelPreflight: null,
  lastMeta: null,
  lastProcess: null,
  geant4State: null,
  lastRuntimeSmokeReport: null,
  sending: false,
  activeThinkingNode: null,
  activeThinkingProgress: [],
  geometryPipeline: localStorage.getItem("g4_geometry_pipeline") || "legacy",
  sourcePipeline: localStorage.getItem("g4_source_pipeline") || "legacy",
};

const i18n = {
  zh: {
    brand_subtitle: "面向 Geant4 规划与运行控制的对话工作台。",
    session_title: "会话",
    session_label: "会话",
    phase_label: "阶段",
    model_label: "模型",
    controls_title: "控制",
    model_config: "模型配置",
    language: "语言",
    confidence: "结构置信度",
    autofix: "自动修正",
    llm_routing: "LLM 路由",
    llm_follow_up: "LLM 追问",
    reset: "重置会话",
    summary: "摘要",
    conversation: "对话",
    conversation_title: "实验规划对话",
    send: "发送",
    incomplete: "未完成",
    complete: "已完成",
    offline: "离线",
    sync_config: "同步配置",
    open_viewer: "打开几何窗口",
    initialize: "初始化",
    run_one: "运行 1 个事件",
    run_ten: "运行 10 个事件",
    refresh: "刷新",
    input_placeholder: "描述你想构建的实验。按 Enter 发送，Shift+Enter 换行。",
    composer_hint: "先在对话中收敛需求，再将配置同步到 Geant4。",
    composer_busy: "正在逐步推进行内分析与配置生成...",
    runtime_ready: "模型资源已就绪。",
    runtime_incomplete: "模型资源不完整。",
    empty_state: "从一句自然语言开始。Agent 会先整理实验结构，再在准备好后同步到 Geant4。",
    you: "你",
    agent: "Agent",
    system: "系统",
    config: "配置",
    runtime_state: "运行状态",
    runtime_log: "运行日志",
    runtime_result: "运行结果",
    no_runtime_result: "尚无运行结果。完成一次 Geant4 run 后这里会显示摘要。",
    process: "过程",
    internal_trace: "内部轨迹",
    geometry_compare: "几何对比",
    thinking: "思考过程",
    thinking_hint: "我会实时告诉你当前正在处理什么，以及下一步准备推进到哪里。",
    request_failed: "请求失败，未能完成本轮渲染。",
    model_switched: "模型已切换",
    geant4_prefix: "Geant4",
    phase: "阶段",
    complete_state: "完成",
    geometry: "几何",
    materials: "材料",
    particle: "粒子",
    source_type: "源类型",
    physics_list: "物理过程",
    output: "输出",
    pending_asks: "待追问",
    new_session: "新会话",
    progress_labels: {
      queued: "排队中",
      loading_runtime: "唤起运行环境",
      runtime_ready: "运行环境就绪",
      start: "读取本轮请求",
      intent: "理解你的目标",
      slot_frame: "整理结构槽位",
      semantic_frame: "补足语义骨架",
      semantic_extract: "提取关键要素",
      normalize: "统一内部表达",
      candidate_merge: "合并更新候选",
      arbitration: "处理冲突与优先级",
      validation: "校验当前配置",
      dialogue: "组织回复内容",
      finalize: "写回会话状态",
      completed: "已完成",
      failed: "已失败",
      runtime_unavailable: "运行环境不可用",
    },
    progress_details: {
      queued: "请求已经进入处理队列，马上开始这一轮分析。",
      loading_runtime: "正在准备本轮需要的 runtime 组件和模型资源。",
      runtime_ready: "后端环境已经就绪，可以进入结构化分析。",
      start: "先把你刚才的输入拆开，确认这一轮真正要处理的重点。",
      intent: "判断你是在补充细节、覆盖旧设置，还是准备进入运行阶段。",
      slot_frame: "把请求折叠成可更新的结构槽位，方便后续稳定落盘。",
      semantic_frame: "当槽位信号还不够强时，先用更宽的语义骨架稳住上下文。",
      semantic_extract: "从描述里抓取 geometry、material、source 和 physics 等关键内容。",
      normalize: "把自然语言里的不同说法，统一成内部能持续处理的表达。",
      candidate_merge: "把新识别到的内容和当前会话状态合并，避免信息断层。",
      arbitration: "处理互相覆盖或冲突的字段，确定这轮最终应当采用的版本。",
      validation: "检查当前配置是否仍缺关键项，或者存在不合理组合。",
      dialogue: "把这一轮结果整理成你更容易读懂的回复和界面状态。",
      finalize: "写回最新会话状态，让下一轮可以直接接着推进。",
      completed: "这一轮已经处理完成。",
      failed: "这一轮在完成前中断了，界面会停在最近一步。",
      runtime_unavailable: "运行环境暂时没有准备好，因此这轮无法继续。",
    },
  },
  en: {
    brand_subtitle: "Dialogue workspace for simulation planning and runtime control.",
    session_title: "Session",
    session_label: "Session",
    phase_label: "Phase",
    model_label: "Model",
    controls_title: "Controls",
    model_config: "Model Config",
    language: "Language",
    confidence: "Structure Confidence",
    autofix: "Auto-fix",
    llm_routing: "LLM routing",
    llm_follow_up: "LLM follow-up",
    reset: "Reset session",
    summary: "Summary",
    conversation: "Conversation",
    conversation_title: "Simulation Planning Chat",
    send: "Send",
    incomplete: "incomplete",
    complete: "complete",
    offline: "offline",
    sync_config: "Sync Config",
    open_viewer: "Open Viewer",
    initialize: "Initialize",
    run_one: "Run 1 Event",
    run_ten: "Run 10 Events",
    refresh: "Refresh",
    input_placeholder: "Describe the experiment you want to build. Press Enter to send, Shift+Enter for a new line.",
    composer_hint: "Build the requirement in dialogue first. Use the runtime buttons after the structure looks right.",
    composer_busy: "Streaming live progress through the orchestration stages...",
    runtime_ready: "Model assets look ready.",
    runtime_incomplete: "Model assets are incomplete.",
    empty_state: "Start with a natural-language request. The agent will convert it into a structured experiment plan, then you can sync it into Geant4 when ready.",
    you: "You",
    agent: "Agent",
    system: "System",
    config: "Config",
    runtime_state: "Runtime State",
    runtime_log: "Runtime Log",
    runtime_result: "Runtime Result",
    no_runtime_result: "No runtime result yet. Complete a Geant4 run to see the summary here.",
    process: "Process",
    internal_trace: "Internal Trace",
    geometry_compare: "Geometry Compare",
    thinking: "Thinking",
    thinking_hint: "This panel updates live so you can see what is being processed now and what comes next.",
    request_failed: "The request failed before a response could be rendered.",
    model_switched: "Model switched",
    geant4_prefix: "Geant4",
    phase: "phase",
    complete_state: "complete",
    geometry: "geometry",
    materials: "materials",
    particle: "particle",
    source_type: "source type",
    physics_list: "physics list",
    output: "output",
    pending_asks: "pending asks",
    new_session: "new",
    progress_labels: {
      queued: "Queued",
      loading_runtime: "Warming up runtime",
      runtime_ready: "Runtime ready",
      start: "Reading request",
      intent: "Understanding intent",
      slot_frame: "Structuring slots",
      semantic_frame: "Stabilizing semantics",
      semantic_extract: "Extracting key signals",
      normalize: "Normalizing phrasing",
      candidate_merge: "Merging candidates",
      arbitration: "Resolving conflicts",
      validation: "Validating config",
      dialogue: "Composing reply",
      finalize: "Finalizing state",
      completed: "Completed",
      failed: "Failed",
      runtime_unavailable: "Runtime unavailable",
    },
    progress_details: {
      queued: "The request is in line and about to enter the orchestration flow.",
      loading_runtime: "Preparing the runtime pieces and model resources needed for this turn.",
      runtime_ready: "The backend environment is ready, so the structured pass can begin.",
      start: "Reading the latest message and isolating what this turn is really changing.",
      intent: "Deciding whether this turn adds detail, overrides prior settings, or prepares a run.",
      slot_frame: "Packing the request into a stable slot layout for downstream updates.",
      semantic_frame: "Using a broader semantic scaffold when the slot signal is still weak.",
      semantic_extract: "Pulling geometry, material, source, and physics clues out of the prompt.",
      normalize: "Converting different phrasings into one consistent internal representation.",
      candidate_merge: "Merging newly extracted updates with the state already held in session.",
      arbitration: "Resolving overlaps and priorities before anything is committed.",
      validation: "Checking for missing requirements or invalid combinations in the current config.",
      dialogue: "Turning the result into a concise assistant response and UI update.",
      finalize: "Writing the newest state back so the next turn can continue from here.",
      completed: "This turn has finished successfully.",
      failed: "The turn stopped before completion, so the latest reachable stage is shown.",
      runtime_unavailable: "The runtime is not ready, so this turn cannot proceed yet.",
    },
  },
};

Object.assign(i18n.zh, {
  config_overview: "配置概览",
  completion_status: "完成情况",
  filled_items: "已填项",
  missing_items: "缺失项",
  raw_config: "原始配置",
  raw_runtime: "原始运行态",
  terminal_log: "终端日志",
  debug_panel: "调试信息",
  geometry_compare: "几何对比",
});

Object.assign(i18n.en, {
  config_overview: "Configuration Overview",
  completion_status: "Completion Status",
  filled_items: "Filled",
  missing_items: "Missing",
  raw_config: "Raw Config",
  raw_runtime: "Raw Runtime",
  terminal_log: "Terminal",
  debug_panel: "Debug Panel",
  geometry_compare: "Geometry Compare",
});

function t(key) {
  return i18n[state.lang][key] || key;
}

function progressLabel(stage, fallback) {
  return i18n[state.lang].progress_labels[stage] || fallback || stage;
}

function progressDetail(stage, fallback) {
  return i18n[state.lang].progress_details?.[stage] || fallback || "";
}

function animateIn(element) {
  element.animate(
    [
      { opacity: 0, transform: "translateY(10px)" },
      { opacity: 1, transform: "translateY(0)" },
    ],
    { duration: 220, easing: "cubic-bezier(.2,.8,.2,1)" },
  );
}

function escapeHtml(text) {
  return String(text || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function ensureEmptyState() {
  const chat = $("chat");
  if (chat.querySelector(".message") || chat.querySelector(".empty-state")) return;
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.textContent = t("empty_state");
  chat.appendChild(empty);
}

function addMessage(role, text, variant = "") {
  const chat = $("chat");
  const empty = chat.querySelector(".empty-state");
  if (empty) empty.remove();

  const item = document.createElement("article");
  item.className = `message ${role} ${variant}`.trim();

  const meta = document.createElement("div");
  meta.className = "message-meta";
  meta.textContent = role === "user" ? t("you") : variant === "system" ? t("system") : t("agent");

  const body = document.createElement("div");
  body.className = "message-body";
  body.textContent = text;

  item.appendChild(meta);
  item.appendChild(body);
  chat.appendChild(item);
  animateIn(item);
  chat.scrollTop = chat.scrollHeight;
  return item;
}

function createThinkingMessage() {
  const item = addMessage("assistant", "", "thinking");
  const body = item.querySelector(".message-body");
  body.innerHTML = `
    <div class="live-thinking-head">${t("thinking")}</div>
    <div class="live-thinking-subhead">${t("thinking_hint")}</div>
    <div class="live-thinking-focus">
      <div class="live-thinking-focus-label">${progressLabel("queued")}</div>
      <div class="live-thinking-focus-detail">${progressDetail("queued")}</div>
      <div class="live-thinking-sheen"></div>
    </div>
    <div class="live-thinking-list"></div>
  `;
  state.activeThinkingNode = item;
  state.activeThinkingProgress = [];
  return item;
}

function renderThinkingProgress(progress = []) {
  if (!state.activeThinkingNode) return;
  state.activeThinkingProgress = Array.isArray(progress) ? [...progress] : [];
  const latest = progress[progress.length - 1] || { stage: "queued" };
  const focusLabel = state.activeThinkingNode.querySelector(".live-thinking-focus-label");
  const focusDetail = state.activeThinkingNode.querySelector(".live-thinking-focus-detail");
  const list = state.activeThinkingNode.querySelector(".live-thinking-list");
  if (!list || !focusLabel || !focusDetail) return;
  focusLabel.textContent = progressLabel(latest.stage, latest.label);
  focusDetail.textContent = progressDetail(latest.stage, latest.detail);
  list.innerHTML = "";
  const recent = progress.slice(-5);
  for (let i = 0; i < recent.length; i += 1) {
    const item = recent[i];
    const isCurrent = i === recent.length - 1;
    const row = document.createElement("div");
    row.className = `live-thinking-row ${isCurrent ? "current" : "done"}`;
    row.innerHTML = `
      <span class="live-thinking-dot"></span>
      <div class="live-thinking-copy">
        <div class="live-thinking-label">${progressLabel(item.stage, item.label)}</div>
        <div class="live-thinking-detail">${progressDetail(item.stage, item.detail)}</div>
      </div>
    `;
    list.appendChild(row);
  }
}

function buildThinkingArchive(progress = []) {
  if (!progress.length) return "";
  const steps = progress
    .slice(-6)
    .map((item) => {
      const label = escapeHtml(progressLabel(item.stage, item.label));
      const detail = escapeHtml(progressDetail(item.stage, item.detail));
      return `
        <div class="thinking-archive-row">
          <span class="thinking-archive-dot"></span>
          <div class="thinking-archive-copy">
            <div class="thinking-archive-label">${label}</div>
            <div class="thinking-archive-detail">${detail}</div>
          </div>
        </div>
      `;
    })
    .join("");
  return `
    <details class="thinking-archive">
      <summary class="thinking-archive-summary">
        <span class="thinking-archive-title">${t("thinking")}</span>
        <span class="thinking-archive-caption">${escapeHtml(progressLabel(progress[progress.length - 1].stage))}</span>
      </summary>
      <div class="thinking-archive-list">${steps}</div>
    </details>
  `;
}

function finalizeThinkingMessage(finalText) {
  if (!state.activeThinkingNode) {
    addMessage("assistant", finalText);
    return;
  }
  const node = state.activeThinkingNode;
  const body = node.querySelector(".message-body");
  const archive = buildThinkingArchive(state.activeThinkingProgress);
  node.querySelector(".message-meta").textContent = t("agent");
  node.className = "message assistant settling";
  const finish = () => {
    body.innerHTML = `
      <div class="message-answer">${escapeHtml(finalText)}</div>
      ${archive}
    `;
    node.className = "message assistant";
    const archiveNode = body.querySelector(".thinking-archive");
    if (archiveNode) {
      archiveNode.animate(
        [
          { opacity: 0, transform: "translateY(-4px)" },
          { opacity: 1, transform: "translateY(0)" },
        ],
        { duration: 220, easing: "cubic-bezier(.2,.8,.2,1)" },
      );
    }
  };
  const animation = body.animate(
    [
      { opacity: 1, transform: "translateY(0) scale(1)" },
      { opacity: 0.35, transform: "translateY(-2px) scale(0.995)" },
    ],
    { duration: 180, easing: "ease-out", fill: "forwards" },
  );
  animation.onfinish = finish;
  state.activeThinkingNode = null;
  state.activeThinkingProgress = [];
}

function failThinkingMessage(errorText) {
  if (!state.activeThinkingNode) {
    addMessage("assistant", errorText, "error");
    return;
  }
  const node = state.activeThinkingNode;
  node.className = "message assistant error";
  node.querySelector(".message-meta").textContent = t("agent");
  node.querySelector(".message-body").textContent = errorText;
  state.activeThinkingNode = null;
  state.activeThinkingProgress = [];
}

function setComposerStatus(text, mode = "neutral") {
  const el = $("composer-status");
  el.textContent = text;
  el.dataset.mode = mode;
}

function setSending(next) {
  state.sending = next;
  $("run-btn").disabled = next;
  $("input-text").disabled = next;
  $("thinking-indicator").hidden = !next;
  setComposerStatus(next ? t("composer_busy") : t("composer_hint"), next ? "busy" : "neutral");
}

function summarizeConfig(cfg) {
  if (!cfg) return "";
  const meta = state.lastMeta || {};
  const lines = [
    `${t("phase")}: ${meta.phase_title || "unknown"}`,
    `${t("complete_state")}: ${meta.is_complete ? "true" : "false"}`,
    `${t("geometry")}: ${cfg.geometry?.structure || cfg.geometry?.chosen_skeleton || "unknown"}`,
    `${t("materials")}: ${(cfg.materials?.selected_materials || []).join(", ") || "missing"}`,
    `${t("particle")}: ${cfg.source?.particle || "missing"}`,
    `${t("source_type")}: ${cfg.source?.type || "missing"}`,
    `${t("physics_list")}: ${cfg.physics?.physics_list || "missing"}`,
    `${t("output")}: ${cfg.output?.format || "missing"}`,
  ];
  const asked = (meta.asked_fields_friendly || []).join(", ");
  if (asked) lines.push(`${t("pending_asks")}: ${asked}`);
  return lines.join("\n");
}

function summarizeProcess(proc) {
  if (!proc) return "";
  const rejected = Array.isArray(proc.rejected_updates) ? proc.rejected_updates : [];
  const violations = Array.isArray(proc.violations) ? proc.violations : [];
  const rules = Array.isArray(proc.applied_rules) ? proc.applied_rules : [];
  return [
    `llm_used: ${proc.llm_used ? "true" : "false"}`,
    `fallback_reason: ${proc.fallback_reason || "none"}`,
    `phase: ${proc.phase_title || proc.phase || "unknown"}`,
    `asked: ${(proc.asked_fields_friendly || []).join(", ") || "none"}`,
    "rejected_updates:",
    ...(rejected.length ? rejected.slice(0, 6).map((x) => `${x.path || "?"} :: ${x.reason_code || "unknown"}`) : ["none"]),
    "violations:",
    ...(violations.length ? violations.slice(0, 6).map((x) => `${x.path || "?"} :: ${x.code || "unknown"}`) : ["none"]),
    "applied_rules:",
    ...(rules.length ? rules.slice(0, 6).map((x) => `${x.path || "?"} <- ${x.producer || x.rule || "rule"}`) : ["none"]),
  ].join("\n");
}

function summarizeInternalTrace(trace) {
  return trace ? JSON.stringify(trace, null, 2) : "";
}

function parseConfigFromResponse() {
  try {
    return JSON.parse($("response").textContent || "{}");
  } catch (_) {
    return {};
  }
}

function isFilled(value) {
  if (value == null) return false;
  if (typeof value === "string") return value.trim().length > 0;
  if (typeof value === "number" || typeof value === "boolean") return true;
  if (Array.isArray(value)) return value.length > 0;
  if (typeof value === "object") return Object.keys(value).length > 0;
  return false;
}

function primaryMaterial(cfg) {
  const selected = cfg.materials?.selected_materials;
  if (Array.isArray(selected) && selected.length) return selected.join(", ");
  return cfg.geometry?.material || "";
}

function outputFormat(cfg) {
  return cfg.output?.format || cfg.output?.mode || "";
}

function uiWord(key) {
  const zh = {
    not_set: "未设置",
    waiting: "待补充",
    none: "暂无",
    offline: "离线",
    ready: "就绪",
    pending: "待准备",
    last_action: "最近动作",
    status: "状态",
  };
  const en = {
    not_set: "not set",
    waiting: "waiting",
    none: "none",
    offline: "offline",
    ready: "ready",
    pending: "pending",
    last_action: "Last Action",
    status: "Status",
  };
  return (state.lang === "zh" ? zh : en)[key] || key;
}

function buildConfigOverview(cfg = {}) {
  const meta = state.lastMeta || {};
  return [
    { label: t("phase_label"), value: meta.phase_title || "idle" },
    { label: t("geometry"), value: cfg.geometry?.structure || cfg.geometry?.chosen_skeleton || uiWord("not_set") },
    { label: t("materials"), value: primaryMaterial(cfg) || uiWord("not_set") },
    { label: t("particle"), value: cfg.source?.particle || uiWord("not_set") },
    { label: t("source_type"), value: cfg.source?.type || uiWord("not_set") },
    { label: t("physics_list"), value: cfg.physics?.physics_list || uiWord("not_set") },
    { label: t("output"), value: outputFormat(cfg) || uiWord("not_set") },
  ];
}

function buildCompletionBuckets(cfg = {}) {
  const items = [
    { label: t("geometry"), filled: isFilled(cfg.geometry?.structure || cfg.geometry?.chosen_skeleton) },
    { label: t("materials"), filled: isFilled(primaryMaterial(cfg)) },
    { label: t("particle"), filled: isFilled(cfg.source?.particle) },
    { label: t("source_type"), filled: isFilled(cfg.source?.type) },
    { label: t("physics_list"), filled: isFilled(cfg.physics?.physics_list) },
    { label: t("output"), filled: isFilled(outputFormat(cfg)) },
  ];
  const asked = (state.lastMeta?.asked_fields_friendly || []).map((x) => String(x).trim()).filter(Boolean);
  const filled = items.filter((item) => item.filled).map((item) => item.label);
  const missing = [...new Set([...items.filter((item) => !item.filled).map((item) => item.label), ...asked])];
  return { filled, missing };
}

function buildRuntimeOverview(runtimePayload = {}) {
  return [
    { label: uiWord("status"), value: runtimePayload.status || uiWord("offline") },
    { label: t("phase_label"), value: runtimePayload.runtime_phase || "idle" },
    { label: t("geometry"), value: runtimePayload.geometry_ready ? uiWord("ready") : uiWord("pending") },
    { label: t("source_type"), value: runtimePayload.source_ready ? uiWord("ready") : uiWord("pending") },
    { label: t("physics_list"), value: runtimePayload.physics_ready ? uiWord("ready") : uiWord("pending") },
    { label: uiWord("last_action"), value: runtimePayload.last_action || uiWord("none") },
  ];
}

function renderOverviewGrid(targetId, entries = []) {
  const root = $(targetId);
  if (!root) return;
  root.innerHTML = "";
  entries.forEach((entry) => {
    const card = document.createElement("div");
    card.className = "overview-item";
    card.innerHTML = `
      <div class="overview-label">${escapeHtml(entry.label)}</div>
      <div class="overview-value">${escapeHtml(entry.value || "-")}</div>
    `;
    root.appendChild(card);
  });
}

function renderTagList(targetId, items = [], variant = "") {
  const root = $(targetId);
  if (!root) return;
  root.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("span");
    empty.className = `tag empty ${variant}`.trim();
    empty.textContent = variant === "missing" ? uiWord("none") : uiWord("waiting");
    root.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const tag = document.createElement("span");
    tag.className = `tag ${variant}`.trim();
    tag.textContent = item;
    root.appendChild(tag);
  });
}

function renderConfigInspector(cfg = {}) {
  renderOverviewGrid("config-overview", buildConfigOverview(cfg));
  const buckets = buildCompletionBuckets(cfg);
  renderTagList("filled-items", buckets.filled, "filled");
  renderTagList("missing-items", buckets.missing, "missing");
}

function renderRuntimeInspector(runtimePayload = {}) {
  renderOverviewGrid("runtime-overview", buildRuntimeOverview(runtimePayload));
}

function summarizeGeant4State(statePayload) {
  if (!statePayload) return "";
  return [
    `status: ${statePayload.status || ""}`,
    `runtime_phase: ${statePayload.runtime_phase || ""}`,
    `connected: ${statePayload.connected}`,
    `geometry_ready: ${statePayload.geometry_ready}`,
    `source_ready: ${statePayload.source_ready}`,
    `physics_ready: ${statePayload.physics_ready}`,
    `last_action: ${statePayload.last_action || ""}`,
    `last_error: ${statePayload.last_error || ""}`,
    `available_actions: ${(statePayload.available_actions || []).join(", ")}`,
    `metadata: ${JSON.stringify(statePayload.metadata || {}, null, 2)}`,
  ].join("\n");
}

function summarizeGeant4Log(payload) {
  if (!payload) return "";
  const lines = payload.lines || payload.stdout_tail || [];
  return Array.isArray(lines) ? lines.join("\n") : JSON.stringify(payload, null, 2);
}

function summarizeGeometryCompare(compare) {
  if (!compare) return "";
  const mismatches = Array.isArray(compare.mismatches) ? compare.mismatches : [];
  const lines = [
    `compile_ok: ${compare.compile_ok === true ? "true" : "false"}`,
    `matches: ${compare.matches === true ? "true" : "false"}`,
    `spec_structure: ${compare.spec_structure || ""}`,
    `finalization_status: ${compare.finalization_status || ""}`,
  ];
  if (Array.isArray(compare.errors) && compare.errors.length) {
    lines.push(`errors: ${compare.errors.join(", ")}`);
  }
  if (Array.isArray(compare.missing_fields) && compare.missing_fields.length) {
    lines.push(`missing_fields: ${compare.missing_fields.join(", ")}`);
  }
  if (mismatches.length) {
    lines.push("mismatches:");
    mismatches.slice(0, 12).forEach((item) => {
      lines.push(`- ${item.field}: legacy=${JSON.stringify(item.legacy)} new=${JSON.stringify(item.new)}`);
    });
  } else {
    lines.push("mismatches: none");
  }
  return lines.join("\n");
}

function summarizeSourceCompare(compare) {
  if (!compare) return "";
  const mismatches = Array.isArray(compare.mismatches) ? compare.mismatches : [];
  const lines = [
    `compile_ok: ${compare.compile_ok === true ? "true" : "false"}`,
    `matches: ${compare.matches === true ? "true" : "false"}`,
    `spec_source_type: ${compare.spec_source_type || ""}`,
    `finalization_status: ${compare.finalization_status || ""}`,
  ];
  if (Array.isArray(compare.errors) && compare.errors.length) {
    lines.push(`errors: ${compare.errors.join(", ")}`);
  }
  if (Array.isArray(compare.missing_fields) && compare.missing_fields.length) {
    lines.push(`missing_fields: ${compare.missing_fields.join(", ")}`);
  }
  if (mismatches.length) {
    lines.push("mismatches:");
    mismatches.slice(0, 12).forEach((item) => {
      lines.push(`- ${item.field}: expected=${JSON.stringify(item.expected)} actual=${JSON.stringify(item.actual)}`);
    });
  } else {
    lines.push("mismatches: none");
  }
  return lines.join("\n");
}

function buildRuntimeLogSummary(payload) {
  const lines = Array.isArray(payload?.lines)
    ? payload.lines
    : Array.isArray(payload?.stdout_tail)
      ? payload.stdout_tail
      : [];
  const trimmed = lines.map((line) => String(line).trim()).filter(Boolean);
  const highlights = [];

  for (let i = trimmed.length - 1; i >= 0; i -= 1) {
    const line = trimmed[i];
    if (/Run Summary/i.test(line)) highlights.push({ kind: "ok", text: line });
    else if (/Number of events processed/i.test(line)) highlights.push({ kind: "ok", text: line });
    else if (/terminated/i.test(line)) highlights.push({ kind: "ok", text: line });
    else if (/error|fatal|exception/i.test(line)) highlights.push({ kind: "warn", text: line });
    else if (/warning/i.test(line)) highlights.push({ kind: "warn", text: line });
    if (highlights.length >= 4) break;
  }

  if (!highlights.length && trimmed.length) {
    const tail = trimmed.slice(-3);
    tail.forEach((line) => highlights.push({ kind: "plain", text: line }));
  }

  return highlights.reverse();
}

function renderRuntimeLogSummary(payload = {}) {
  const root = $("runtime-log-summary");
  if (!root) return;
  const items = buildRuntimeLogSummary(payload);
  root.innerHTML = "";
  if (!items.length) {
    const empty = document.createElement("div");
    empty.className = "log-line empty";
    empty.textContent = uiWord("waiting");
    root.appendChild(empty);
    return;
  }
  items.forEach((item) => {
    const row = document.createElement("div");
    row.className = `log-line ${item.kind}`.trim();
    row.textContent = item.text;
    root.appendChild(row);
  });
}

function formatRuntimeValue(value) {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number" && Number.isFinite(value)) {
    if (Math.abs(value) > 0 && Math.abs(value) < 0.001) return value.toExponential(3);
    return Number.isInteger(value) ? String(value) : String(Number(value.toPrecision(6)));
  }
  if (typeof value === "boolean") return value ? "true" : "false";
  return String(value);
}

function buildRuntimeResultRows(report = {}) {
  const config = report.configuration || {};
  const metrics = report.key_metrics || {};
  const completed = report.events_completed;
  const requested = report.events_requested;
  const eventText = completed === null || completed === undefined
    ? formatRuntimeValue(requested)
    : `${formatRuntimeValue(completed)} / ${formatRuntimeValue(requested)}`;
  return [
    ["ok", report.ok],
    ["events", eventText],
    ["completion", report.completion_fraction],
    ["geometry", config.geometry_structure],
    ["material", config.material],
    ["particle", config.particle],
    ["physics", config.physics_list],
    ["target edep MeV", metrics.target_edep_total_mev],
    ["target hits", metrics.target_hit_events],
    ["detector crossings", metrics.detector_crossing_count],
    ["plane crossings", metrics.plane_crossing_count],
    ["run summary", report.run_summary_path],
  ];
}

function renderRuntimeResultSummary(report = null) {
  const root = $("runtime-result-summary");
  if (!root) return;
  root.innerHTML = "";
  if (!report) {
    const empty = document.createElement("div");
    empty.className = "result-empty";
    empty.textContent = t("no_runtime_result");
    root.appendChild(empty);
    return;
  }
  buildRuntimeResultRows(report).forEach(([label, value]) => {
    const row = document.createElement("div");
    row.className = "result-row";
    const key = document.createElement("span");
    key.className = "result-label";
    key.textContent = label;
    const val = document.createElement("strong");
    val.className = "result-value";
    val.textContent = formatRuntimeValue(value);
    row.appendChild(key);
    row.appendChild(val);
    root.appendChild(row);
  });
}

function runtimeResultMessage(report) {
  if (!report) return "";
  const metrics = report.key_metrics || {};
  return [
    `events_completed=${formatRuntimeValue(report.events_completed)}`,
    `completion_fraction=${formatRuntimeValue(report.completion_fraction)}`,
    `target_edep_total_mev=${formatRuntimeValue(metrics.target_edep_total_mev)}`,
    `detector_crossing_count=${formatRuntimeValue(metrics.detector_crossing_count)}`,
    `plane_crossing_count=${formatRuntimeValue(metrics.plane_crossing_count)}`,
  ].join("\n");
}

function isRuntimeResultQuestion(text) {
  const raw = String(text || "").trim().toLowerCase();
  if (!raw) return false;
  const zhHit = /(刚才|上次|最近|当前).*(结果|模拟|运行|计分|得分)|结果怎么样|模拟怎么样|运行怎么样|剂量多少|沉积能量|hit|crossing/i.test(raw);
  const enHit = /\b(last|latest|previous|current)\b.*\b(result|simulation|run|scoring|score|edep|hit|crossing)\b|\bwhat happened\b.*\b(run|simulation)\b|\bhow did\b.*\b(run|simulation)\b/.test(raw);
  return zhHit || enHit;
}

async function answerRuntimeResultQuestion() {
  const res = await fetch("/api/geant4/summary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      lang: state.lang,
      llm_result_summary: $("llm-question")?.checked === true,
      ollama_config_path: state.ollamaConfigPath,
    }),
  });
  const data = await res.json();
  if (res.ok && data.runtime_smoke_report) {
    state.lastRuntimeSmokeReport = data.runtime_smoke_report;
    renderRuntimeResultSummary(state.lastRuntimeSmokeReport);
  }
  if (data.runtime_result_explanation?.message) {
    return data.runtime_result_explanation.message;
  }
  if (data.errors?.includes("no_result_summary_available")) {
    return state.lang === "zh"
      ? "当前还没有可解释的 Geant4 运行结果。请先显式点击运行按钮，再询问结果。"
      : "No Geant4 runtime result is available yet. Please explicitly run the simulation first, then ask about the result.";
  }
  return data.message || (state.lang === "zh" ? "暂时无法读取运行结果。" : "I could not read the runtime result yet.");
}

function updateDebugPanelVisibility() {
  const blocks = [
    ["debug-terminal-block", $("geant4-log")?.textContent],
    ["debug-process-block", $("process-log")?.textContent],
    ["debug-trace-block", $("internal-trace")?.textContent],
    ["debug-geometry-block", $("geometry-compare")?.textContent],
    ["debug-source-block", $("source-compare")?.textContent],
    ["debug-config-block", $("response")?.textContent],
    ["debug-runtime-block", $("geant4-state")?.textContent],
  ];
  let visibleCount = 0;
  blocks.forEach(([id, text]) => {
    const node = $(id);
    if (!node) return;
    const hasText = String(text || "").trim().length > 0;
    node.hidden = !hasText;
    if (hasText) visibleCount += 1;
  });
  const details = $("debug-details");
  const card = details?.closest(".debug-card");
  if (card) card.hidden = visibleCount === 0;
  if (details && visibleCount === 0) details.open = false;
}

function currentConfigPatch() {
  const cfg = parseConfigFromResponse();
  return {
    geometry: cfg.geometry || {},
    source: cfg.source || {},
    physics_list: cfg.physics || {},
    output: cfg.output || {},
  };
}

function renderTopbar() {
  const runtimePayload = state.geant4State || {};
  const meta = state.lastMeta || {};
  $("header-phase-chip").textContent = meta.phase_title || runtimePayload.runtime_phase || "idle";
  $("header-complete-chip").textContent = meta.is_complete ? t("complete") : t("incomplete");
  $("runtime-status-inline").textContent = runtimePayload.connected ? runtimePayload.status || "connected" : t("offline");
  $("sidebar-runtime-phase").textContent = runtimePayload.runtime_phase || "idle";
  $("sidebar-model-name").textContent = state.ollamaModel ? `${state.llmProvider || "llm"} / ${state.ollamaModel}` : "-";
  $("sidebar-session-id").textContent = state.sessionId || t("new_session");
}

function renderRuntimeNotice() {
  const box = $("runtime-notice");
  const p = state.modelPreflight;
  if (!p) {
    box.className = "runtime-notice";
    box.textContent = "";
    return;
  }
  if (p.ready) {
    box.className = "runtime-notice ready";
    box.textContent = t("runtime_ready");
    return;
  }
  const missing = [];
  const warnings = [];
  for (const key of ["structure", "ner"]) {
    const item = p[key] || {};
    const m = Array.isArray(item.missing_files) ? item.missing_files : [];
    const w = Array.isArray(item.warnings) ? item.warnings : [];
    if (m.length) missing.push(`${key}: ${m.join(", ")}`);
    if (w.length) warnings.push(`${key}: ${w.join(", ")}`);
  }
  box.className = "runtime-notice warn";
  box.textContent = [t("runtime_incomplete"), missing.join(" | "), warnings.join(" | ")].filter(Boolean).join(" ");
}

function buildAssistantMessage(data) {
  if (data.assistant_message) return data.assistant_message;
  return "The configuration state has been updated.";
}

function applyI18n() {
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.dataset.i18n;
    el.textContent = t(key);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const key = el.dataset.i18nPlaceholder;
    el.setAttribute("placeholder", t(key));
  });
  ensureEmptyState();
  renderTopbar();
  renderRuntimeNotice();
  renderConfigInspector(parseConfigFromResponse());
  renderRuntimeInspector(state.geant4State || {});
  renderRuntimeResultSummary(state.lastRuntimeSmokeReport);
  updateDebugPanelVisibility();
  setComposerStatus(state.sending ? t("composer_busy") : t("composer_hint"), state.sending ? "busy" : "neutral");
}

async function refreshGeant4State() {
  const res = await fetch("/api/geant4/state");
  const data = await res.json();
  state.geant4State = data;
  $("geant4-state").textContent = summarizeGeant4State(data);
  renderRuntimeInspector(data);
  updateDebugPanelVisibility();
  renderTopbar();
}

async function refreshGeant4Log() {
  const res = await fetch("/api/geant4/log", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const data = await res.json();
  const payload = data.payload || {};
  $("geant4-log").textContent = summarizeGeant4Log(payload);
  renderRuntimeLogSummary(payload);
  updateDebugPanelVisibility();
}

async function refreshGeant4Summary() {
  const res = await fetch("/api/geant4/summary", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      lang: state.lang,
      llm_result_summary: $("llm-question")?.checked === true,
      ollama_config_path: state.ollamaConfigPath,
    }),
  });
  const data = await res.json();
  if (res.ok && data.runtime_smoke_report) {
    state.lastRuntimeSmokeReport = data.runtime_smoke_report;
  } else if (data.errors?.includes("no_result_summary_available")) {
    state.lastRuntimeSmokeReport = null;
  }
  renderRuntimeResultSummary(state.lastRuntimeSmokeReport);
}

async function syncGeant4Config() {
  const res = await fetch("/api/geant4/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patch: currentConfigPatch() }),
  });
  const data = await res.json();
  await refreshGeant4State();
  if (data.message) addMessage("assistant", `${t("geant4_prefix")}: ${data.message}`, "system");
}

async function initializeGeant4() {
  const res = await fetch("/api/geant4/initialize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  const data = await res.json();
  await refreshGeant4State();
  if (data.message) addMessage("assistant", `${t("geant4_prefix")}: ${data.message}`, "system");
}

async function openGeant4Viewer() {
  const res = await fetch("/api/geant4/viewer/open", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ patch: currentConfigPatch(), events: 12 }),
  });
  const data = await res.json();
  await refreshGeant4State();
  state.lastRuntimeSmokeReport = data.runtime_smoke_report || state.lastRuntimeSmokeReport;
  renderRuntimeResultSummary(state.lastRuntimeSmokeReport);
  $("geant4-log").textContent = summarizeGeant4Log({
    lines: [...(data.payload?.stdout_tail || []), ...(data.payload?.stderr_tail || [])],
  });
  renderRuntimeLogSummary({
    lines: [...(data.payload?.stdout_tail || []), ...(data.payload?.stderr_tail || [])],
  });
  if (data.message) {
    const resultText = data.runtime_smoke_report ? `\n${runtimeResultMessage(data.runtime_smoke_report)}` : "";
    addMessage("assistant", `${t("geant4_prefix")}: ${data.message}${resultText}`, "system");
  }
}

async function runGeant4(events) {
  const res = await fetch("/api/geant4/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      events,
      lang: state.lang,
      llm_result_summary: $("llm-question")?.checked === true,
      ollama_config_path: state.ollamaConfigPath,
    }),
  });
  const data = await res.json();
  await refreshGeant4State();
  $("geant4-log").textContent = summarizeGeant4Log({
    lines: [...(data.payload?.stdout_tail || []), ...(data.payload?.stderr_tail || [])],
  });
  renderRuntimeLogSummary({
    lines: [...(data.payload?.stdout_tail || []), ...(data.payload?.stderr_tail || [])],
  });
  if (data.message) {
    const explanation = data.runtime_result_explanation?.message || (
      data.runtime_smoke_report ? runtimeResultMessage(data.runtime_smoke_report) : ""
    );
    const resultText = explanation ? `\n${explanation}` : "";
    addMessage("assistant", `${t("geant4_prefix")}: ${data.message}${resultText}`, "system");
  }
}

async function loadRuntimeConfigs() {
  const sel = $("model-config-select");
  const res = await fetch("/api/runtime");
  const data = await res.json();
  const available = Array.isArray(data.available) ? data.available : [];

  sel.innerHTML = "";
  available.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.path;
    opt.textContent = `${item.provider || "ollama"} / ${item.model}`;
    sel.appendChild(opt);
  });

  const preferred = state.ollamaConfigPath || data.current_path || (available[0] && available[0].path) || "";
  if (preferred) sel.value = preferred;
  state.ollamaModel = data.current_model || "";
  state.llmProvider = data.current_provider || "";
  state.ollamaConfigPath = sel.value || "";
  state.modelPreflight = data.model_preflight || null;
  renderRuntimeNotice();
  renderTopbar();
}

async function applyRuntimeConfig(path) {
  const res = await fetch("/api/runtime", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ollama_config_path: path }),
  });
  const data = await res.json();
  if (!res.ok || !data.ok) throw new Error(data.message || "failed to set runtime config");
  state.ollamaConfigPath = data.current_path || path;
  state.ollamaModel = data.current_model || "";
  state.llmProvider = data.current_provider || "";
  state.modelPreflight = data.model_preflight || null;
  localStorage.setItem("g4_ollama_config_path", state.ollamaConfigPath);
  renderRuntimeNotice();
  renderTopbar();
}

async function pollStepJob(jobId) {
  while (true) {
    await new Promise((resolve) => setTimeout(resolve, 380));
    const res = await fetch("/api/step_status", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ job_id: jobId }),
    });
    if (!res.ok) throw new Error(`status polling failed: ${res.status}`);
    const payload = await res.json();
    renderThinkingProgress(payload.progress || []);
    if (payload.status === "completed") return payload.result || {};
    if (payload.status === "failed") throw new Error(payload.error || "async job failed");
  }
}

async function sendStep() {
  if (state.sending) return;
  const text = $("input-text").value.trim();
  if (!text) return;

  addMessage("user", text);
  $("input-text").value = "";
  setSending(true);
  createThinkingMessage();

  try {
    if (isRuntimeResultQuestion(text)) {
      const answer = await answerRuntimeResultQuestion();
      finalizeThinkingMessage(answer);
      await refreshGeant4State();
      return;
    }

    const payload = {
      session_id: state.sessionId || null,
      text,
      min_confidence: Number($("min-conf").value || 0.6),
      strict_mode: true,
      autofix: $("autofix").checked,
      llm_router: $("llm-router").checked,
      llm_question: $("llm-question").checked,
      lang: state.lang,
      geometry_pipeline: state.geometryPipeline,
      source_pipeline: state.sourcePipeline,
    };

    const kickoff = await fetch("/api/step_async", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!kickoff.ok) throw new Error(`request failed: ${kickoff.status}`);
    const kickoffData = await kickoff.json();
    if (!kickoffData.job_id) throw new Error("async job was not created");

    const data = await pollStepJob(kickoffData.job_id);
    if (data.session_id && data.session_id !== state.sessionId) {
      state.sessionId = data.session_id;
      localStorage.setItem("g4_session_id", state.sessionId);
    }

    finalizeThinkingMessage(buildAssistantMessage(data));

    state.lastMeta = {
      phase: data.phase,
      phase_title: data.phase_title,
      asked_fields_friendly: data.asked_fields_friendly || [],
      is_complete: !!data.is_complete,
    };
    state.lastProcess = {
      llm_used: !!data.llm_used,
      fallback_reason: data.fallback_reason || "",
      temperatures: data.temperatures || {},
      phase: data.phase,
      phase_title: data.phase_title,
      asked_fields_friendly: data.asked_fields_friendly || [],
      rejected_updates: data.rejected_updates || [],
      violations: data.violations || [],
      applied_rules: data.applied_rules || [],
      internal_trace: data.internal_trace || null,
      geometry_compare: data.geometry_compare || null,
      source_compare: data.source_compare || null,
    };

    $("summary").textContent = summarizeConfig(data.config);
    $("response").textContent = JSON.stringify(data.config || {}, null, 2);
    renderConfigInspector(data.config || {});
    $("process-log").textContent = summarizeProcess(state.lastProcess);
    $("internal-trace").textContent = summarizeInternalTrace(state.lastProcess.internal_trace);
    $("geometry-compare").textContent = summarizeGeometryCompare(state.lastProcess.geometry_compare);
    $("source-compare").textContent = summarizeSourceCompare(state.lastProcess.source_compare);
    updateDebugPanelVisibility();
    renderTopbar();
    await refreshGeant4State();
  } catch (error) {
    failThinkingMessage(`${t("request_failed")}\n${error.message}`);
  } finally {
    setSending(false);
  }
}

async function resetSession() {
  if (state.sessionId) {
    await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: state.sessionId }),
    });
  }
  state.sessionId = "";
  state.lastMeta = null;
  state.lastProcess = null;
  state.lastRuntimeSmokeReport = null;
  state.activeThinkingNode = null;
  state.activeThinkingProgress = [];
  localStorage.removeItem("g4_session_id");
  $("chat").innerHTML = "";
  $("summary").textContent = "";
  $("response").textContent = "";
  renderConfigInspector({});
  $("process-log").textContent = "";
  $("internal-trace").textContent = "";
  $("geometry-compare").textContent = "";
  $("source-compare").textContent = "";
  $("geant4-state").textContent = "";
  $("geant4-log").textContent = "";
  renderRuntimeLogSummary({});
  renderRuntimeResultSummary(null);
  renderRuntimeInspector({});
  updateDebugPanelVisibility();
  renderTopbar();
  ensureEmptyState();
}

function bindComposerHotkeys() {
  $("input-text").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendStep();
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  ensureEmptyState();
  $("lang-select").value = state.lang;
  $("geometry-pipeline-select").value = state.geometryPipeline;
  $("source-pipeline-select").value = state.sourcePipeline;
  applyI18n();
  bindComposerHotkeys();

  loadRuntimeConfigs().catch((err) => addMessage("assistant", `Runtime config load failed: ${err.message}`, "error"));
  refreshGeant4State().catch(() => {});
  refreshGeant4Log().catch(() => {});
  refreshGeant4Summary().catch(() => {});

  $("run-btn").addEventListener("click", sendStep);
  $("reset-btn").addEventListener("click", resetSession);
  $("g4-sync-btn").addEventListener("click", syncGeant4Config);
  $("g4-viewer-btn").addEventListener("click", openGeant4Viewer);
  $("g4-init-btn").addEventListener("click", initializeGeant4);
  $("g4-run-btn").addEventListener("click", () => runGeant4(1));
  $("g4-run10-btn").addEventListener("click", () => runGeant4(10));
  $("g4-refresh-btn").addEventListener("click", async () => {
    await refreshGeant4State();
    await refreshGeant4Log();
    await refreshGeant4Summary();
  });
  $("lang-select").addEventListener("change", (event) => {
    state.lang = event.target.value || "zh";
    localStorage.setItem("g4_lang", state.lang);
    applyI18n();
    if ($("summary").textContent) {
      try {
        const cfg = JSON.parse($("response").textContent || "{}");
        $("summary").textContent = summarizeConfig(cfg);
      } catch (_) {}
    }
  });
  $("geometry-pipeline-select").addEventListener("change", (event) => {
    state.geometryPipeline = event.target.value || "legacy";
    localStorage.setItem("g4_geometry_pipeline", state.geometryPipeline);
  });
  $("source-pipeline-select").addEventListener("change", (event) => {
    state.sourcePipeline = event.target.value || "legacy";
    localStorage.setItem("g4_source_pipeline", state.sourcePipeline);
  });
  $("model-config-select").addEventListener("change", async (event) => {
    const nextPath = event.target.value || "";
    if (!nextPath) return;
    try {
      await applyRuntimeConfig(nextPath);
      addMessage("assistant", `${t("model_switched")}: ${state.llmProvider || "provider"} / ${state.ollamaModel}.`, "system");
    } catch (error) {
      addMessage("assistant", `Model switch failed.\n${error.message}`, "error");
    }
  });
});
