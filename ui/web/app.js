const $ = (id) => document.getElementById(id);

const state = {
  sessionId: localStorage.getItem("g4_session_id") || "",
  lang: localStorage.getItem("g4_lang") || "zh",
  lastMeta: null,
};

const i18n = {
  zh: {
    badge: "本地 Web 控制台",
    subtitle: "通过多轮对话补全需求，生成可用配置。",
    input_title: "输入",
    input_label: "请描述你的实验设想",
    input_placeholder: "例如：我想用gamma打一个铜立方体，看看能量沉积...",
    lang_label: "Language",
    min_conf: "结构置信度阈值",
    autofix: "自动修正",
    llm_routing: "LLM 参与路由",
    llm_question: "LLM 生成追问",
    send: "发送",
    reset: "重置对话",
    dialogue: "对话",
    output: "输出",
    summary: "摘要",
    config_json: "配置 JSON",
    footer: "本地运行，不依赖外部前端服务。",
    summary_lines: {
      geometry_structure: "几何结构",
      geometry_feasible: "几何可行",
      materials: "材料",
      source_particle: "粒子",
      source_type: "源类型",
      physics_list: "物理过程",
      output_format: "输出格式",
      phase: "当前阶段",
      asked: "本轮追问",
      complete: "完成状态",
      missing: "未指定",
      unknown: "未确定",
      undefined: "未判定",
    },
  },
  en: {
    badge: "Local Web Console",
    subtitle: "Multi-turn dialogue to complete requirements and generate configs.",
    input_title: "Input",
    input_label: "Describe your experiment",
    input_placeholder: "e.g., shoot gamma at a copper cube and observe energy deposition...",
    lang_label: "Language",
    min_conf: "Structure confidence threshold",
    autofix: "Auto-fix",
    llm_routing: "LLM routing",
    llm_question: "LLM question",
    send: "Send",
    reset: "Reset Session",
    dialogue: "Dialogue",
    output: "Output",
    summary: "Summary",
    config_json: "Config JSON",
    footer: "Local-only: no external UI dependencies.",
    summary_lines: {
      geometry_structure: "Geometry",
      geometry_feasible: "Feasible",
      materials: "Materials",
      source_particle: "Particle",
      source_type: "Source type",
      physics_list: "Physics list",
      output_format: "Output format",
      phase: "Phase",
      asked: "Asked",
      complete: "Complete",
      missing: "missing",
      unknown: "unknown",
      undefined: "undefined",
    },
  },
};

function addMessage(role, text) {
  const chat = $("chat");
  const bubble = document.createElement("div");
  bubble.className = `msg ${role}`;
  bubble.textContent = text;
  chat.appendChild(bubble);
  chat.scrollTop = chat.scrollHeight;
}

function summarizeConfig(cfg) {
  if (!cfg) return "No config yet.";
  const t = i18n[state.lang].summary_lines;
  const meta = state.lastMeta || {};
  const geomLabel = cfg.geometry?.structure ?? cfg.geometry?.chosen_skeleton ?? t.unknown;
  return [
    `${t.phase}: ${meta.phase_title ?? t.unknown}`,
    `${t.complete}: ${meta.is_complete ? "true" : "false"}`,
    `${t.asked}: ${(meta.asked_fields_friendly || []).join(", ") || t.missing}`,
    `${t.geometry_structure}: ${geomLabel}`,
    `${t.geometry_feasible}: ${cfg.geometry?.feasible ?? t.undefined}`,
    `${t.materials}: ${(cfg.materials?.selected_materials || []).join(", ") || t.missing}`,
    `${t.source_particle}: ${cfg.source?.particle ?? t.missing}`,
    `${t.source_type}: ${cfg.source?.type ?? t.missing}`,
    `${t.physics_list}: ${cfg.physics?.physics_list ?? t.missing}`,
    `${t.output_format}: ${cfg.output?.format ?? t.missing}`,
  ].join("\n");
}

async function sendStep() {
  const text = $("input-text").value.trim();
  if (!text) return;
  addMessage("user", text);
  $("input-text").value = "";

  const payload = {
    session_id: state.sessionId || null,
    text,
    min_confidence: Number($("min-conf").value || 0.6),
    autofix: $("autofix").checked,
    llm_router: $("llm-router").checked,
    llm_question: $("llm-question") ? $("llm-question").checked : true,
    lang: state.lang,
  };

  const res = await fetch("/api/step", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  const data = await res.json();
  if (data.session_id && data.session_id !== state.sessionId) {
    state.sessionId = data.session_id;
    localStorage.setItem("g4_session_id", state.sessionId);
  }

  if (data.assistant_message) {
    addMessage("assistant", data.assistant_message);
  }

  state.lastMeta = {
    phase: data.phase,
    phase_title: data.phase_title,
    asked_fields_friendly: data.asked_fields_friendly || [],
    is_complete: !!data.is_complete,
  };
  $("summary").textContent = summarizeConfig(data.config);
  $("response").textContent = JSON.stringify(data.config || {}, null, 2);
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
  localStorage.removeItem("g4_session_id");
  $("chat").innerHTML = "";
  $("summary").textContent = "";
  $("response").textContent = "";
}

document.addEventListener("DOMContentLoaded", () => {
  const select = $("lang-select");
  select.value = state.lang;
  select.addEventListener("change", (e) => {
    state.lang = e.target.value || "zh";
    localStorage.setItem("g4_lang", state.lang);
    applyI18n();
    if ($("summary").textContent) {
      try {
        const cfg = JSON.parse($("response").textContent || "{}");
        $("summary").textContent = summarizeConfig(cfg);
      } catch (_) {}
    }
  });

  applyI18n();
  $("run-btn").addEventListener("click", sendStep);
  $("reset-btn").addEventListener("click", resetSession);
});

function applyI18n() {
  const dict = i18n[state.lang];
  document.querySelectorAll("[data-i18n]").forEach((el) => {
    const key = el.getAttribute("data-i18n");
    if (dict[key]) el.textContent = dict[key];
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => {
    const key = el.getAttribute("data-i18n-placeholder");
    if (dict[key]) el.setAttribute("placeholder", dict[key]);
  });
}
