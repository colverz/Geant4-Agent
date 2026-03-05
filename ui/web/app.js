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
};

const i18n = {
  zh: {
    badge: "本地 Web 控制台",
    subtitle: "通过多轮对话补全需求，生成可用配置。",
    input_title: "输入",
    input_label: "请描述你的实验设想",
    input_placeholder: "例如：我想用gamma打一个铜立方体，看看能量沉积...",
    model_label: "LLM 配置",
    provider_label: "供应商",
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
    process_title: "过程面板",
    internal_trace_title: "内部迭代链（程序↔LLM）",
    footer: "本地运行，不依赖外部前端服务。",
    runtime_ready: "模型检查通过：structure/ner 目录完整。",
    runtime_not_ready: "模型检查未通过：",
    runtime_missing: "缺失项",
    runtime_warning: "告警",
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
      none: "无",
    },
  },
  en: {
    badge: "Local Web Console",
    subtitle: "Multi-turn dialogue to complete requirements and generate configs.",
    input_title: "Input",
    input_label: "Describe your experiment",
    input_placeholder: "e.g., shoot gamma at a copper cube and observe energy deposition...",
    model_label: "LLM Config",
    provider_label: "Provider",
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
    process_title: "Process Panel",
    internal_trace_title: "Internal Iteration Chain (Program↔LLM)",
    footer: "Local-only: no external UI dependencies.",
    runtime_ready: "Model preflight passed: structure/ner assets are available.",
    runtime_not_ready: "Model preflight failed:",
    runtime_missing: "missing",
    runtime_warning: "warning",
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
      none: "none",
    },
  },
};

function renderRuntimeNotice() {
  const box = $("runtime-notice");
  if (!box) return;
  const dict = i18n[state.lang];
  const p = state.modelPreflight;
  if (!p) {
    box.className = "runtime-notice";
    box.textContent = "";
    return;
  }
  if (p.ready) {
    box.className = "runtime-notice ready";
    box.textContent = dict.runtime_ready;
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
  const parts = [dict.runtime_not_ready];
  if (missing.length) parts.push(`${dict.runtime_missing}: ${missing.join(" | ")}`);
  if (warnings.length) parts.push(`${dict.runtime_warning}: ${warnings.join(" | ")}`);
  box.className = "runtime-notice warn";
  box.textContent = parts.join(" ");
}

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
    `LLM: ${(state.llmProvider ? state.llmProvider + " / " : "") + (state.ollamaModel || t.unknown)}`,
    `${t.geometry_structure}: ${geomLabel}`,
    `${t.geometry_feasible}: ${cfg.geometry?.feasible ?? t.undefined}`,
    `${t.materials}: ${(cfg.materials?.selected_materials || []).join(", ") || t.missing}`,
    `${t.source_particle}: ${cfg.source?.particle ?? t.missing}`,
    `${t.source_type}: ${cfg.source?.type ?? t.missing}`,
    `${t.physics_list}: ${cfg.physics?.physics_list ?? t.missing}`,
    `${t.output_format}: ${cfg.output?.format ?? t.missing}`,
  ].join("\n");
}

function summarizeProcess(proc) {
  const t = i18n[state.lang].summary_lines;
  if (!proc) return "";
  const rejected = Array.isArray(proc.rejected_updates) ? proc.rejected_updates : [];
  const violations = Array.isArray(proc.violations) ? proc.violations : [];
  const rules = Array.isArray(proc.applied_rules) ? proc.applied_rules : [];
  const asked = Array.isArray(proc.asked_fields_friendly) ? proc.asked_fields_friendly : [];
  const topRejected = rejected.slice(0, 6).map((x) => `${x.path || "?"} :: ${x.reason_code || "unknown"}`);
  const topViolations = violations.slice(0, 6).map((x) => `${x.path || "?"} :: ${x.code || "unknown"}`);
  const topRules = rules.slice(0, 6).map((x) => `${x.path || "?"} <- ${x.producer || x.rule || "rule"}`);
  return [
    `llm_used: ${proc.llm_used ? "true" : "false"}`,
    `fallback_reason: ${proc.fallback_reason || t.none}`,
    `internal_temp: ${proc.temperatures?.internal ?? t.none}`,
    `user_temp: ${proc.temperatures?.user ?? t.none}`,
    `phase: ${proc.phase_title || proc.phase || t.unknown}`,
    `asked: ${asked.join(", ") || t.none}`,
    `rejected_updates:`,
    ...(topRejected.length ? topRejected : [t.none]),
    `violations:`,
    ...(topViolations.length ? topViolations : [t.none]),
    `applied_rules:`,
    ...(topRules.length ? topRules : [t.none]),
  ].join("\n");
}

function summarizeInternalTrace(trace) {
  if (!trace) return "";
  return JSON.stringify(trace, null, 2);
}

async function loadRuntimeConfigs() {
  const sel = $("model-config-select");
  if (!sel) return;
  const res = await fetch("/api/runtime");
  const data = await res.json();
  const available = Array.isArray(data.available) ? data.available : [];

  sel.innerHTML = "";
  available.forEach((item) => {
    const opt = document.createElement("option");
    opt.value = item.path;
    const provider = item.provider || "ollama";
    opt.textContent = `${provider} / ${item.model} (${item.path.split("/").pop()})`;
    sel.appendChild(opt);
  });

  const preferred = state.ollamaConfigPath || data.current_path || (available[0] && available[0].path) || "";
  if (preferred) {
    sel.value = preferred;
  }
  state.ollamaModel = data.current_model || "";
  state.llmProvider = data.current_provider || "";
  state.ollamaConfigPath = sel.value || "";
  state.modelPreflight = data.model_preflight || null;
  renderRuntimeNotice();
  if (state.ollamaConfigPath) {
    localStorage.setItem("g4_ollama_config_path", state.ollamaConfigPath);
  }
}

async function applyRuntimeConfig(path) {
  const res = await fetch("/api/runtime", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ollama_config_path: path }),
  });
  const data = await res.json();
  if (!res.ok || !data.ok) {
    throw new Error(data.message || "failed to set runtime config");
  }
  state.ollamaConfigPath = data.current_path || path;
  state.ollamaModel = data.current_model || "";
  state.llmProvider = data.current_provider || "";
  state.modelPreflight = data.model_preflight || null;
  localStorage.setItem("g4_ollama_config_path", state.ollamaConfigPath);
  renderRuntimeNotice();
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
    strict_mode: true,
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
  };
  $("summary").textContent = summarizeConfig(data.config);
  $("response").textContent = JSON.stringify(data.config || {}, null, 2);
  $("process-log").textContent = summarizeProcess(state.lastProcess);
  $("internal-trace").textContent = summarizeInternalTrace(state.lastProcess.internal_trace);
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
  $("process-log").textContent = "";
  $("internal-trace").textContent = "";
  state.lastProcess = null;
}

document.addEventListener("DOMContentLoaded", () => {
  const modelSelect = $("model-config-select");
  if (modelSelect) {
    loadRuntimeConfigs()
      .then(() => {
        if ($("summary").textContent) {
          try {
            const cfg = JSON.parse($("response").textContent || "{}");
            $("summary").textContent = summarizeConfig(cfg);
          } catch (_) {}
        }
      })
      .catch((err) => {
        addMessage("assistant", `runtime config load failed: ${err.message}`);
      });
    modelSelect.addEventListener("change", async (e) => {
      const nextPath = e.target.value || "";
      if (!nextPath) return;
      try {
        await applyRuntimeConfig(nextPath);
        addMessage("assistant", `LLM config switched: ${state.llmProvider || "provider"} / ${state.ollamaModel}`);
        if ($("summary").textContent) {
          try {
            const cfg = JSON.parse($("response").textContent || "{}");
            $("summary").textContent = summarizeConfig(cfg);
          } catch (_) {}
        }
      } catch (err) {
        addMessage("assistant", `runtime config update failed: ${err.message}`);
      }
    });
  }

  const select = $("lang-select");
  select.value = state.lang;
  select.addEventListener("change", (e) => {
    state.lang = e.target.value || "zh";
    localStorage.setItem("g4_lang", state.lang);
    applyI18n();
    renderRuntimeNotice();
    if ($("summary").textContent) {
      try {
        const cfg = JSON.parse($("response").textContent || "{}");
        $("summary").textContent = summarizeConfig(cfg);
      } catch (_) {}
    }
  });

  applyI18n();
  renderRuntimeNotice();
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
