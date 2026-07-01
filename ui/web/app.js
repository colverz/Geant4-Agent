// Geant4 Agent v3 - minimal reliable UI
const $ = (id) => document.getElementById(id);
const state = {
  sessionId: localStorage.getItem("g4_session_id") === "default" ? "" : (localStorage.getItem("g4_session_id") || ""),
  lang: localStorage.getItem("g4_lang") || "zh",
  sending: false,
  lastPendingAction: null,
  llmConfigPath: "",
};

function esc(s) { return String(s||"").replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }
function t(key) {
  const map = { sendFailed: state.lang==='zh'?'请求失败':'Request failed', welcome: state.lang==='zh'?'描述你的物理模拟目标':'Describe your physics simulation goal' };
  return map[key] || key;
}

// API
async function postJson(path, payload, timeoutMs=45000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const r = await fetch(path, { method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(payload), signal: ctrl.signal });
    const d = await r.json().catch(()=>({}));
    if (!r.ok) throw new Error(d.message||d.error||`${r.status}`);
    return d;
  } finally { clearTimeout(timer); }
}

// Render
function clearWelcome() {
  const w = document.querySelector(".welcome");
  if (w) w.remove();
}

function bindQuickActions(root=document) {
  root.querySelectorAll(".quick-btn").forEach((button) => {
    if (button.dataset.bound === "1") return;
    button.dataset.bound = "1";
    button.addEventListener("click", () => sendPreset(button.getAttribute("data-text") || button.textContent || ""));
  });
}

function addMsg(role, text, cls) {
  clearWelcome();
  const el = document.createElement("div");
  el.className = `msg ${role}${cls ? ' '+cls : ''}`;
  el.innerHTML = `<div class="msg-body">${text.replace(/\n/g,'<br>')}</div>`;
  const tl = $("timeline");
  if (tl) { tl.appendChild(el); el.scrollIntoView({block:'end',behavior:'smooth'}); }
}

function addAgentMsg(data) {
  const msg = data.display_message || data.answer?.display_message || data.answer?.message || "done.";
  const parts = Array.isArray(data.answer_parts) ? data.answer_parts : (Array.isArray(data.dialogue?.answer_parts) ? data.dialogue.answer_parts : []);
  if (!parts.length) {
    addMsg("agent", esc(msg));
    return;
  }
  clearWelcome();
  const el = document.createElement("div");
  el.className = "msg agent";
  const body = document.createElement("div");
  body.className = "msg-body";
  body.appendChild(renderAnswerPart({kind: "summary", text: msg}));
  parts.filter(part => part && part.kind !== "summary" && part.kind !== "dialogue_act").forEach(part => {
    body.appendChild(renderAnswerPart(part));
  });
  el.appendChild(body);
  const tl = $("timeline");
  if (tl) { tl.appendChild(el); el.scrollIntoView({block:'end',behavior:'smooth'}); }
}

function renderAnswerPart(part) {
  const section = document.createElement("section");
  const kindClass = String(part.kind || "part").replace(/[^a-z0-9_-]/gi, "") || "part";
  section.className = `answer-part ${kindClass}`;
  if (part.kind === "summary") {
    section.innerHTML = `<p>${esc(part.text || "")}</p>`;
    return section;
  }
  const title = document.createElement("div");
  title.className = "answer-part-title";
  title.textContent = part.title || (part.kind === "evidence" ? "Evidence" : part.kind === "next_step" ? "Next" : "Context");
  section.appendChild(title);
  const items = Array.isArray(part.items) ? part.items : [];
  if (items.length) {
    const list = document.createElement("div");
    list.className = "answer-part-list";
    items.slice(0, 4).forEach(item => {
      const row = document.createElement("span");
      row.className = "answer-part-item";
      if (part.kind === "evidence") row.textContent = [item.source, item.status].filter(Boolean).join(" / ");
      else row.textContent = item.text || item.prefill || item.source || "";
      list.appendChild(row);
    });
    section.appendChild(list);
  } else if (part.text) {
    const p = document.createElement("p");
    p.textContent = part.text;
    section.appendChild(p);
  }
  return section;
}

function ensureSessionId() {
  if (state.sessionId) return state.sessionId;
  state.sessionId = (globalThis.crypto?.randomUUID?.() || `ui-${Date.now()}-${Math.random().toString(16).slice(2)}`);
  localStorage.setItem("g4_session_id", state.sessionId);
  updateHeader();
  return state.sessionId;
}

function readEvents() {
  const raw = Number.parseInt($("events-input")?.value || "100", 10);
  return Number.isFinite(raw) && raw > 0 ? raw : 100;
}

function sendPreset(text, extraPayload={}) {
  const input = $("prompt-input");
  if (!input || !text) return;
  input.value = text;
  sendTurn(extraPayload);
}

function clearSuggestionBars() {
  document.querySelectorAll(".pending-bar,.suggestion-bar").forEach(e => e.remove());
}

function normalizeSuggestion(item) {
  if (!item) return null;
  if (typeof item === "string") {
    const text = item.trim();
    return text ? {label: text, prefill: text} : null;
  }
  if (typeof item === "object") {
    const label = String(item.text || item.label || item.title || item.prefill || "").trim();
    const prefill = String(item.prefill || item.text || item.label || item.title || "").trim();
    const action = item.action || item.event || null;
    const confirmationEvent = item.confirmation_event || item.confirmationEvent || null;
    return label && prefill ? {label, prefill, action, confirmationEvent} : null;
  }
  const text = String(item).trim();
  return text ? {label: text, prefill: text} : null;
}

function showSuggestions(data) {
  const suggestions = [];
  const dialogueSuggestions = data?.dialogue?.next_suggestions;
  const answerOptions = data?.answer?.next_options;
  const llmSuggestions = data?.suggestions;
  if (Array.isArray(dialogueSuggestions)) suggestions.push(...dialogueSuggestions);
  if (Array.isArray(answerOptions)) suggestions.push(...answerOptions);
  if (Array.isArray(llmSuggestions)) suggestions.push(...llmSuggestions);
  const unique = [];
  const seen = new Set();
  suggestions.map(normalizeSuggestion).filter(Boolean).forEach((item) => {
    const key = item.prefill;
    if (!seen.has(key)) {
      seen.add(key);
      unique.push(item);
    }
  });
  unique.splice(4);
  if (!unique.length) return;
  const bar = document.createElement("div");
  bar.className = "suggestion-bar";
  unique.forEach((item) => {
    const button = document.createElement("button");
    button.className = "suggestion-btn btn-ghost sm";
    button.type = "button";
    button.textContent = item.label;
    button.dataset.prefill = item.prefill;
    button.title = item.prefill;
    button.setAttribute("aria-label", item.prefill);
    button.addEventListener("click", () => {
      const extraPayload = {};
      if (item.action) extraPayload.action_event = item.action;
      if (item.confirmationEvent) extraPayload.confirmation_event = item.confirmationEvent;
      sendPreset(item.prefill, extraPayload);
    });
    bar.appendChild(button);
  });
  const area = document.querySelector(".composer-area");
  const comp = area?.querySelector(".composer");
  if (area && comp) area.insertBefore(bar, comp);
}

function showPending(data) {
  clearSuggestionBars();
  if (!data?.pending_action?.requires_confirmation) return;
  const bar = document.createElement("div");
  bar.className = "pending-bar";
  bar.innerHTML = `<span>${state.lang==='zh'?'等待确认运行':'Waiting run confirmation'}</span>
    <button class="btn-primary sm" id="pb-confirm">${state.lang==='zh'?'确认运行':'Confirm'}</button>
    <button class="btn-ghost sm" id="pb-cancel">${state.lang==='zh'?'取消':'Cancel'}</button>`;
  const area = document.querySelector(".composer-area");
  const comp = area?.querySelector(".composer");
  if (area && comp) area.insertBefore(bar, comp);
  const actionId = data.pending_action.action_id || data.pending_action.intent || data.pending_action.kind || "pending";
  const confirmButton = bar.querySelector("#pb-confirm");
  const cancelButton = bar.querySelector("#pb-cancel");
  if (confirmButton) {
    confirmButton.addEventListener("click", (event) => {
      event.stopImmediatePropagation();
      sendPreset(
        state.lang==='zh'?'确认运行':'confirm run',
        {confirmation_event: {action_id: actionId, decision: "confirm"}}
      );
    });
  }
  if (cancelButton) {
    cancelButton.addEventListener("click", (event) => {
      event.stopImmediatePropagation();
      sendPreset(
        state.lang==='zh'?'取消运行':'cancel run',
        {confirmation_event: {action_id: actionId, decision: "cancel"}}
      );
    });
  }
}

function setSending(v) {
  state.sending = v;
  const sb = $("send-btn");
  if (sb) sb.disabled = v;
}

// Main flow
async function sendTurn(extraPayload={}) {
  if (state.sending) return;
  const input = $("prompt-input");
  const text = (input?.value||"").trim();
  if (!text) return;
  const sessionId = ensureSessionId();
  addMsg("user", esc(text));
  if (input) input.value = "";
  setSending(true);
  try {
    const turnPayload = {
      session_id: sessionId,
      text: text,
      lang: state.lang,
      locale: state.lang === 'zh' ? 'zh-CN' : 'en-US',
      events: readEvents(),
      llm_design_enabled: true,
      auto_discover_runtime: true,
      allow_in_memory: false,
      ...extraPayload,
    };
    if (state.llmConfigPath) turnPayload.llm_config_path = state.llmConfigPath;
    const data = await postJson("/api/v3/agent/turn", turnPayload);
    addAgentMsg(data);
    // Save session
    if (data.state?.session_id) {
      state.sessionId = data.state.session_id;
      localStorage.setItem("g4_session_id", data.state.session_id);
    }
    showPending(data);
    if (!data?.pending_action?.requires_confirmation) showSuggestions(data);
    await refreshV3AgentState();
  } catch (e) {
    addMsg("agent", `${t('sendFailed')}: ${esc(e.message||String(e))}`, "error");
  } finally {
    setSending(false);
  }
}

// Init
async function refreshV3AgentState() {
  if (!state.sessionId) return null;
  try {
    const data = await postJson("/api/v3/agent/state", {session_id: state.sessionId, lang: state.lang}, 10000);
    const phase = data.summary?.phase || "idle";
    const pp = $("phase-pill");
    if (pp) pp.textContent = phase;
    return data;
  } catch (e) {
    return null;
  }
}

async function loadRuntimeConfig() {
  const select = $("model-config-select");
  if (!select) return;
  try {
    const data = await postJson("/api/runtime", {});
    const available = Array.isArray(data.available) ? data.available : [];
    const current = data.current_path || "";
    select.innerHTML = "";
    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = current ? "Server default" : "No model config";
    select.appendChild(defaultOption);
    available.forEach((item) => {
      const option = document.createElement("option");
      option.value = item.path || "";
      option.textContent = [item.provider, item.model].filter(Boolean).join(" / ") || item.name || item.path || "config";
      if (option.value && option.value === current) option.selected = true;
      select.appendChild(option);
    });
    state.llmConfigPath = select.value || "";
    select.addEventListener("change", () => { state.llmConfigPath = select.value || ""; });
  } catch (e) {
    select.innerHTML = '<option value="">Server default</option>';
    state.llmConfigPath = "";
  }
}

function initUI() {
  ["send-btn","prompt-input","reset-btn","lang-select","timeline"].forEach(id => {
    if (!$(id)) { fetch('/api/log',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({level:'error',message:'MISSING DOM: #'+id})}).catch(()=>{}); }
  });
  fetch('/api/log',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({level:'info',message:'UI initialized, domReady='+(document.readyState)})}).catch(()=>{});
  const sb = $("send-btn");
  const pi = $("prompt-input");
  const rb = $("reset-btn");
  const ls = $("lang-select");
  if (sb) sb.addEventListener("click", () => sendTurn());
  if (pi) pi.addEventListener("keydown", e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendTurn(); } });
  bindQuickActions();
  if (rb) rb.addEventListener("click", async () => {
    if (state.sessionId) { await postJson("/api/v3/agent/reset",{session_id:state.sessionId}).catch(()=>{}); }
    state.sessionId = ""; localStorage.removeItem("g4_session_id");
    clearSuggestionBars();
    updateHeader();
    const pp = $("phase-pill");
    if (pp) pp.textContent = "idle";
    $("timeline").innerHTML = `<div class="welcome"><h2>Geant4 Agent</h2><p>${t('welcome')}</p>
      <div class="quick-actions">
        <button class="quick-btn" data-text="Evaluate 1 MeV gamma transmission through a 10 mm lead slab">Gamma shielding</button>
        <button class="quick-btn" data-text="Model a proton beam depth-dose curve in a water phantom">Proton depth dose</button>
        <button class="quick-btn" data-text="Estimate astronaut proton radiation exposure in space">Space exposure</button>
        <button class="quick-btn" data-text="Run NDT contrast simulation for a void inside an aluminum block">NDT void contrast</button>
      </div></div>`;
    bindQuickActions($("timeline"));
  });
  if (ls) { ls.value = state.lang; ls.addEventListener("change", e => { state.lang = e.target.value; localStorage.setItem("g4_lang", state.lang); }); }
  loadRuntimeConfig();
}
// Run init when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initUI);
} else {
  initUI();
}

// Header update
function updateHeader() {
  const sp = $("session-pill");
  if (sp) sp.textContent = state.sessionId ? `session: ${state.sessionId.slice(0,12)}` : "new session";
}
setInterval(updateHeader, 1000);
updateHeader();

// Desktop mode
if (new URLSearchParams(window.location.search).get("desktop") === "1") {
  document.body.classList.add("desktop");
}

// Window controls (null-safe for browser)
$("window-min-btn")?.addEventListener?.("click", () => window.geant4Desktop?.minimize?.());
$("window-max-btn")?.addEventListener?.("click", () => window.geant4Desktop?.toggleMaximize?.());
$("window-close-btn")?.addEventListener?.("click", () => { if (window.geant4Desktop?.close) window.geant4Desktop.close(); else window.close(); });
