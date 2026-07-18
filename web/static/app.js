const STORAGE_KEY = "minichat-web-state-v1";

const elements = {
  form: document.querySelector("#chatForm"),
  input: document.querySelector("#messageInput"),
  send: document.querySelector("#sendButton"),
  messages: document.querySelector("#messages"),
  empty: document.querySelector("#emptyState"),
  newChat: document.querySelector("#newChatButton"),
  mobileNewChat: document.querySelector("#mobileNewChatButton"),
  mobileSettings: document.querySelector("#mobileSettingsButton"),
  sidebar: document.querySelector("#sidebar"),
  sidebarClose: document.querySelector("#sidebarCloseButton"),
  sidebarScrim: document.querySelector("#sidebarScrim"),
  temperature: document.querySelector("#temperature"),
  temperatureValue: document.querySelector("#temperatureValue"),
  topP: document.querySelector("#topP"),
  topPValue: document.querySelector("#topPValue"),
  maxTokens: document.querySelector("#maxTokens"),
  maxTokensValue: document.querySelector("#maxTokensValue"),
  modelStatus: document.querySelector("#modelStatus"),
  statusText: document.querySelector("#statusText"),
  contextUsage: document.querySelector("#contextUsage"),
  toast: document.querySelector("#toast"),
};

let state = {
  history: [],
  temperature: 0.7,
  topP: 0.85,
  maxTokens: 128,
};
let generating = false;
let toastTimer = null;

function loadState() {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY));
    if (saved && Array.isArray(saved.history)) {
      state.history = saved.history.slice(-8);
      state.temperature = Number(saved.temperature ?? state.temperature);
      state.topP = Number(saved.topP ?? state.topP);
      state.maxTokens = Number(saved.maxTokens ?? state.maxTokens);
    }
  } catch (_error) {
    localStorage.removeItem(STORAGE_KEY);
  }
}

function saveState() {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

function syncControls() {
  elements.temperature.value = String(state.temperature);
  elements.temperatureValue.value = state.temperature.toFixed(2);
  elements.topP.value = String(state.topP);
  elements.topPValue.value = state.topP.toFixed(2);
  elements.maxTokens.value = String(state.maxTokens);
  elements.maxTokensValue.value = String(state.maxTokens);
}

function showToast(message) {
  elements.toast.textContent = message;
  elements.toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => elements.toast.classList.remove("visible"), 3600);
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    elements.messages.scrollTop = elements.messages.scrollHeight;
  });
}

function createMessage(role, text, stats = null) {
  const row = document.createElement("article");
  row.className = `message-row ${role}`;
  const inner = document.createElement("div");
  inner.className = "message-inner";
  const avatar = document.createElement("div");
  avatar.className = "message-avatar";
  avatar.textContent = role === "assistant" ? "M" : "你";
  const body = document.createElement("div");
  body.className = "message-body";
  const content = document.createElement("div");
  content.className = "message-content";
  content.textContent = text;
  body.append(content);
  if (stats) {
    const metadata = document.createElement("div");
    metadata.className = "message-stats";
    metadata.textContent = `${stats.elapsed_seconds.toFixed(2)} 秒 · ${stats.generated_tokens} tokens`;
    body.append(metadata);
  }
  inner.append(avatar, body);
  row.append(inner);
  return row;
}

function createTypingMessage() {
  const row = createMessage("assistant", "");
  row.id = "typingMessage";
  const content = row.querySelector(".message-content");
  content.className = "typing";
  content.innerHTML = "<span></span><span></span><span></span>";
  return row;
}

function renderHistory() {
  elements.messages.replaceChildren();
  if (state.history.length === 0) {
    elements.messages.append(elements.empty);
    elements.empty.hidden = false;
    elements.contextUsage.textContent = "0 / 512";
    return;
  }
  elements.empty.hidden = true;
  for (const turn of state.history) {
    elements.messages.append(createMessage("user", turn.user));
    elements.messages.append(createMessage("assistant", turn.assistant, turn.stats || null));
  }
  updateContextEstimate();
  scrollToBottom();
}

function updateContextEstimate() {
  const characters = state.history.reduce(
    (sum, turn) => sum + turn.user.length + turn.assistant.length,
    0,
  );
  const estimate = Math.min(512, Math.ceil(characters * 0.75));
  elements.contextUsage.textContent = `${estimate} / 512`;
}

function resizeInput() {
  elements.input.style.height = "auto";
  elements.input.style.height = `${Math.min(elements.input.scrollHeight, 150)}px`;
}

function setGenerating(value) {
  generating = value;
  elements.send.disabled = value || !elements.input.value.trim();
  elements.input.readOnly = value;
}

async function sendMessage(message) {
  if (generating || !message.trim()) return;
  const cleanMessage = message.trim();
  elements.input.value = "";
  resizeInput();
  elements.empty.hidden = true;
  if (state.history.length === 0) elements.messages.replaceChildren();
  elements.messages.append(createMessage("user", cleanMessage));
  elements.messages.append(createTypingMessage());
  setGenerating(true);
  scrollToBottom();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: cleanMessage,
        history: state.history.map(({ user, assistant }) => ({ user, assistant })),
        temperature: state.temperature,
        top_p: state.topP,
        max_new_tokens: state.maxTokens,
        repetition_penalty: 1.25,
      }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "生成失败");
    document.querySelector("#typingMessage")?.remove();
    const stats = {
      elapsed_seconds: Number(result.elapsed_seconds),
      generated_tokens: Number(result.generated_tokens),
    };
    elements.messages.append(createMessage("assistant", result.answer, stats));
    state.history.push({ user: cleanMessage, assistant: result.answer, stats });
    state.history = state.history.slice(-8);
    saveState();
    updateContextEstimate();
  } catch (error) {
    document.querySelector("#typingMessage")?.remove();
    showToast(error.message || "无法连接本地模型");
  } finally {
    setGenerating(false);
    elements.input.focus();
    scrollToBottom();
  }
}

function clearConversation() {
  if (generating) return;
  state.history = [];
  saveState();
  renderHistory();
  elements.input.focus();
}

function setSidebarOpen(open) {
  elements.sidebar.classList.toggle("open", open);
  elements.sidebarScrim.classList.toggle("visible", open);
}

async function checkHealth() {
  try {
    const response = await fetch("/api/health", { cache: "no-store" });
    if (!response.ok) throw new Error();
    const result = await response.json();
    elements.modelStatus.className = "model-status ready";
    elements.statusText.textContent = result.model.device === "cuda" ? "GPU 已就绪" : "CPU 已就绪";
  } catch (_error) {
    elements.modelStatus.className = "model-status error";
    elements.statusText.textContent = "模型未连接";
  }
}

elements.form.addEventListener("submit", (event) => {
  event.preventDefault();
  sendMessage(elements.input.value);
});

elements.input.addEventListener("input", () => {
  resizeInput();
  elements.send.disabled = generating || !elements.input.value.trim();
});

elements.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    sendMessage(elements.input.value);
  }
});

elements.newChat.addEventListener("click", clearConversation);
elements.mobileNewChat.addEventListener("click", clearConversation);
elements.mobileSettings.addEventListener("click", () => setSidebarOpen(true));
elements.sidebarClose.addEventListener("click", () => setSidebarOpen(false));
elements.sidebarScrim.addEventListener("click", () => setSidebarOpen(false));

elements.temperature.addEventListener("input", () => {
  state.temperature = Number(elements.temperature.value);
  elements.temperatureValue.value = state.temperature.toFixed(2);
  saveState();
});

elements.topP.addEventListener("input", () => {
  state.topP = Number(elements.topP.value);
  elements.topPValue.value = state.topP.toFixed(2);
  saveState();
});

elements.maxTokens.addEventListener("input", () => {
  state.maxTokens = Number(elements.maxTokens.value);
  elements.maxTokensValue.value = String(state.maxTokens);
  saveState();
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => sendMessage(button.dataset.prompt));
});

loadState();
syncControls();
renderHistory();
resizeInput();
setGenerating(false);
checkHealth();
elements.input.focus();
