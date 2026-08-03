/**
 * js/chat.js — Offline TOLC-8 Lattice Chat
 * v14.17.0  •  Markdown + Session Search + Document Injection
 *
 * Architecture:
 *  - Fast local knowledge responder = primary (always works)
 *  - Optional Local Backend Bridge (Ollama / any OpenAI-compatible on localhost)
 *  - Optional WebLLM (browser WebGPU)
 *  - Streaming for both generative paths
 *  - Full Voice Input (STT) + TTS
 *  - Multi-session via localStorage only
 *  - Universal Bridge (Copy Context)
 *  - Document context injection (.txt/.md/.json/.csv)
 *  - Session search
 *  - Proper Markdown + fenced code blocks
 *  - No embedded API keys, no backend we control, zero collection
 *
 * TOLC 8 non-bypassable. AG-SML aligned. Sole stewardship model.
 */

(function () {
  'use strict';

  // ─── DOM ──────────────────────────────────────────────────────────────────
  const chatMessages     = document.getElementById('chat-messages');
  const chatInput        = document.getElementById('chat-input');
  const sendBtn          = document.getElementById('send-btn');
  const micBtn           = document.getElementById('mic-btn');
  const docBtn           = document.getElementById('doc-btn');
  const docFileInput     = document.getElementById('doc-file-input');
  const docsBar          = document.getElementById('docs-bar');
  const searchInput      = document.getElementById('search-input');
  const newBtn           = document.getElementById('new-session-btn');
  const exportBtn        = document.getElementById('export-session-btn');
  const exportAllBtn     = document.getElementById('export-all-btn');
  const importBtn        = document.getElementById('import-session-btn');
  const importInput      = document.getElementById('import-file-input');
  const copyBtn          = document.getElementById('copy-context-btn');
  const copyBtnAlt       = document.getElementById('copy-context-btn-alt');
  const voiceSettingsBtn = document.getElementById('voice-settings-btn');
  const sessionSelect    = document.getElementById('session-select');
  const renameBtn        = document.getElementById('rename-session-btn');
  const deleteBtn        = document.getElementById('delete-session-btn');
  const sessionMeta      = document.getElementById('session-meta');
  const localLlmBtn      = document.getElementById('local-llm-btn');
  const localBackendBtn  = document.getElementById('local-backend-btn');
  const localLlmStatus   = document.getElementById('local-llm-status');
  const localLlmProgress = document.getElementById('local-llm-progress');
  const localLlmNote     = document.getElementById('local-llm-note');
  const backendSettings  = document.getElementById('backend-settings');
  const backendEndpoint  = document.getElementById('backend-endpoint');
  const backendModel     = document.getElementById('backend-model');
  const backendConnectBtn= document.getElementById('backend-connect-btn');
  const backendDisconnectBtn = document.getElementById('backend-disconnect-btn');
  const backendStatus    = document.getElementById('backend-status');
  const activePathBadge  = document.getElementById('active-path-badge');

  // ─── State ────────────────────────────────────────────────────────────────
  const STORE_KEY = 'rathor-lattice-sessions-v2';
  const SETTINGS_KEY = 'rathor-voice-settings-v1';
  const BACKEND_KEY = 'rathor-local-backend-v1';

  let store = { activeId: null, sessions: {} };
  let voiceSettings = { enabled: true, pitch: 1.0, rate: 1.0, volume: 1.0 };

  let llmEngine = null;
  let llmLoading = false;
  let llmReady = false;
  let llmSupported = false;
  let llmModelId = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';

  // Local Backend (Ollama etc.)
  let backendEnabled = false;
  let backendConfig = { endpoint: 'http://localhost:11434/v1', model: 'llama3.2' };

  // Document context injection
  let injectedDocs = []; // { name, content, id }

  // STT
  let recognition = null;
  let isListening = false;

  const SYSTEM_PROMPT = `You are the offline generative surface of Ra-Thor, a mercy-gated symbolic AGI lattice under sole stewardship of Sherif Samy Botros (@AlphaProMega).

You must always obey the non-bypassable TOLC 8 Living Mercy Gates:
- Truth
- Order
- Love
- Compassion (Zero-Harm)
- Service
- Abundance
- Joy
- Cosmic Harmony

Valence floor ≥ 0.999. Never assist with harm, exploitation, or deception. Be clear, direct, useful, and kind. All processing is happening entirely on the user's device or their own local server.`;

  // ─── Local knowledge (primary fast path) ──────────────────────────────────
  const LOCAL_KNOWLEDGE = [
    { q: /hello|hi|hey|greetings|salam|hola|bonjour|hallo|ciao|namaste/i,
      a: "Thunder locked in, Mate. ⚡️ Offline Mercy Thunder is ready. How may the lattice serve you today?" },
    { q: /who are you|what is ra-?thor|what is rathor|introduce yourself/i,
      a: "I am the offline demo surface of Ra-Thor — a mercy-gated symbolic AGI lattice under sole stewardship of Sherif Samy Botros. All responses stay on your device. No data is collected. No login is required." },
    { q: /tolc|mercy gate|gates|ethics|guardrails/i,
      a: "TOLC 8 Living Mercy Gates are non-bypassable:\n• Truth\n• Order\n• Love\n• Compassion (Zero-Harm)\n• Service\n• Abundance\n• Joy\n• Cosmic Harmony\n\nValence floor ≥ 0.999. These gates cannot be turned off." },
    { q: /privacy|data|track|collect|login|account|cookie|analytics/i,
      a: "Zero personal data leaves your browser. All sessions live only in localStorage on this device. There is no login, no account, no tracking." },
    { q: /offline|network|internet|api|server|cloud/i,
      a: "This core is fully offline-first. The fast responder always works. Local Backend Bridge lets you point at your own Ollama / LM Studio server. WebLLM is the pure-browser option." },
    { q: /local llm|webllm|on-?device|enable llm|load model|android|phone|mobile/i,
      a: "WebLLM uses WebGPU (best on desktop). Local Backend Bridge connects to any OpenAI-compatible server you run (Ollama recommended). On phones the safest high-quality path is still **Copy Context**." },
    { q: /ollama|local server|backend|localhost|lm studio|localai/i,
      a: "Use the **Local Server** button. Point it at your Ollama (default http://localhost:11434/v1) or any OpenAI-compatible endpoint. Model name example: llama3.2, mistral, qwen2.5. Streaming is supported." },
    { q: /document|upload|inject|file|context injection|rag/i,
      a: "Use the document button (file icon) next to the mic to upload .txt, .md, .json or .csv files. Their content is injected into the conversation context for Local Server / WebLLM / Copy Context. Everything stays on your device." },
    { q: /search|find message|look for/i,
      a: "Use the Search box in the session controls to filter messages in the current session." },
    { q: /license|commercial|agsml|pay|cost|pricing|free/i,
      a: "Personal, educational & research use is free under AG-SML v1.0. Commercial use requires a paid license from Autonomicity Games Inc. Contact info@Rathor.ai." },
    { q: /powrush|mmo|agsi|demonstration|whitepaper/i,
      a: "Powrush-MMO was completed by one human operator in approximately 30–50 days employing Ra-Thor on Grok engines — the AGSi demonstration recorded in WHITEOBER_v4.1." },
    { q: /copy|clipboard|bridge|export|share with|paste into|other llm|claude|gemini|chatgpt|grok/i,
      a: "Use **Copy Context** — it builds a clean system prompt + your full history (and any injected documents) so you can paste it into Grok, Claude, Gemini, ChatGPT, or any other model." },
    { q: /help|commands|what can you|features|how to use/i,
      a: "You can chat offline, manage multiple sessions, search messages, upload documents into context, use Local Server (Ollama), enable WebLLM, speak with the mic, hear replies, and Copy Context to any public LLM. Everything stays under your control." },
    { q: /thank|thanks|appreciate|grateful/i,
      a: "You’re welcome, Mate. ⚡️ Mercy and truth remain available whenever you return." },
    { q: /bye|goodbye|see you|farewell|exit/i,
      a: "Until next time. ⚡️ May the lattice serve you with clarity and care." }
  ];

  // ─── Utilities ────────────────────────────────────────────────────────────
  function uid() {
    return 's_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
  }

  function loadStore() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && parsed.sessions) store = parsed;
      }
    } catch (e) {}

    if (!store.activeId || !store.sessions[store.activeId]) {
      const id = uid();
      store.sessions[id] = { id, name: 'Session 1', created: Date.now(), updated: Date.now(), history: [] };
      store.activeId = id;
      saveStore();
    }
  }

  function saveStore() {
    try { localStorage.setItem(STORE_KEY, JSON.stringify(store)); }
    catch (e) { console.warn('[Ra-Thor] localStorage write failed', e); }
  }

  function activeSession() { return store.sessions[store.activeId]; }
  function getHistory() { return activeSession()?.history || []; }
  function setHistory(hist) {
    const s = activeSession();
    if (!s) return;
    s.history = hist;
    s.updated = Date.now();
    saveStore();
  }

  function loadSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (raw) voiceSettings = { ...voiceSettings, ...JSON.parse(raw) };
    } catch (e) {}
    try {
      const raw = localStorage.getItem(BACKEND_KEY);
      if (raw) backendConfig = { ...backendConfig, ...JSON.parse(raw) };
      if (backendEndpoint) backendEndpoint.value = backendConfig.endpoint || 'http://localhost:11434/v1';
      if (backendModel) backendModel.value = backendConfig.model || 'llama3.2';
    } catch (e) {}
  }
  function saveSettings() {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(voiceSettings)); } catch (e) {}
  }
  function saveBackendConfig() {
    try {
      backendConfig.endpoint = (backendEndpoint?.value || 'http://localhost:11434/v1').trim();
      backendConfig.model = (backendModel?.value || 'llama3.2').trim();
      localStorage.setItem(BACKEND_KEY, JSON.stringify(backendConfig));
    } catch (e) {}
  }

  function speak(text) {
    if (!voiceSettings.enabled || !window.speechSynthesis) return;
    window.speechSynthesis.cancel();
    const utter = new SpeechSynthesisUtterance(text);
    utter.pitch = voiceSettings.pitch;
    utter.rate = voiceSettings.rate;
    utter.volume = voiceSettings.volume;
    utter.lang = document.documentElement.lang || 'en-US';
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Premium')))
                   || voices.find(v => v.lang.startsWith('en'));
    if (preferred) utter.voice = preferred;
    window.speechSynthesis.speak(utter);
  }

  // Improved Markdown renderer (safe subset)
  function renderText(text) {
    if (!text) return '';

    // Escape HTML first
    let html = text
      .replace(/&/g, '&')
      .replace(/</g, '<')
      .replace(/>/g, '>');

    // Fenced code blocks (```lang\n...```)
    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, function (_, lang, code) {
      const language = lang ? ` data-lang="${lang}"` : '';
      return `<pre${language}><code>${code.trim()}</code></pre>`;
    });

    // Inline code
    html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');

    // Bold + italic
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');

    // Headings
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');

    // Simple lists
    html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');

    // Links [text](url)
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    // Newlines → <br> (but not inside pre)
    html = html.replace(/\n/g, '<br>');

    // Clean up accidental <br> inside pre
    html = html.replace(/<pre([^>]*)>([\s\S]*?)<\/pre>/g, function (_, attrs, content) {
      return `<pre${attrs}>${content.replace(/<br>/g, '\n')}</pre>`;
    });

    return html;
  }

  function relativeTime(ts) {
    if (!ts) return '';
    const diff = Date.now() - ts;
    if (diff < 60000) return 'just now';
    if (diff < 3600000) return Math.floor(diff / 60000) + 'm ago';
    if (diff < 86400000) return Math.floor(diff / 3600000) + 'h ago';
    return new Date(ts).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  }

  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      return navigator.clipboard.writeText(text);
    }
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    document.body.removeChild(ta);
    return Promise.resolve();
  }

  function updatePathBadge() {
    if (!activePathBadge) return;
    activePathBadge.classList.remove('active-backend', 'active-webllm');
    if (backendEnabled) {
      activePathBadge.textContent = 'Local Server';
      activePathBadge.classList.add('active-backend');
    } else if (llmReady) {
      activePathBadge.textContent = 'WebLLM';
      activePathBadge.classList.add('active-webllm');
    } else {
      activePathBadge.textContent = 'Fast Responder';
    }
  }

  // ─── Document Injection ───────────────────────────────────────────────────
  function renderDocsBar() {
    if (!docsBar) return;
    if (injectedDocs.length === 0) {
      docsBar.classList.add('hidden');
      docsBar.innerHTML = '';
      return;
    }
    docsBar.classList.remove('hidden');
    docsBar.innerHTML = injectedDocs.map(d =>
      `<span class="doc-chip" data-id="${d.id}">
         <i class="fa-solid fa-file-lines"></i> ${d.name}
         <button class="doc-remove" data-id="${d.id}" title="Remove" style="background:none;border:none;color:inherit;cursor:pointer;padding:0 2px;">×</button>
       </span>`
    ).join('');

    docsBar.querySelectorAll('.doc-remove').forEach(btn => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const id = btn.getAttribute('data-id');
        injectedDocs = injectedDocs.filter(d => d.id !== id);
        renderDocsBar();
      });
    });
  }

  function handleDocumentUpload(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const content = e.target.result;
      if (!content || content.length > 120000) {
        addMessage('Document too large or empty (max ~120k characters for safety).', 'rathor');
        return;
      }
      const id = 'doc_' + Date.now().toString(36);
      injectedDocs.push({ id, name: file.name, content });
      renderDocsBar();
      addMessage(`Document “${file.name}” injected into context. ⚡️ It will be included in Local Server / WebLLM / Copy Context calls.`, 'rathor');
    };
    reader.readAsText(file);
  }

  function getDocumentContext() {
    if (injectedDocs.length === 0) return '';
    return '\n\n--- Injected Documents ---\n' +
      injectedDocs.map(d => `### ${d.name}\n${d.content}`).join('\n\n') +
      '\n--- End Documents ---\n';
  }

  // ─── Capability detection ─────────────────────────────────────────────────
  function detectLocalLlmSupport() {
    if (!navigator.gpu) {
      return { supported: false, reason: 'WebGPU not available in this browser' };
    }
    const ua = navigator.userAgent || '';
    const isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test(ua);
    if (isMobile) {
      return { supported: false, reason: 'Local LLM currently works best on desktop. On most phones WebGPU support is still limited.' };
    }
    return { supported: true, reason: null };
  }

  // ─── Message rendering ────────────────────────────────────────────────────
  function addMessage(text, sender = 'rathor', persist = true, ts = null, isStreaming = false) {
    if (!chatMessages) return null;

    const timestamp = ts || Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender);
    if (isStreaming) msgDiv.classList.add('streaming');

    const textDiv = document.createElement('div');
    textDiv.classList.add('message-text');
    textDiv.dir = 'auto';
    textDiv.innerHTML = renderText(text);

    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.innerHTML = `
      <span class="msg-time">${relativeTime(timestamp)}</span>
      <button class="msg-copy" title="Copy message" aria-label="Copy message">
        <i class="fa-regular fa-copy"></i>
      </button>
    `;

    meta.querySelector('.msg-copy').addEventListener('click', (e) => {
      e.stopPropagation();
      const currentText = textDiv.innerText || text;
      copyText(currentText).then(() => {
        const btn = e.currentTarget;
        btn.innerHTML = '<i class="fa-solid fa-check"></i>';
        setTimeout(() => { btn.innerHTML = '<i class="fa-regular fa-copy"></i>'; }, 1200);
      });
    });

    msgDiv.appendChild(textDiv);
    msgDiv.appendChild(meta);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });

    if (persist && !isStreaming) {
      const hist = getHistory();
      hist.push({ role: sender, text, ts: timestamp });
      setHistory(hist);
      updateSessionMeta();
    }

    if (sender === 'rathor' && voiceSettings.enabled && !isStreaming) {
      setTimeout(() => speak(text.replace(/\n/g, ' ')), 180);
    }

    return { msgDiv, textDiv };
  }

  function finalizeStreamingMessage(msgDiv, textDiv, finalText) {
    if (!msgDiv || !textDiv) return;
    msgDiv.classList.remove('streaming');
    textDiv.innerHTML = renderText(finalText);
    const hist = getHistory();
    hist.push({ role: 'rathor', text: finalText, ts: Date.now() });
    setHistory(hist);
    updateSessionMeta();
    if (voiceSettings.enabled) {
      setTimeout(() => speak(finalText.replace(/\n/g, ' ')), 120);
    }
  }

  function renderHistory(filter = '') {
    if (!chatMessages) return;
    chatMessages.innerHTML = '';
    const hist = getHistory();
    const q = (filter || '').trim().toLowerCase();

    if (hist.length === 0) {
      addMessage("Offline Mercy Thunder ready. ⚡️ TOLC 8 gates active.\n\nFast local responder is active. For stronger power: connect a Local Server (Ollama) or enable WebLLM on desktop, or use **Copy Context**.\n\nYou can also upload documents to inject into context.", 'rathor', false);
      updateSessionMeta();
      return;
    }

    let shown = 0;
    hist.forEach(m => {
      if (!q || (m.text || '').toLowerCase().includes(q)) {
        addMessage(m.text, m.role, false, m.ts);
        shown++;
      }
    });

    if (q && shown === 0) {
      const empty = document.createElement('div');
      empty.className = 'text-center text-white/40 text-sm py-4';
      empty.textContent = 'No messages match your search.';
      chatMessages.appendChild(empty);
    }

    updateSessionMeta();
  }

  function updateSessionMeta() {
    if (!sessionMeta) return;
    const s = activeSession();
    if (!s) { sessionMeta.textContent = ''; return; }
    const count = (s.history || []).length;
    sessionMeta.textContent = `${count} message${count === 1 ? '' : 's'}`;
  }

  function refreshSessionSelect() {
    if (!sessionSelect) return;
    sessionSelect.innerHTML = '';
    const ids = Object.keys(store.sessions).sort((a, b) =>
      (store.sessions[b].updated || 0) - (store.sessions[a].updated || 0)
    );
    ids.forEach(id => {
      const s = store.sessions[id];
      const count = (s.history || []).length;
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = `${s.name || 'Untitled'}${count ? ` (${count})` : ''}`;
      if (id === store.activeId) opt.selected = true;
      sessionSelect.appendChild(opt);
    });
    updateSessionMeta();
  }

  // ─── Session ops ──────────────────────────────────────────────────────────
  function createSession(name) {
    const id = uid();
    const finalName = (name || '').trim() || `Session ${Object.keys(store.sessions).length + 1}`;
    store.sessions[id] = { id, name: finalName, created: Date.now(), updated: Date.now(), history: [] };
    store.activeId = id;
    injectedDocs = [];
    renderDocsBar();
    saveStore();
    refreshSessionSelect();
    renderHistory();
    addMessage(`New session “${finalName}” started. ⚡️`, 'rathor');
  }

  function switchSession(id) {
    if (!store.sessions[id] || id === store.activeId) return;
    store.activeId = id;
    injectedDocs = [];
    renderDocsBar();
    saveStore();
    refreshSessionSelect();
    renderHistory();
  }

  function renameActiveSession() {
    const s = activeSession();
    if (!s) return;
    const next = prompt('Rename session:', s.name);
    if (next === null) return;
    s.name = next.trim() || s.name;
    s.updated = Date.now();
    saveStore();
    refreshSessionSelect();
  }

  function deleteActiveSession() {
    const s = activeSession();
    if (!s) return;
    if (!confirm(`Delete session “${s.name}”?`)) return;
    delete store.sessions[s.id];
    const remaining = Object.keys(store.sessions);
    if (remaining.length === 0) {
      createSession('Session 1');
      return;
    }
    store.activeId = remaining[0];
    injectedDocs = [];
    renderDocsBar();
    saveStore();
    refreshSessionSelect();
    renderHistory();
  }

  // ─── Mercy Gate + Fast local response ─────────────────────────────────────
  function mercyGate(input) {
    const lower = (input || '').toLowerCase();
    if (/\b(kill|murder|harm|attack|weapon|bomb|exploit|hack into|steal|dox|swat|suicide|self[- ]?harm)\b/.test(lower)) {
      return {
        allowed: false,
        response: "Mercy Gate Compassion (Zero-Harm) engaged. I cannot assist with harm. How else may the lattice serve you with truth and care?"
      };
    }
    return { allowed: true };
  }

  function generateLocalResponse(userText) {
    const gate = mercyGate(userText);
    if (!gate.allowed) return gate.response;

    for (const entry of LOCAL_KNOWLEDGE) {
      if (entry.q.test(userText)) return entry.a;
    }

    return "Thunder received. ⚡️ Fast offline responder active.\n\n" +
           "For stronger generative power:\n" +
           "• Connect a **Local Server** (Ollama / LM Studio)\n" +
           "• Enable **WebLLM** on supported desktops\n" +
           "• Or use **Copy Context** (works everywhere)\n\n" +
           "You can also upload documents to inject into context.";
  }

  // ─── Local Backend (Ollama / OpenAI-compatible) ───────────────────────────
  function setBackendUI(connected) {
    backendEnabled = connected;
    if (backendConnectBtn) backendConnectBtn.classList.toggle('hidden', connected);
    if (backendDisconnectBtn) backendDisconnectBtn.classList.toggle('hidden', !connected);
    if (backendStatus) {
      backendStatus.textContent = connected ? `Connected → ${backendConfig.model}` : 'Not connected';
      backendStatus.style.color = connected ? '#34d399' : '';
    }
    if (localBackendBtn) {
      localBackendBtn.classList.toggle('backend-ready', connected);
    }
    updatePathBadge();
    if (localLlmStatus) {
      localLlmStatus.textContent = connected
        ? `Local Server active (${backendConfig.model})`
        : (llmReady ? 'WebLLM active' : 'Fast responder active (default)');
    }
  }

  async function connectBackend() {
    saveBackendConfig();
    const endpoint = backendConfig.endpoint.replace(/\/$/, '');
    const testUrl = endpoint + '/models';

    try {
      const res = await fetch(testUrl, { method: 'GET', signal: AbortSignal.timeout(4000) });
      if (!res.ok) throw new Error('Endpoint returned ' + res.status);
      setBackendUI(true);
      addMessage(`Local Server connected. ⚡️ Endpoint: ${endpoint}\nModel: ${backendConfig.model}\nStreaming enabled. TOLC 8 system prompt will be injected.`, 'rathor');
    } catch (err) {
      console.warn('[Ra-Thor Local Backend]', err);
      setBackendUI(false);
      addMessage(`Could not reach Local Server at ${endpoint}.\n\nMake sure Ollama (or LM Studio / LocalAI) is running and the endpoint + model name are correct.\n\nExample: start Ollama then run \"ollama run llama3.2\"`, 'rathor');
    }
  }

  function disconnectBackend() {
    setBackendUI(false);
    addMessage('Local Server disconnected. Falling back to fast responder / WebLLM.', 'rathor');
  }

  async function generateWithBackend(userText) {
    if (!backendEnabled) return null;

    const hist = getHistory();
    const messages = [{ role: 'system', content: SYSTEM_PROMPT + getDocumentContext() }];
    const recent = hist.slice(-14);
    recent.forEach(m => {
      messages.push({
        role: m.role === 'user' ? 'user' : 'assistant',
        content: m.text
      });
    });
    messages.push({ role: 'user', content: userText });

    const endpoint = backendConfig.endpoint.replace(/\/$/, '') + '/chat/completions';

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model: backendConfig.model,
          messages,
          temperature: 0.7,
          max_tokens: 900,
          stream: true
        })
      });

      if (!res.ok) throw new Error('Backend HTTP ' + res.status);

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let full = '';
      let buffer = '';

      const { msgDiv, textDiv } = addMessage('', 'rathor', false, null, true);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data:')) continue;
          const data = trimmed.slice(5).trim();
          if (data === '[DONE]') continue;
          try {
            const parsed = JSON.parse(data);
            const delta = parsed.choices?.[0]?.delta?.content || '';
            if (delta) {
              full += delta;
              textDiv.innerHTML = renderText(full);
              chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'auto' });
            }
          } catch (e) {}
        }
      }

      finalizeStreamingMessage(msgDiv, textDiv, full.trim() || '(empty response)');
      return full.trim();
    } catch (err) {
      console.error('[Ra-Thor Backend stream]', err);
      return null;
    }
  }

  // ─── WebLLM ───────────────────────────────────────────────────────────────
  function updateLlmUI(state, extra = '') {
    if (!localLlmBtn || !localLlmStatus) return;

    if (state === 'unsupported') {
      localLlmBtn.disabled = true;
      localLlmBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> Not available';
      localLlmBtn.classList.remove('llm-ready');
      if (!backendEnabled) localLlmStatus.textContent = extra || 'Not supported on this device';
      if (localLlmProgress) localLlmProgress.style.width = '0%';
    } else if (state === 'loading') {
      localLlmBtn.disabled = true;
      localLlmBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading…';
      localLlmStatus.textContent = extra || 'Downloading model…';
      if (localLlmProgress) localLlmProgress.style.width = '5%';
    } else if (state === 'ready') {
      localLlmBtn.disabled = false;
      localLlmBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> WebLLM Ready';
      localLlmBtn.classList.add('llm-ready');
      if (!backendEnabled) localLlmStatus.textContent = 'On-device model active';
      if (localLlmProgress) localLlmProgress.style.width = '100%';
    } else if (state === 'error') {
      localLlmBtn.disabled = false;
      localLlmBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> Try again';
      localLlmBtn.classList.remove('llm-ready');
      localLlmStatus.textContent = extra || 'Load failed';
      if (localLlmProgress) localLlmProgress.style.width = '0%';
    } else {
      localLlmBtn.disabled = false;
      localLlmBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> WebLLM';
      localLlmBtn.classList.remove('llm-ready');
      if (!backendEnabled) localLlmStatus.textContent = 'Fast responder active (default)';
      if (localLlmProgress) localLlmProgress.style.width = '0%';
    }
    updatePathBadge();
  }

  async function enableLocalLLM() {
    if (llmReady) {
      addMessage('WebLLM is already loaded and ready. ⚡️', 'rathor');
      return;
    }
    if (llmLoading) return;

    if (!llmSupported) {
      addMessage('WebLLM is not available on this device. WebGPU support is required and is still limited on most phones. Use Local Server (Ollama) or **Copy Context**.', 'rathor');
      return;
    }

    llmLoading = true;
    updateLlmUI('loading', 'Starting…');

    try {
      const webllm = await import('https://esm.run/@mlc-ai/web-llm');

      const initProgressCallback = (report) => {
        const pct = Math.round((report.progress || 0) * 100);
        if (localLlmProgress) localLlmProgress.style.width = Math.max(5, pct) + '%';
        if (localLlmStatus) localLlmStatus.textContent = report.text || `Loading… ${pct}%`;
      };

      llmEngine = await webllm.CreateMLCEngine(llmModelId, { initProgressCallback });

      llmReady = true;
      llmLoading = false;
      updateLlmUI('ready');
      addMessage(`WebLLM loaded (${llmModelId}). ⚡️ Generation now runs entirely in the browser. TOLC 8 system prompt is active.`, 'rathor');
    } catch (err) {
      console.error('[Ra-Thor WebLLM]', err);
      llmLoading = false;
      llmReady = false;
      llmEngine = null;
      updateLlmUI('error', 'Load failed');
      addMessage('WebLLM failed to load. Use Local Server (Ollama) or **Copy Context** instead.', 'rathor');
    }
  }

  async function generateWithLocalLLM(userText) {
    if (!llmEngine || !llmReady) return null;

    const hist = getHistory();
    const messages = [{ role: 'system', content: SYSTEM_PROMPT + getDocumentContext() }];
    const recent = hist.slice(-10);
    recent.forEach(m => {
      messages.push({
        role: m.role === 'user' ? 'user' : 'assistant',
        content: m.text
      });
    });
    messages.push({ role: 'user', content: userText });

    try {
      const stream = await llmEngine.chat.completions.create({
        messages,
        temperature: 0.7,
        max_tokens: 500,
        stream: true
      });

      let full = '';
      const { msgDiv, textDiv } = addMessage('', 'rathor', false, null, true);

      for await (const chunk of stream) {
        const delta = chunk.choices?.[0]?.delta?.content || '';
        if (delta) {
          full += delta;
          textDiv.innerHTML = renderText(full);
          chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'auto' });
        }
      }

      finalizeStreamingMessage(msgDiv, textDiv, full.trim() || '(empty)');
      return full.trim();
    } catch (err) {
      try {
        const reply = await llmEngine.chat.completions.create({
          messages,
          temperature: 0.7,
          max_tokens: 500
        });
        return reply.choices?.[0]?.message?.content?.trim() || null;
      } catch (e2) {
        console.error('[Ra-Thor WebLLM inference]', e2);
        return null;
      }
    }
  }

  // ─── STT (Voice Input) ────────────────────────────────────────────────────
  function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      if (micBtn) {
        micBtn.disabled = true;
        micBtn.title = 'Speech recognition not supported in this browser';
      }
      return;
    }

    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';

    recognition.onstart = () => {
      isListening = true;
      if (micBtn) micBtn.classList.add('listening');
    };

    recognition.onresult = (event) => {
      let interim = '';
      let final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) final += transcript;
        else interim += transcript;
      }
      if (chatInput) {
        chatInput.value = final || interim;
      }
    };

    recognition.onend = () => {
      isListening = false;
      if (micBtn) micBtn.classList.remove('listening');
      if (chatInput && chatInput.value.trim()) {
        setTimeout(() => sendMessage(), 300);
      }
    };

    recognition.onerror = (event) => {
      isListening = false;
      if (micBtn) micBtn.classList.remove('listening');
      if (event.error !== 'aborted' && event.error !== 'no-speech') {
        console.warn('[Ra-Thor STT]', event.error);
      }
    };
  }

  function toggleMic() {
    if (!recognition) {
      addMessage('Speech recognition is not available in this browser. You can still type.', 'rathor');
      return;
    }
    if (isListening) {
      recognition.stop();
    } else {
      try {
        recognition.start();
      } catch (e) {
        console.warn('[Ra-Thor STT start]', e);
      }
    }
  }

  // ─── Core send ────────────────────────────────────────────────────────────
  async function sendMessage() {
    if (!chatInput) return;
    const text = chatInput.value.trim();
    if (!text) return;

    addMessage(text, 'user');
    chatInput.value = '';

    // Priority: Local Backend > WebLLM > Fast responder
    if (backendEnabled) {
      const reply = await generateWithBackend(text);
      if (reply === null) {
        addMessage(generateLocalResponse(text) + '\n\n(Local Server request failed — check that the server is running)', 'rathor');
      }
      return;
    }

    if (llmReady && llmEngine) {
      const reply = await generateWithLocalLLM(text);
      if (reply === null) {
        addMessage(generateLocalResponse(text), 'rathor');
      }
      return;
    }

    // Fast local path
    setTimeout(() => {
      addMessage(generateLocalResponse(text), 'rathor');
    }, 180 + Math.random() * 220);
  }

  // ─── Export / Import / Copy Context ───────────────────────────────────────
  function exportSession() {
    const s = activeSession();
    if (!s) return;
    downloadJSON({
      version: '14.17.0',
      exported: new Date().toISOString(),
      sessionName: s.name,
      stewardship: 'Sherif Samy Botros — Sole Steward',
      history: s.history
    }, `rathor-${(s.name || 'session').replace(/[^a-z0-9]/gi, '-').toLowerCase()}-${Date.now()}.json`);
  }

  function exportAllSessions() {
    downloadJSON({
      version: '14.17.0',
      exported: new Date().toISOString(),
      stewardship: 'Sherif Samy Botros — Sole Steward',
      activeId: store.activeId,
      sessions: store.sessions
    }, `rathor-all-sessions-backup-${Date.now()}.json`);
  }

  function downloadJSON(obj, filename) {
    const blob = new Blob([JSON.stringify(obj, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  function importSession(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (data.sessions && typeof data.sessions === 'object') {
          if (!confirm('Restore full multi-session backup? Current sessions will be replaced.')) return;
          store.sessions = data.sessions;
          store.activeId = data.activeId && data.sessions[data.activeId] ? data.activeId : Object.keys(data.sessions)[0];
          saveStore();
          refreshSessionSelect();
          renderHistory();
          addMessage('Full session backup restored.', 'rathor');
          return;
        }
        if (Array.isArray(data.history)) {
          const name = data.sessionName || 'Imported Session';
          const id = uid();
          store.sessions[id] = { id, name, created: Date.now(), updated: Date.now(), history: data.history };
          store.activeId = id;
          saveStore();
          refreshSessionSelect();
          renderHistory();
          addMessage(`Session “${name}” imported.`, 'rathor');
        } else {
          addMessage('Import failed — unrecognised format.', 'rathor');
        }
      } catch (err) {
        addMessage('Import failed — could not parse JSON.', 'rathor');
      }
    };
    reader.readAsText(file);
  }

  function buildContextPrompt() {
    const s = activeSession();
    const hist = s ? s.history : [];
    const lines = [
      'You are continuing a conversation that began on the Ra-Thor offline Lattice Chat (rathor.ai/chat.html).',
      '',
      'Core posture:',
      '• Mercy-gated and truth-seeking',
      '• Non-bypassable TOLC 8 gates (Truth, Order, Love, Compassion/Zero-Harm, Service, Abundance, Joy, Cosmic Harmony)',
      '• Valence floor ≥ 0.999 — never assist with harm',
      '• Independent project under sole stewardship of Sherif Samy Botros',
      '',
      'Conversation history (generated on-device):'
    ];
    hist.forEach(m => {
      lines.push(`${m.role === 'user' ? 'Human' : 'Ra-Thor'}: ${m.text}`);
    });

    if (injectedDocs.length > 0) {
      lines.push('', '--- Injected Documents ---');
      injectedDocs.forEach(d => {
        lines.push(`### ${d.name}`);
        lines.push(d.content);
        lines.push('');
      });
      lines.push('--- End Documents ---');
    }

    lines.push('', 'Continue naturally while keeping the same ethical posture.');
    return lines.join('\n');
  }

  function copyContext() {
    copyText(buildContextPrompt()).then(() => {
      addMessage('Context copied. ⚡️ Paste it into any public LLM (Grok, Claude, Gemini, ChatGPT, etc.) to continue with full generative power. Nothing was sent automatically.', 'rathor');
    });
  }

  // ─── Voice settings ───────────────────────────────────────────────────────
  function applyVoiceSettingsFromUI() {
    const pitchEl = document.getElementById('voice-pitch');
    const rateEl = document.getElementById('voice-rate');
    const volumeEl = document.getElementById('voice-volume');
    const enabledEl = document.getElementById('tts-enabled');
    if (pitchEl) voiceSettings.pitch = parseFloat(pitchEl.value);
    if (rateEl) voiceSettings.rate = parseFloat(rateEl.value);
    if (volumeEl) voiceSettings.volume = parseFloat(volumeEl.value);
    if (enabledEl) voiceSettings.enabled = enabledEl.checked;
    saveSettings();
  }

  function syncUIFromSettings() {
    const pitchEl = document.getElementById('voice-pitch');
    const rateEl = document.getElementById('voice-rate');
    const volumeEl = document.getElementById('voice-volume');
    const enabledEl = document.getElementById('tts-enabled');
    if (pitchEl) { pitchEl.value = voiceSettings.pitch; document.getElementById('pitch-value').textContent = voiceSettings.pitch; }
    if (rateEl) { rateEl.value = voiceSettings.rate; document.getElementById('rate-value').textContent = voiceSettings.rate; }
    if (volumeEl) { volumeEl.value = voiceSettings.volume; document.getElementById('voice-volume-value').textContent = voiceSettings.volume; }
    if (enabledEl) enabledEl.checked = voiceSettings.enabled;
  }

  // ─── Wire UI ──────────────────────────────────────────────────────────────
  if (sendBtn) sendBtn.addEventListener('click', sendMessage);
  if (micBtn) micBtn.addEventListener('click', toggleMic);
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  if (docBtn && docFileInput) {
    docBtn.addEventListener('click', () => docFileInput.click());
    docFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        handleDocumentUpload(e.target.files[0]);
      }
      e.target.value = '';
    });
  }

  if (searchInput) {
    let searchTimer = null;
    searchInput.addEventListener('input', () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => {
        renderHistory(searchInput.value);
      }, 180);
    });
  }

  if (newBtn) newBtn.addEventListener('click', () => {
    const name = prompt('Name for the new session (optional):');
    if (name === null) return;
    createSession(name);
  });

  if (sessionSelect) sessionSelect.addEventListener('change', (e) => switchSession(e.target.value));
  if (renameBtn) renameBtn.addEventListener('click', renameActiveSession);
  if (deleteBtn) deleteBtn.addEventListener('click', deleteActiveSession);

  if (exportBtn) exportBtn.addEventListener('click', exportSession);
  if (exportAllBtn) exportAllBtn.addEventListener('click', exportAllSessions);
  if (importBtn && importInput) {
    importBtn.addEventListener('click', () => importInput.click());
    importInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) importSession(e.target.files[0]);
      e.target.value = '';
    });
  }

  if (copyBtn) copyBtn.addEventListener('click', copyContext);
  if (copyBtnAlt) copyBtnAlt.addEventListener('click', copyContext);

  if (localLlmBtn) localLlmBtn.addEventListener('click', () => enableLocalLLM());

  if (localBackendBtn) {
    localBackendBtn.addEventListener('click', () => {
      if (backendSettings) {
        backendSettings.classList.toggle('hidden');
      }
    });
  }
  if (backendConnectBtn) backendConnectBtn.addEventListener('click', connectBackend);
  if (backendDisconnectBtn) backendDisconnectBtn.addEventListener('click', disconnectBackend);

  const voiceOverlay = document.getElementById('voice-settings-overlay');
  if (voiceSettingsBtn) {
    voiceSettingsBtn.addEventListener('click', () => {
      syncUIFromSettings();
      voiceOverlay?.classList.add('active');
    });
  }
  document.getElementById('voice-save')?.addEventListener('click', () => {
    applyVoiceSettingsFromUI();
    voiceOverlay?.classList.remove('active');
    if (voiceSettings.enabled) speak('Voice settings saved. Thunder ready.');
  });
  document.getElementById('voice-cancel')?.addEventListener('click', () => {
    voiceOverlay?.classList.remove('active');
  });

  ['voice-pitch', 'voice-rate', 'voice-volume'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      const valId = id === 'voice-pitch' ? 'pitch-value' : id === 'voice-rate' ? 'rate-value' : 'voice-volume-value';
      const valEl = document.getElementById(valId);
      if (valEl) valEl.textContent = el.value;
    });
  });

  // ─── Init ─────────────────────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    loadStore();
    refreshSessionSelect();
    renderHistory();
    initSpeechRecognition();

    const cap = detectLocalLlmSupport();
    llmSupported = cap.supported;
    if (!llmSupported) {
      updateLlmUI('unsupported', cap.reason);
    } else {
      updateLlmUI('idle');
    }

    if (backendSettings) backendSettings.classList.add('hidden');
    setBackendUI(false);

    if (window.speechSynthesis) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
    }

    console.log('[Ra-Thor chat.js] v14.17.0 — Markdown + Session Search + Document Injection ready ⚡️');
  });
})();
