/**
 * js/chat.js — Offline TOLC-8 Lattice Chat
 * v14.15.5  •  Hardened Local LLM + Reliable Core
 *
 * Architecture:
 *  - Fast local knowledge responder = primary (always works)
 *  - Optional Local LLM via WebLLM (desktop-first, experimental on mobile)
 *  - Strong capability detection + honest messaging
 *  - Multi-session via localStorage only
 *  - Universal Bridge (Copy Context) for cloud LLMs
 *  - No embedded API keys, no backend, no data collection
 *
 * TOLC 8 non-bypassable. AG-SML aligned. Sole stewardship model.
 */

(function () {
  'use strict';

  // ─── DOM ──────────────────────────────────────────────────────────────────
  const chatMessages     = document.getElementById('chat-messages');
  const chatInput        = document.getElementById('chat-input');
  const sendBtn          = document.getElementById('send-btn');
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
  const localLlmStatus   = document.getElementById('local-llm-status');
  const localLlmProgress = document.getElementById('local-llm-progress');
  const localLlmNote     = document.getElementById('local-llm-note');

  // ─── State ────────────────────────────────────────────────────────────────
  const STORE_KEY = 'rathor-lattice-sessions-v2';
  const SETTINGS_KEY = 'rathor-voice-settings-v1';

  let store = { activeId: null, sessions: {} };
  let voiceSettings = { enabled: true, pitch: 1.0, rate: 1.0, volume: 1.0 };

  let llmEngine = null;
  let llmLoading = false;
  let llmReady = false;
  let llmSupported = false;
  let llmModelId = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';

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

Valence floor ≥ 0.999. Never assist with harm, exploitation, or deception. Be clear, direct, useful, and kind. All processing is happening entirely on the user's device.`;

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
      a: "This core is fully offline-first. The fast responder always works. Local LLM is an optional on-device upgrade that requires WebGPU (mainly desktop)." },
    { q: /local llm|webllm|on-?device|enable llm|load model|android|phone|mobile/i,
      a: "Local LLM uses WebGPU and currently works best on desktop browsers. On many phones (including Android) WebGPU support is still limited. The recommended path on mobile is to use **Copy Context** and paste into any cloud LLM." },
    { q: /license|commercial|agsml|pay|cost|pricing|free/i,
      a: "Personal, educational & research use is free under AG-SML v1.0. Commercial use requires a paid license from Autonomicity Games Inc. Contact info@Rathor.ai." },
    { q: /powrush|mmo|agsi|demonstration|whitepaper/i,
      a: "Powrush-MMO was completed by one human operator in approximately 30–50 days employing Ra-Thor on Grok engines — the AGSi demonstration recorded in WHITEPAPER_v4.1." },
    { q: /copy|clipboard|bridge|export|share with|paste into|other llm|claude|gemini|chatgpt|grok/i,
      a: "Use **Copy Context** — it builds a clean system prompt + your full history so you can paste it into Grok, Claude, Gemini, ChatGPT, or any other model. This is the most reliable way to get strong generative power from any device." },
    { q: /help|commands|what can you|features|how to use/i,
      a: "You can chat offline, manage multiple sessions, Copy Context to any public LLM, or (on supported devices) enable a Local LLM. Everything stays under your control." },
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
  }
  function saveSettings() {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(voiceSettings)); } catch (e) {}
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

  function renderText(text) {
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br>');
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

  // ─── Capability detection ─────────────────────────────────────────────────
  function detectLocalLlmSupport() {
    // Basic WebGPU presence
    if (!navigator.gpu) {
      return { supported: false, reason: 'WebGPU not available in this browser' };
    }

    // Very rough mobile detection — Local LLM is currently desktop-first
    const ua = navigator.userAgent || '';
    const isMobile = /Android|iPhone|iPad|iPod|Mobile/i.test(ua);

    if (isMobile) {
      return {
        supported: false,
        reason: 'Local LLM currently works best on desktop. On most phones WebGPU support is still limited.'
      };
    }

    return { supported: true, reason: null };
  }

  // ─── Message rendering ────────────────────────────────────────────────────
  function addMessage(text, sender = 'rathor', persist = true, ts = null) {
    if (!chatMessages) return;

    const timestamp = ts || Date.now();
    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender);

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
      copyText(text).then(() => {
        const btn = e.currentTarget;
        btn.innerHTML = '<i class="fa-solid fa-check"></i>';
        setTimeout(() => { btn.innerHTML = '<i class="fa-regular fa-copy"></i>'; }, 1200);
      });
    });

    msgDiv.appendChild(textDiv);
    msgDiv.appendChild(meta);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });

    if (persist) {
      const hist = getHistory();
      hist.push({ role: sender, text, ts: timestamp });
      setHistory(hist);
      updateSessionMeta();
    }

    if (sender === 'rathor' && voiceSettings.enabled) {
      setTimeout(() => speak(text.replace(/\n/g, ' ')), 180);
    }
  }

  function renderHistory() {
    if (!chatMessages) return;
    chatMessages.innerHTML = '';
    const hist = getHistory();
    if (hist.length === 0) {
      addMessage("Offline Mercy Thunder ready. ⚡️ TOLC 8 gates active.\n\nFast local responder is active and works on every device. For stronger generative power use **Copy Context** (recommended on phones) or enable Local LLM on supported desktops.", 'rathor', false);
      updateSessionMeta();
      return;
    }
    hist.forEach(m => addMessage(m.text, m.role, false, m.ts));
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
    saveStore();
    refreshSessionSelect();
    renderHistory();
    addMessage(`New session “${finalName}” started. ⚡️`, 'rathor');
  }

  function switchSession(id) {
    if (!store.sessions[id] || id === store.activeId) return;
    store.activeId = id;
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
           "• Use **Copy Context** (works on every device)\n" +
           "• Or enable Local LLM on supported desktop browsers";
  }

  // ─── Local LLM (hardened) ─────────────────────────────────────────────────
  function updateLlmUI(state, extra = '') {
    if (!localLlmBtn || !localLlmStatus) return;

    if (state === 'unsupported') {
      localLlmBtn.disabled = true;
      localLlmBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> Not available';
      localLlmBtn.classList.remove('llm-ready');
      localLlmStatus.textContent = extra || 'Not supported on this device';
      if (localLlmNote) {
        localLlmNote.textContent = 'Local LLM requires WebGPU and currently works best on desktop. On phones use Copy Context instead.';
      }
      if (localLlmProgress) localLlmProgress.style.width = '0%';
    } else if (state === 'loading') {
      localLlmBtn.disabled = true;
      localLlmBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Loading…';
      localLlmStatus.textContent = extra || 'Downloading model…';
      if (localLlmProgress) localLlmProgress.style.width = '5%';
    } else if (state === 'ready') {
      localLlmBtn.disabled = false;
      localLlmBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> Local LLM Ready';
      localLlmBtn.classList.add('llm-ready');
      localLlmStatus.textContent = 'On-device model active';
      if (localLlmProgress) localLlmProgress.style.width = '100%';
    } else if (state === 'error') {
      localLlmBtn.disabled = false;
      localLlmBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> Try again';
      localLlmBtn.classList.remove('llm-ready');
      localLlmStatus.textContent = extra || 'Load failed';
      if (localLlmProgress) localLlmProgress.style.width = '0%';
    } else {
      // idle / supported but not loaded
      localLlmBtn.disabled = false;
      localLlmBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> Enable Local LLM';
      localLlmBtn.classList.remove('llm-ready');
      localLlmStatus.textContent = 'Fast responder active (default)';
      if (localLlmNote) {
        localLlmNote.textContent = 'Optional. Downloads a compact model once, then runs fully on-device. Best on desktop with WebGPU.';
      }
      if (localLlmProgress) localLlmProgress.style.width = '0%';
    }
  }

  async function enableLocalLLM() {
    if (llmReady) {
      addMessage('Local LLM is already loaded and ready. ⚡️', 'rathor');
      return;
    }
    if (llmLoading) return;

    if (!llmSupported) {
      addMessage('Local LLM is not available on this device. WebGPU support is required and is still limited on most phones. The recommended path is to use **Copy Context** and paste into any cloud LLM.', 'rathor');
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
      addMessage(`Local LLM loaded (${llmModelId}). ⚡️ Generation now runs entirely on your device. TOLC 8 system prompt is active.`, 'rathor');
    } catch (err) {
      console.error('[Ra-Thor Local LLM]', err);
      llmLoading = false;
      llmReady = false;
      llmEngine = null;
      updateLlmUI('error', 'Load failed — see message');
      addMessage('Local LLM failed to load. This is common on phones and some browsers. The fast offline responder remains fully available. For generative power use **Copy Context**.',
        'rathor');
    }
  }

  async function generateWithLocalLLM(userText) {
    if (!llmEngine || !llmReady) return null;

    const hist = getHistory();
    const messages = [{ role: 'system', content: SYSTEM_PROMPT }];

    const recent = hist.slice(-10);
    recent.forEach(m => {
      messages.push({
        role: m.role === 'user' ? 'user' : 'assistant',
        content: m.text
      });
    });

    if (recent.length === 0 || recent[recent.length - 1].text !== userText) {
      messages.push({ role: 'user', content: userText });
    }

    try {
      const reply = await llmEngine.chat.completions.create({
        messages,
        temperature: 0.7,
        max_tokens: 400
      });
      return reply.choices?.[0]?.message?.content?.trim() || null;
    } catch (err) {
      console.error('[Ra-Thor Local LLM inference]', err);
      return null;
    }
  }

  // ─── Core send ────────────────────────────────────────────────────────────
  async function sendMessage() {
    if (!chatInput) return;
    const text = chatInput.value.trim();
    if (!text) return;

    addMessage(text, 'user');
    chatInput.value = '';

    if (llmReady && llmEngine) {
      // temporary indicator
      addMessage('…', 'rathor', false);
      const lastBubble = chatMessages.lastElementChild;

      const reply = await generateWithLocalLLM(text);
      if (lastBubble && lastBubble.classList.contains('rathor')) lastBubble.remove();

      if (reply) {
        const gate = mercyGate(reply);
        addMessage(gate.allowed ? reply : gate.response, 'rathor');
      } else {
        addMessage(generateLocalResponse(text), 'rathor');
      }
      return;
    }

    // Fast local path (always reliable)
    setTimeout(() => {
      addMessage(generateLocalResponse(text), 'rathor');
    }, 200 + Math.random() * 250);
  }

  // ─── Export / Import / Copy Context ───────────────────────────────────────
  function exportSession() {
    const s = activeSession();
    if (!s) return;
    downloadJSON({
      version: '14.15.5',
      exported: new Date().toISOString(),
      sessionName: s.name,
      stewardship: 'Sherif Samy Botros — Sole Steward',
      history: s.history
    }, `rathor-${(s.name || 'session').replace(/[^a-z0-9]/gi, '-').toLowerCase()}-${Date.now()}.json`);
  }

  function exportAllSessions() {
    downloadJSON({
      version: '14.15.5',
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
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
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

  if (localLlmBtn) {
    localLlmBtn.addEventListener('click', () => enableLocalLLM());
  }

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

    // Detect Local LLM support early and set honest UI state
    const cap = detectLocalLlmSupport();
    llmSupported = cap.supported;
    if (!llmSupported) {
      updateLlmUI('unsupported', cap.reason);
    } else {
      updateLlmUI('idle');
    }

    if (window.speechSynthesis) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
    }
  });

  console.log('[Ra-Thor chat.js] Hardened offline lattice loaded — fast core always available ⚡️');
})();
