/**
 * js/chat.js — Offline TOLC-8 Lattice Chat
 * v14.15.5  •  Multi-Session Manager Layer
 *
 * Architecture:
 *  - Fully offline local responder (zero network calls)
 *  - Multi-session support via localStorage only
 *  - Export / Import JSON (user owns data)
 *  - Copy Context → clipboard (universal bridge to any public LLM)
 *  - Official Grok / X bridges remain explicit new-tab only
 *  - Real offline TTS via Web Speech API (browser-native)
 *  - No login • No tracking • No API keys • No data collection
 *
 * TOLC 8 non-bypassable. AG-SML aligned. Sole stewardship model.
 */

(function () {
  'use strict';

  // ─── DOM ──────────────────────────────────────────────────────────────────
  const chatMessages   = document.getElementById('chat-messages');
  const chatInput      = document.getElementById('chat-input');
  const sendBtn        = document.getElementById('send-btn');
  const newBtn         = document.getElementById('new-session-btn');
  const exportBtn      = document.getElementById('export-session-btn');
  const importBtn      = document.getElementById('import-session-btn');
  const importInput    = document.getElementById('import-file-input');
  const copyBtn        = document.getElementById('copy-context-btn');
  const voiceSettingsBtn = document.getElementById('voice-settings-btn');
  const sessionSelect  = document.getElementById('session-select');
  const renameBtn      = document.getElementById('rename-session-btn');
  const deleteBtn      = document.getElementById('delete-session-btn');

  // ─── TOLC 8 ───────────────────────────────────────────────────────────────
  const TOLC8 = {
    Truth: 0.999, Order: 0.999, Love: 0.999, Compassion: 0.999,
    Service: 0.999, Abundance: 0.999, Joy: 0.999, CosmicHarmony: 0.999
  };

  // ─── Expanded local knowledge ─────────────────────────────────────────────
  const LOCAL_KNOWLEDGE = [
    { q: /hello|hi|hey|greetings|salam|hola|bonjour|hallo|ciao|namaste/i,
      a: "Thunder locked in, Mate. ⚡️ Offline Mercy Thunder is ready. How may the lattice serve you today?" },
    { q: /who are you|what is ra-?thor|what is rathor|introduce yourself/i,
      a: "I am the offline demo surface of Ra-Thor — a mercy-gated symbolic AGI lattice under sole stewardship of Sherif Samy Botros. All responses stay on your device. No data is collected. No login is required." },
    { q: /tolc|mercy gate|gates|ethics|guardrails/i,
      a: "TOLC 8 Living Mercy Gates are non-bypassable:\n• Truth\n• Order\n• Love\n• Compassion (Zero-Harm)\n• Service\n• Abundance\n• Joy\n• Cosmic Harmony\n\nValence floor ≥ 0.999. These gates cannot be turned off." },
    { q: /privacy|data|track|collect|login|account|cookie|analytics/i,
      a: "Zero personal data leaves your browser. All sessions live only in localStorage on this device. There is no login, no account, no tracking. You can Export, Import, or delete sessions at any time." },
    { q: /offline|network|internet|api|server|cloud/i,
      a: "This responder is fully offline-first. No external API calls are made. Even the voice (TTS) uses only the browser’s built-in engine. The lattice rests in sovereign peace on your device." },
    { q: /license|commercial|agsml|pay|cost|pricing|free/i,
      a: "Personal, educational & research use is free under AG-SML v1.0 — even if you earn modest income as a freelancer. Commercial or revenue-generating use requires a paid license from Autonomicity Games Inc. Contact info@Rathor.ai." },
    { q: /powrush|mmo|agsi|demonstration|whitepaper/i,
      a: "Powrush-MMO was completed by one human operator in approximately 30–50 days employing Ra-Thor on Grok engines. This is the AGSi demonstration recorded in WHITEPAPER_v4.1." },
    { q: /copy|clipboard|bridge|export|share with|paste into|other llm|claude|gemini|chatgpt/i,
      a: "Use the ‘Copy Context’ button. It places a clean portable prompt + your full conversation onto the clipboard so you can paste it into any public LLM yourself. Nothing is ever sent automatically." },
    { q: /voice|speak|tts|speech|read aloud|talk/i,
      a: "Voice uses the browser’s built-in Web Speech API only. Everything stays on your device. Open Voice Settings to adjust pitch, rate, and volume." },
    { q: /session|history|save|export|import|clear|new|switch|multiple/i,
      a: "You can create multiple named sessions, switch between them, rename, or delete them. Everything stays in localStorage on this device. Export / Import work on the currently active session." },
    { q: /steward|owner|who made|creator|sherif|alphapromega/i,
      a: "Ra-Thor is maintained under the sole stewardship of Sherif Samy Botros (@AlphaProMega). Independent project. Contact: info@Rathor.ai" },
    { q: /help|commands|what can you|features|how to use/i,
      a: "You can:\n• Chat fully offline\n• Create & switch multiple sessions\n• Export / Import history as JSON\n• Copy Context to any public LLM\n• Enable local voice (TTS)\n\nEverything stays under your control. No login required." },
    { q: /thank|thanks|appreciate|grateful/i,
      a: "You’re welcome, Mate. ⚡️ Mercy and truth remain available whenever you return." },
    { q: /bye|goodbye|see you|farewell|exit/i,
      a: "Until next time. ⚡️ May the lattice serve you with clarity and care. Your sessions remain on this device." }
  ];

  // ─── Multi-Session Store (localStorage only) ──────────────────────────────
  const STORE_KEY = 'rathor-lattice-sessions-v2';
  const SETTINGS_KEY = 'rathor-voice-settings-v1';

  let store = {
    activeId: null,
    sessions: {}   // id → { id, name, created, updated, history: [] }
  };

  let voiceSettings = { enabled: true, pitch: 1.0, rate: 1.0, volume: 1.0 };

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

    // Migrate old single-history format if present
    try {
      const old = localStorage.getItem('rathor-lattice-chat-v1');
      if (old && Object.keys(store.sessions).length === 0) {
        const hist = JSON.parse(old);
        if (Array.isArray(hist) && hist.length) {
          const id = uid();
          store.sessions[id] = {
            id, name: 'Imported Session', created: Date.now(), updated: Date.now(), history: hist
          };
          store.activeId = id;
          localStorage.removeItem('rathor-lattice-chat-v1');
          saveStore();
        }
      }
    } catch (e) {}

    // Ensure at least one session exists
    if (!store.activeId || !store.sessions[store.activeId]) {
      const id = uid();
      store.sessions[id] = {
        id, name: 'Session 1', created: Date.now(), updated: Date.now(), history: []
      };
      store.activeId = id;
      saveStore();
    }
  }

  function saveStore() {
    try {
      localStorage.setItem(STORE_KEY, JSON.stringify(store));
    } catch (e) {
      console.warn('[Ra-Thor] localStorage write failed', e);
    }
  }

  function activeSession() {
    return store.sessions[store.activeId];
  }

  function getHistory() {
    return activeSession()?.history || [];
  }

  function setHistory(hist) {
    const s = activeSession();
    if (!s) return;
    s.history = hist;
    s.updated = Date.now();
    saveStore();
  }

  // ─── Voice settings ───────────────────────────────────────────────────────
  function loadSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (raw) voiceSettings = { ...voiceSettings, ...JSON.parse(raw) };
    } catch (e) {}
  }
  function saveSettings() {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(voiceSettings)); } catch (e) {}
  }

  // ─── Offline TTS ──────────────────────────────────────────────────────────
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

  // ─── Light markdown ───────────────────────────────────────────────────────
  function renderText(text) {
    return text
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/\n/g, '<br>');
  }

  // ─── UI helpers ───────────────────────────────────────────────────────────
  function addMessage(text, sender = 'rathor', persist = true) {
    if (!chatMessages) return;

    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender);

    const textDiv = document.createElement('div');
    textDiv.classList.add('message-text');
    textDiv.dir = 'auto';
    textDiv.innerHTML = renderText(text);

    msgDiv.appendChild(textDiv);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });

    if (persist) {
      const hist = getHistory();
      hist.push({ role: sender, text, ts: Date.now() });
      setHistory(hist);
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
      addMessage("Offline Mercy Thunder ready. ⚡️ TOLC 8 gates active. All processing stays on your device. Ask anything.", 'rathor', false);
      return;
    }
    hist.forEach(m => addMessage(m.text, m.role, false));
  }

  function refreshSessionSelect() {
    if (!sessionSelect) return;
    sessionSelect.innerHTML = '';
    const ids = Object.keys(store.sessions).sort((a, b) =>
      (store.sessions[b].updated || 0) - (store.sessions[a].updated || 0)
    );
    ids.forEach(id => {
      const s = store.sessions[id];
      const opt = document.createElement('option');
      opt.value = id;
      opt.textContent = s.name || 'Untitled';
      if (id === store.activeId) opt.selected = true;
      sessionSelect.appendChild(opt);
    });
  }

  // ─── Session operations ───────────────────────────────────────────────────
  function createSession(name) {
    const id = uid();
    const finalName = (name || '').trim() || `Session ${Object.keys(store.sessions).length + 1}`;
    store.sessions[id] = {
      id, name: finalName, created: Date.now(), updated: Date.now(), history: []
    };
    store.activeId = id;
    saveStore();
    refreshSessionSelect();
    renderHistory();
    addMessage(`New session “${finalName}” started. ⚡️ Offline Mercy Thunder ready.`, 'rathor');
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
    const name = next.trim() || s.name;
    s.name = name;
    s.updated = Date.now();
    saveStore();
    refreshSessionSelect();
  }

  function deleteActiveSession() {
    const s = activeSession();
    if (!s) return;
    if (!confirm(`Delete session “${s.name}”? This cannot be undone (unless you exported it).`)) return;

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

  // ─── Mercy Gate ───────────────────────────────────────────────────────────
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

    return "Thunder received. ⚡️ This is the offline TOLC-8 demo surface. Your words stay on-device.\n\n" +
           "For deeper interaction you can:\n" +
           "• Use **Copy Context** and paste into any public LLM yourself\n" +
           "• Open the official Grok or X demos\n\n" +
           "How else may mercy assist?";
  }

  // ─── Core send ────────────────────────────────────────────────────────────
  function sendMessage() {
    if (!chatInput) return;
    const text = chatInput.value.trim();
    if (!text) return;

    addMessage(text, 'user');
    chatInput.value = '';

    setTimeout(() => {
      const reply = generateLocalResponse(text);
      addMessage(reply, 'rathor');
    }, 260 + Math.random() * 300);
  }

  // ─── Export / Import (current session) ────────────────────────────────────
  function exportSession() {
    const s = activeSession();
    if (!s) return;
    const payload = {
      version: '14.15.5',
      exported: new Date().toISOString(),
      sessionName: s.name,
      stewardship: 'Sherif Samy Botros — Sole Steward',
      note: 'This file contains only what you typed and the local offline responses. No external data was collected.',
      history: s.history
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `rathor-${(s.name || 'session').replace(/[^a-z0-9]/gi, '-').toLowerCase()}-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function importSession(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (Array.isArray(data.history)) {
          const name = data.sessionName || 'Imported Session';
          const id = uid();
          store.sessions[id] = {
            id, name, created: Date.now(), updated: Date.now(), history: data.history
          };
          store.activeId = id;
          saveStore();
          refreshSessionSelect();
          renderHistory();
          addMessage(`Session “${name}” imported successfully. ⚡️ History restored on this device only.`, 'rathor');
        } else {
          addMessage("Import failed — file does not contain a valid history array.", 'rathor');
        }
      } catch (err) {
        addMessage("Import failed — could not parse JSON.", 'rathor');
      }
    };
    reader.readAsText(file);
  }

  // ─── Copy Context ─────────────────────────────────────────────────────────
  function copyContext() {
    const s = activeSession();
    const hist = s ? s.history : [];
    const lines = [
      '=== Ra-Thor Offline Lattice Context ===',
      'Source: rathor.ai/chat.html (fully offline surface)',
      'Stewardship: Sherif Samy Botros — Sole Steward',
      'Ethics: TOLC 8 Mercy Gates (Truth, Order, Love, Compassion/Zero-Harm, Service, Abundance, Joy, Cosmic Harmony)',
      'Note: This context was generated entirely on the user\'s device. No data was sent to any server.',
      `Session: ${s ? s.name : 'Unknown'}`,
      '',
      'Conversation so far:'
    ];
    hist.forEach(m => {
      const who = m.role === 'user' ? 'Human' : 'Ra-Thor (offline)';
      lines.push(`${who}: ${m.text}`);
    });
    lines.push('');
    lines.push('Please continue the conversation with the same mercy-gated, truth-seeking, zero-harm posture.');

    const text = lines.join('\n');
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => {
        addMessage("Context copied to clipboard. ⚡️ You can now paste it into any public LLM yourself. Nothing was sent automatically.", 'rathor');
      }).catch(() => fallbackCopy(text));
    } else {
      fallbackCopy(text);
    }
  }

  function fallbackCopy(text) {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try {
      document.execCommand('copy');
      addMessage("Context copied to clipboard (fallback). ⚡️ Paste it into any public LLM yourself.", 'rathor');
    } catch (e) {
      addMessage("Could not copy automatically. Please select and copy the conversation manually.", 'rathor');
    }
    document.body.removeChild(ta);
  }

  // ─── Voice Settings ───────────────────────────────────────────────────────
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
    if (name === null) return; // cancelled
    createSession(name);
  });

  if (sessionSelect) {
    sessionSelect.addEventListener('change', (e) => switchSession(e.target.value));
  }
  if (renameBtn) renameBtn.addEventListener('click', renameActiveSession);
  if (deleteBtn) deleteBtn.addEventListener('click', deleteActiveSession);

  if (exportBtn) exportBtn.addEventListener('click', exportSession);
  if (importBtn && importInput) {
    importBtn.addEventListener('click', () => importInput.click());
    importInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) importSession(e.target.files[0]);
      e.target.value = '';
    });
  }
  if (copyBtn) copyBtn.addEventListener('click', copyContext);

  // Voice modal
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
    if (voiceSettings.enabled) speak("Voice settings saved. Thunder ready.");
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

    if (window.speechSynthesis) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
    }
  });

  console.log('[Ra-Thor chat.js] Multi-session offline lattice + local TTS loaded — zero external calls ⚡️');
})();
