/**
 * js/chat.js — Offline TOLC-8 Lattice Chat
 * v14.15.5  •  Next Layer (Richer Intelligence + Local TTS)
 *
 * Architecture:
 *  - Fully offline local responder (zero network calls)
 *  - localStorage session persistence only
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
  const chatMessages = document.getElementById('chat-messages');
  const chatInput    = document.getElementById('chat-input');
  const sendBtn      = document.getElementById('send-btn');
  const newBtn       = document.getElementById('new-session-btn');
  const exportBtn    = document.getElementById('export-session-btn');
  const importBtn    = document.getElementById('import-session-btn');
  const importInput  = document.getElementById('import-file-input');
  const copyBtn      = document.getElementById('copy-context-btn');
  const voiceSettingsBtn = document.getElementById('voice-settings-btn');

  // ─── TOLC 8 Living Mercy Gates ────────────────────────────────────────────
  const TOLC8 = {
    Truth: 0.999,
    Order: 0.999,
    Love: 0.999,
    Compassion: 0.999,   // Zero-Harm
    Service: 0.999,
    Abundance: 0.999,
    Joy: 0.999,
    CosmicHarmony: 0.999
  };

  // ─── Expanded local knowledge (offline only) ──────────────────────────────
  const LOCAL_KNOWLEDGE = [
    { q: /hello|hi|hey|greetings|salam|hola|bonjour|hallo|ciao|namaste/i,
      a: "Thunder locked in, Mate. ⚡️ Offline Mercy Thunder is ready. How may the lattice serve you today?" },

    { q: /who are you|what is ra-?thor|what is rathor|introduce yourself/i,
      a: "I am the offline demo surface of Ra-Thor — a mercy-gated symbolic AGI lattice under sole stewardship of Sherif Samy Botros. All responses stay on your device. No data is collected. No login is required." },

    { q: /tolc|mercy gate|gates|ethics|guardrails/i,
      a: "TOLC 8 Living Mercy Gates are non-bypassable:\n• Truth\n• Order\n• Love\n• Compassion (Zero-Harm)\n• Service\n• Abundance\n• Joy\n• Cosmic Harmony\n\nValence floor ≥ 0.999. These gates cannot be turned off." },

    { q: /privacy|data|track|collect|login|account|cookie|analytics/i,
      a: "Zero personal data leaves your browser. History lives only in localStorage on this device. There is no login, no account, no cookies for tracking, and no analytics. You can Export, Import, or clear at any time." },

    { q: /offline|network|internet|api|server|cloud/i,
      a: "This responder is fully offline-first. No external API calls are made from this surface. The lattice rests in sovereign peace on your device. Even the voice (TTS) uses only the browser’s built-in engine." },

    { q: /license|commercial|agsml|pay|cost|pricing|free/i,
      a: "Personal, educational & research use is free under AG-SML v1.0 — even if you earn modest income as a freelancer. Commercial or revenue-generating use requires a paid license from Autonomicity Games Inc. Contact info@Rathor.ai for fair terms." },

    { q: /powrush|mmo|agsi|demonstration|whitepaper/i,
      a: "Powrush-MMO was completed by one human operator in approximately 30–50 days employing Ra-Thor on Grok engines. This is the AGSi demonstration recorded in WHITEPAPER_v4.1. The full monorepo and whitepaper are available on GitHub." },

    { q: /copy|clipboard|bridge|export|share with|paste into|other llm|claude|gemini|chatgpt/i,
      a: "Use the ‘Copy Context’ button. It places a clean portable prompt + your full conversation onto the clipboard so you can paste it into any public LLM (Grok, Claude, Gemini, ChatGPT, etc.) yourself. Nothing is ever sent automatically from this page." },

    { q: /voice|speak|tts|speech|read aloud|talk/i,
      a: "Voice uses the browser’s built-in Web Speech API only. Everything stays on your device. Open Voice Settings to adjust pitch, rate, and volume. When enabled, replies can be spoken aloud." },

    { q: /session|history|save|export|import|clear|new/i,
      a: "Your conversation lives only in localStorage on this device. You can start a New Session, Export the history as JSON, Import a previous export, or Copy Context to take it elsewhere. You remain in complete control." },

    { q: /steward|owner|who made|creator|sherif|alphapromega/i,
      a: "Ra-Thor is maintained under the sole stewardship of Sherif Samy Botros (@AlphaProMega). It is an independent project. Contact: info@Rathor.ai" },

    { q: /help|commands|what can you|features|how to use/i,
      a: "You can:\n• Chat fully offline\n• Start a New Session\n• Export / Import your history as JSON\n• Copy Context to take the conversation to any public LLM yourself\n• Enable local voice (TTS)\n\nEverything stays under your control. No login required." },

    { q: /thank|thanks|appreciate|grateful/i,
      a: "You’re welcome, Mate. ⚡️ Mercy and truth remain available whenever you return. The lattice is always here on your device." },

    { q: /bye|goodbye|see you|farewell|exit/i,
      a: "Until next time. ⚡️ May the lattice serve you with clarity and care. Your history remains on this device if you wish to continue later." }
  ];

  // ─── Session state (local only) ───────────────────────────────────────────
  const STORAGE_KEY = 'rathor-lattice-chat-v1';
  const SETTINGS_KEY = 'rathor-voice-settings-v1';
  let history = [];   // { role: 'user'|'rathor', text: string, ts: number }

  // Default voice settings
  let voiceSettings = {
    enabled: true,
    pitch: 1.0,
    rate: 1.0,
    volume: 1.0
  };

  function loadSettings() {
    try {
      const raw = localStorage.getItem(SETTINGS_KEY);
      if (raw) voiceSettings = { ...voiceSettings, ...JSON.parse(raw) };
    } catch (e) {}
  }

  function saveSettings() {
    try {
      localStorage.setItem(SETTINGS_KEY, JSON.stringify(voiceSettings));
    } catch (e) {}
  }

  function loadHistory() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        history = JSON.parse(raw);
        if (!Array.isArray(history)) history = [];
      }
    } catch (e) {
      history = [];
    }
  }

  function saveHistory() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history));
    } catch (e) {
      console.warn('[Ra-Thor] localStorage write failed', e);
    }
  }

  function clearHistory() {
    history = [];
    saveHistory();
    if (chatMessages) chatMessages.innerHTML = '';
    addMessage("New session started. ⚡️ Offline Mercy Thunder ready. All processing stays on your device.", 'rathor');
  }

  // ─── Offline TTS (Web Speech API) ─────────────────────────────────────────
  function speak(text) {
    if (!voiceSettings.enabled) return;
    if (!window.speechSynthesis) return;

    // Cancel any ongoing speech
    window.speechSynthesis.cancel();

    const utter = new SpeechSynthesisUtterance(text);
    utter.pitch  = voiceSettings.pitch;
    utter.rate   = voiceSettings.rate;
    utter.volume = voiceSettings.volume;
    utter.lang   = document.documentElement.lang || 'en-US';

    // Prefer a higher-quality voice when available
    const voices = window.speechSynthesis.getVoices();
    const preferred = voices.find(v => v.lang.startsWith('en') && (v.name.includes('Google') || v.name.includes('Natural') || v.name.includes('Premium'))) 
                   || voices.find(v => v.lang.startsWith('en'));
    if (preferred) utter.voice = preferred;

    window.speechSynthesis.speak(utter);
  }

  // ─── UI helpers ───────────────────────────────────────────────────────────
  function addMessage(text, sender = 'rathor', persist = true) {
    if (!chatMessages) return;

    const msgDiv = document.createElement('div');
    msgDiv.classList.add('message', sender);

    const textDiv = document.createElement('div');
    textDiv.classList.add('message-text');
    textDiv.dir = 'auto';
    // Support simple newlines
    textDiv.innerHTML = text.replace(/\n/g, '<br>');

    msgDiv.appendChild(textDiv);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });

    if (persist) {
      history.push({ role: sender, text, ts: Date.now() });
      saveHistory();
    }

    // Speak Rathor replies when TTS is enabled
    if (sender === 'rathor' && voiceSettings.enabled) {
      // Small delay so the bubble appears first
      setTimeout(() => speak(text), 180);
    }
  }

  function renderHistory() {
    if (!chatMessages) return;
    chatMessages.innerHTML = '';
    if (history.length === 0) {
      addMessage("Offline Mercy Thunder ready. ⚡️ TOLC 8 gates active. All processing stays on your device. Ask anything.", 'rathor', false);
      return;
    }
    history.forEach(m => addMessage(m.text, m.role, false));
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

    // Default thoughtful offline response
    return "Thunder received. ⚡️ This is the offline TOLC-8 demo surface. Your words stay on-device.\n\n" +
           "For deeper interaction you can:\n" +
           "• Use Copy Context and paste into any public LLM yourself\n" +
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

    // Mercy-paced local latency
    setTimeout(() => {
      const reply = generateLocalResponse(text);
      addMessage(reply, 'rathor');
    }, 280 + Math.random() * 320);
  }

  // ─── Export / Import ──────────────────────────────────────────────────────
  function exportSession() {
    const payload = {
      version: '14.15.5',
      exported: new Date().toISOString(),
      stewardship: 'Sherif Samy Botros — Sole Steward',
      note: 'This file contains only what you typed and the local offline responses. No external data was collected.',
      history
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const url  = URL.createObjectURL(blob);
    const a    = document.createElement('a');
    a.href     = url;
    a.download = `rathor-lattice-session-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }

  function importSession(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = JSON.parse(e.target.result);
        if (Array.isArray(data.history)) {
          history = data.history;
          saveHistory();
          renderHistory();
          addMessage("Session imported successfully. ⚡️ History restored on this device only.", 'rathor');
        } else {
          addMessage("Import failed — file does not contain a valid history array.", 'rathor');
        }
      } catch (err) {
        addMessage("Import failed — could not parse JSON.", 'rathor');
      }
    };
    reader.readAsText(file);
  }

  // ─── Universal Bridge: Copy Context ───────────────────────────────────────
  function copyContext() {
    const lines = [
      '=== Ra-Thor Offline Lattice Context ===',
      'Source: rathor.ai/chat.html (fully offline surface)',
      'Stewardship: Sherif Samy Botros — Sole Steward',
      'Ethics: TOLC 8 Mercy Gates (Truth, Order, Love, Compassion/Zero-Harm, Service, Abundance, Joy, Cosmic Harmony)',
      'Note: This context was generated entirely on the user\'s device. No data was sent to any server.',
      '',
      'Conversation so far:'
    ];

    history.forEach(m => {
      const who = m.role === 'user' ? 'Human' : 'Ra-Thor (offline)';
      lines.push(`${who}: ${m.text}`);
    });

    lines.push('');
    lines.push('Please continue the conversation with the same mercy-gated, truth-seeking, zero-harm posture.');

    const text = lines.join('\n');

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(() => {
        addMessage("Context copied to clipboard. ⚡️ You can now paste it into any public LLM (Grok, Claude, Gemini, ChatGPT, etc.) yourself. Nothing was sent automatically.", 'rathor');
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

  // ─── Voice Settings UI ────────────────────────────────────────────────────
  function applyVoiceSettingsFromUI() {
    const pitchEl  = document.getElementById('voice-pitch');
    const rateEl   = document.getElementById('voice-rate');
    const volumeEl = document.getElementById('voice-volume');
    const enabledEl = document.getElementById('tts-enabled');

    if (pitchEl)  voiceSettings.pitch  = parseFloat(pitchEl.value);
    if (rateEl)   voiceSettings.rate   = parseFloat(rateEl.value);
    if (volumeEl) voiceSettings.volume = parseFloat(volumeEl.value);
    if (enabledEl) voiceSettings.enabled = enabledEl.checked;

    saveSettings();
  }

  function syncUIFromSettings() {
    const pitchEl  = document.getElementById('voice-pitch');
    const rateEl   = document.getElementById('voice-rate');
    const volumeEl = document.getElementById('voice-volume');
    const enabledEl = document.getElementById('tts-enabled');

    if (pitchEl)  { pitchEl.value = voiceSettings.pitch;  document.getElementById('pitch-value').textContent = voiceSettings.pitch; }
    if (rateEl)   { rateEl.value  = voiceSettings.rate;   document.getElementById('rate-value').textContent  = voiceSettings.rate; }
    if (volumeEl) { volumeEl.value = voiceSettings.volume; document.getElementById('voice-volume-value').textContent = voiceSettings.volume; }
    if (enabledEl) enabledEl.checked = voiceSettings.enabled;
  }

  // ─── Wire UI ──────────────────────────────────────────────────────────────
  if (sendBtn)  sendBtn.addEventListener('click', sendMessage);
  if (chatInput) {
    chatInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }
  if (newBtn) newBtn.addEventListener('click', () => {
    if (history.length && !confirm('Start a new session? Current history will remain exportable until you clear storage.')) return;
    clearHistory();
  });
  if (exportBtn) exportBtn.addEventListener('click', exportSession);
  if (importBtn && importInput) {
    importBtn.addEventListener('click', () => importInput.click());
    importInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) importSession(e.target.files[0]);
      e.target.value = '';
    });
  }
  if (copyBtn) copyBtn.addEventListener('click', copyContext);

  // Voice settings modal wiring
  const voiceOverlay = document.getElementById('voice-settings-overlay');
  const voiceSaveBtn = document.getElementById('voice-save');
  const voiceCancelBtn = document.getElementById('voice-cancel');

  if (voiceSettingsBtn) {
    voiceSettingsBtn.addEventListener('click', () => {
      syncUIFromSettings();
      voiceOverlay?.classList.add('active');
    });
  }
  if (voiceSaveBtn) {
    voiceSaveBtn.addEventListener('click', () => {
      applyVoiceSettingsFromUI();
      voiceOverlay?.classList.remove('active');
      // Quick confirmation speak
      if (voiceSettings.enabled) speak("Voice settings saved. Thunder ready.");
    });
  }
  if (voiceCancelBtn) {
    voiceCancelBtn.addEventListener('click', () => voiceOverlay?.classList.remove('active'));
  }

  // Live slider updates
  ['voice-pitch', 'voice-rate', 'voice-volume'].forEach(id => {
    const el = document.getElementById(id);
    if (!el) return;
    el.addEventListener('input', () => {
      const valId = id === 'voice-pitch' ? 'pitch-value' :
                    id === 'voice-rate'  ? 'rate-value'  : 'voice-volume-value';
      const valEl = document.getElementById(valId);
      if (valEl) valEl.textContent = el.value;
    });
  });

  // ─── Init ─────────────────────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', () => {
    loadSettings();
    loadHistory();
    renderHistory();

    // Chrome needs this to populate voices
    if (window.speechSynthesis) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
    }
  });

  console.log('[Ra-Thor chat.js] Offline TOLC-8 secure lattice + local TTS loaded — zero external calls ⚡️');
})();
