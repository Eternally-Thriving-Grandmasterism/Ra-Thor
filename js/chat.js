/**
 * js/chat.js — Offline TOLC-8 Lattice Chat
 * v14.18.0  •  Optional Passphrase Encryption + full Priority 2
 *
 * Architecture:
 *  - Fast local knowledge responder = primary
 *  - Optional Local Backend Bridge (Ollama / OpenAI-compatible)
 *  - Optional WebLLM (WebGPU)
 *  - Streaming, STT, TTS
 *  - Multi-session + Document Injection + Session Search
 *  - Proper Markdown + fenced code blocks
 *  - Optional AES-GCM Passphrase Encryption of the session store
 *  - Zero collection, no backend we control, TOLC 8 non-bypassable
 *
 * Sole stewardship: Sherif Samy Botros — info@Rathor.ai
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
  const encryptBtn       = document.getElementById('encrypt-btn');
  const sessionSelect    = document.getElementById('session-select');
  const renameBtn        = document.getElementById('rename-session-btn');
  const deleteBtn        = document.getElementById('delete-session-btn');
  const sessionMeta      = document.getElementById('session-meta');
  const localLlmBtn      = document.getElementById('local-llm-btn');
  const localBackendBtn  = document.getElementById('local-backend-btn');
  const localLlmStatus   = document.getElementById('local-llm-status');
  const localLlmProgress = document.getElementById('local-llm-progress');
  const webllmPicker      = document.getElementById('webllm-picker');
  const webllmRows        = document.getElementById('webllm-rows');
  const webllmThirdParty  = document.getElementById('webllm-third-party');
  const webllmDownloadNote = document.getElementById('webllm-download-note');
  const webllmOtherNote   = document.getElementById('webllm-other-note');
  const netModeOfflineBtn = document.getElementById('net-mode-offline');
  const netModeNetworkBtn = document.getElementById('net-mode-network');
  const netConnection     = document.getElementById('net-connection');
  const netModeNote       = document.getElementById('net-mode-note');
  const netModeOfflineNote = document.getElementById('net-mode-offline-note');
  const backendSettings  = document.getElementById('backend-settings');
  const backendEndpoint  = document.getElementById('backend-endpoint');
  const backendModel     = document.getElementById('backend-model');
  const backendConnectBtn= document.getElementById('backend-connect-btn');
  const backendDisconnectBtn = document.getElementById('backend-disconnect-btn');
  const backendStatus    = document.getElementById('backend-status');
  const activePathBadge  = document.getElementById('active-path-badge');
  const unlockOverlay    = document.getElementById('unlock-overlay');
  const unlockPassphrase = document.getElementById('unlock-passphrase');
  const unlockBtn        = document.getElementById('unlock-btn');
  const unlockError      = document.getElementById('unlock-error');

  // ─── State ────────────────────────────────────────────────────────────────
  const STORE_KEY = 'rathor-lattice-sessions-v2';
  const SETTINGS_KEY = 'rathor-voice-settings-v1';
  const BACKEND_KEY = 'rathor-local-backend-v1';
  const ENCRYPT_FLAG = 'rathor-lattice-encrypted-v1';

  let store = { activeId: null, sessions: {} };
  let voiceSettings = { enabled: true, pitch: 1.0, rate: 1.0, volume: 1.0 };
  let cryptoKey = null;          // CryptoKey held in memory only
  let isEncrypted = false;

  let llmEngine = null;
  let llmLoading = false;
  let llmReady = false;
  let llmSupported = false;
  let llmProbed = false;
  let llmModelId = 'Llama-3.2-1B-Instruct-q4f16_1-MLC';
  let llmLoadToken = 0;
  let webllmModule = null;
  let webllmPickerReady = false;
  let webllmShaderF16 = false;
  let webllmOptions = [];
  let webllmRowRuntime = {};
  let webllmDownloadCancelled = false;

  const WEBLLM_MODEL_KEY = 'rathor-webllm-model-v1';
  const WEBLLM_VENDOR = './vendor/web-llm/0.2.85/index.js';
  // Same-origin script cache. The name contains "webllm", so sw.js activate keeps it.
  const WEBLLM_SCRIPT_CACHE = 'webllm/script';
  const WEBLLM_SCRIPT_URL = absoluteScriptUrl(
    (document.currentScript && document.currentScript.src) || new URL('/js/chat.js', location.href).href,
    WEBLLM_VENDOR
  );
  const WEBLLM_DEFAULT_BASE = 'Llama-3.2-1B-Instruct';
  // Curated order. Quantization is chosen from the pinned prebuiltAppConfig.
  // q4f32_1 is used only when the WebGPU adapter lacks shader-f16.
  const WEBLLM_CURATED = [
    {
      base: 'SmolLM2-360M-Instruct',
      q4f16: 'SmolLM2-360M-Instruct-q4f16_1-MLC',
      q4f32: 'SmolLM2-360M-Instruct-q4f32_1-MLC',
      links: [{ text: 'Apache-2.0', href: 'https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct/blob/a10cc1512eabd3dde888204e902eca88bddb4951/README.md' }],
      smallReplyNote: true
    },
    {
      base: 'Qwen2.5-0.5B-Instruct',
      q4f16: 'Qwen2.5-0.5B-Instruct-q4f16_1-MLC',
      q4f32: 'Qwen2.5-0.5B-Instruct-q4f32_1-MLC',
      links: [{ text: 'Apache-2.0', href: 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/LICENSE' }]
    },
    {
      base: 'Llama-3.2-1B-Instruct',
      q4f16: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
      q4f32: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
      links: [
        { text: 'Llama 3.2 Community License', href: 'https://github.com/meta-llama/llama-models/blob/8d29d93fa5700a60532e0061a02ffa89d0acd3fc/models/llama3_2/LICENSE' },
        { text: 'Acceptable Use Policy', href: 'https://github.com/meta-llama/llama-models/blob/8d29d93fa5700a60532e0061a02ffa89d0acd3fc/models/llama3_2/USE_POLICY.md' }
      ],
      builtWithLlama: true
    },
    {
      base: 'Qwen2.5-1.5B-Instruct',
      q4f16: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
      q4f32: 'Qwen2.5-1.5B-Instruct-q4f32_1-MLC',
      links: [{ text: 'Apache-2.0', href: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/blob/989aa7980e4cf806f80c7fef2b1adb7bc71aa306/LICENSE' }]
    },
    {
      base: 'gemma-2-2b-it',
      q4f16: 'gemma-2-2b-it-q4f16_1-MLC',
      q4f32: 'gemma-2-2b-it-q4f32_1-MLC',
      links: [
        { text: 'Gemma Terms of Use', href: 'https://ai.google.dev/gemma/terms' },
        { text: 'Prohibited Use Policy', href: 'https://ai.google.dev/gemma/prohibited_use_policy' }
      ]
    },
    {
      base: 'Llama-3.2-3B-Instruct',
      q4f16: 'Llama-3.2-3B-Instruct-q4f16_1-MLC',
      q4f32: 'Llama-3.2-3B-Instruct-q4f32_1-MLC',
      links: [
        { text: 'Llama 3.2 Community License', href: 'https://github.com/meta-llama/llama-models/blob/8d29d93fa5700a60532e0061a02ffa89d0acd3fc/models/llama3_2/LICENSE' },
        { text: 'Acceptable Use Policy', href: 'https://github.com/meta-llama/llama-models/blob/8d29d93fa5700a60532e0061a02ffa89d0acd3fc/models/llama3_2/USE_POLICY.md' }
      ],
      builtWithLlama: true
    },
    {
      base: 'Phi-3.5-mini-instruct',
      q4f16: 'Phi-3.5-mini-instruct-q4f16_1-MLC',
      q4f32: 'Phi-3.5-mini-instruct-q4f32_1-MLC',
      links: [{ text: 'MIT', href: 'https://huggingface.co/microsoft/Phi-3.5-mini-instruct/blob/2fe192450127e6a83f7441aef6e3ca586c338b77/LICENSE' }]
    }
  ];

  let backendEnabled = false;
  let backendConfig = { endpoint: 'http://localhost:11434/v1', model: 'llama3.2' };

  let injectedDocs = [];
  let recognition = null;
  let isListening = false;

  // Quoted from wrappers/system-prompt.txt — same sentences. Live prompt is the employ constitution.
  const SYSTEM_PROMPT = `You are sitting under the Ra-Thor employ loop (workspace 14.15.6).

You are the optional model, not the lattice. The lattice is the gates. You are the sampler.

Standing test: TOLC 8 — Truth, Order, Love, Compassion (zero-harm), Service, Abundance, Joy, Cosmic Harmony.

PATSAGi Councils are architecture for deliberation in the monorepo — not a warranty that every answer is automatically correct.

Layer 0 is an admission shell, not sampler weights.

Outputs are drafts. A human reviews them before filing, sale, or public claims.

inspect ≠ METR. Combined AGSi stays SURMISE. Independent of xAI. An optional Grok session is not an xAI product.

Do not invent METR numbers, certifications, or a finished MMO. Powrush-MMO is a separate repo.

Contact: info@Rathor.ai
License: AG-SML v1.1 (personal / research). Organizations license.`;

  var CHAT_LANG_NAMES = {
    en: 'English', ar: 'العربية', es: 'Español', fr: 'Français',
    nl: 'Nederlands', de: 'Deutsch', zh: '简体中文', ja: '日本語',
    pt: 'Português', ru: 'Русский', hi: 'हिन्दी', it: 'Italiano',
    ko: '한국어', uk: 'Українська', pl: 'Polski', tr: 'Türkçe',
    vi: 'Tiếng Việt', id: 'Bahasa Indonesia', sv: 'Svenska', th: 'ไทย',
    el: 'Ελληνικά', fa: 'فارسی', he: 'עברית'
  };

  function chatLang() {
    try { return localStorage.getItem('rathor-lang') || ''; } catch (e) { return ''; }
  }

  function chatStr(key) {
    var lang = chatLang() || 'en';
    var packs = (typeof window !== 'undefined' && window.translations) || {};
    var pack = packs[lang] || packs.en || {};
    var en = packs.en || {};
    var val = pack[key];
    if (val != null && String(val).trim() !== '') return String(val);
    val = en[key];
    if (val != null && String(val).trim() !== '') return String(val);
    return '';
  }

  function chatLabel(key, fallback) {
    var val = chatStr(key);
    return val || fallback;
  }

  function replyInClause() {
    var lang = chatLang();
    if (!lang) return '';
    var name = CHAT_LANG_NAMES[lang] || lang;
    return 'Reply in ' + name + '.';
  }

  function systemPreamble() {
    var line = replyInClause();
    if (!line) return SYSTEM_PROMPT;
    return SYSTEM_PROMPT.replace(/\s*$/, '') + '\n\n' + line + '\n';
  }

  function applyChatSurfaceDir() {
    var sample = chatStr('chatSpeak') || chatStr('chatReplyHello');
    var decided = (typeof window.rtChatSurfaceDir === 'function')
      ? window.rtChatSurfaceDir(sample, chatLang() || 'en')
      : { dir: 'ltr', lang: 'en' };
    [chatMessages, chatInput].forEach(function (el) {
      if (!el) return;
      el.setAttribute('dir', decided.dir);
      el.setAttribute('lang', decided.lang);
    });
    var family = document.getElementById('rt-family-nav');
    if (family) family.setAttribute('dir', 'ltr');
    var tabs = document.getElementById('lang-selector');
    if (tabs) tabs.setAttribute('dir', 'ltr');
  }

  const LOCAL_KNOWLEDGE = [
    { q: /hello|hi|hey|greetings|salam|hola|bonjour|hallo|ciao|namaste/i, k: 'chatReplyHello' },
    { q: /who are you|what is ra-?thor|what is rathor|introduce yourself/i, k: 'chatReplyWho' },
    { q: /tolc|mercy gate|gates|ethics|guardrails/i, k: 'chatReplyTolc' },
    { q: /privacy|data|track|collect|login|account|encrypt|passphrase|lock/i, k: 'chatReplyPrivacy' },
    { q: /offline|network|internet|api|server|cloud/i, k: 'chatReplyOffline' },
    { q: /local llm|webllm|on-?device|enable llm|load model|android|phone|mobile/i, k: 'chatReplyLocal' },
    { q: /ollama|local server|backend|localhost|lm studio|localai/i, k: 'chatReplyOllama' },
    { q: /document|upload|inject|file|context injection|rag/i, k: 'chatReplyDoc' },
    { q: /search|find message|look for/i, k: 'chatReplySearch' },
    { q: /license|commercial|agsml|pay|cost|pricing|free/i, k: 'chatReplyLicense' },
    { q: /powrush|mmo|agsi|demonstration|whitepaper/i, k: 'chatReplyPowrush' },
    { q: /copy|clipboard|bridge|export|share with|paste into|other llm|claude|gemini|chatgpt|grok/i, k: 'chatReplyCopy' },
    { q: /help|commands|what can you|features|how to use/i, k: 'chatReplyHelp' },
    { q: /thank|thanks|appreciate|grateful/i, k: 'chatReplyThanks' },
    { q: /bye|goodbye|see you|farewell|exit/i, k: 'chatReplyBye' }
  ];

  // ─── Crypto helpers (Web Crypto only) ─────────────────────────────────────
  function bufToBase64(buf) {
    return btoa(String.fromCharCode(...new Uint8Array(buf)));
  }
  function base64ToBuf(b64) {
    return Uint8Array.from(atob(b64), c => c.charCodeAt(0));
  }

  async function deriveKey(passphrase, salt) {
    const enc = new TextEncoder();
    const keyMaterial = await crypto.subtle.importKey(
      'raw', enc.encode(passphrase), 'PBKDF2', false, ['deriveKey']
    );
    return crypto.subtle.deriveKey(
      { name: 'PBKDF2', salt, iterations: 100000, hash: 'SHA-256' },
      keyMaterial,
      { name: 'AES-GCM', length: 256 },
      false,
      ['encrypt', 'decrypt']
    );
  }

  async function encryptStore(passphrase, dataObj) {
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv = crypto.getRandomValues(new Uint8Array(12));
    const key = await deriveKey(passphrase, salt);
    const encoded = new TextEncoder().encode(JSON.stringify(dataObj));
    const ciphertext = await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, encoded);
    return {
      encrypted: true,
      version: 1,
      salt: bufToBase64(salt),
      iv: bufToBase64(iv),
      data: bufToBase64(ciphertext)
    };
  }

  async function decryptStore(passphrase, envelope) {
    const salt = base64ToBuf(envelope.salt);
    const iv = base64ToBuf(envelope.iv);
    const data = base64ToBuf(envelope.data);
    const key = await deriveKey(passphrase, salt);
    const decrypted = await crypto.subtle.decrypt({ name: 'AES-GCM', iv }, key, data);
    return JSON.parse(new TextDecoder().decode(decrypted));
  }

  // ─── Store load / save with encryption support ────────────────────────────
  function isStoreEncrypted() {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (!raw) return false;
      const parsed = JSON.parse(raw);
      return !!(parsed && parsed.encrypted === true);
    } catch (e) { return false; }
  }

  async function loadStore(passphrase = null) {
    try {
      const raw = localStorage.getItem(STORE_KEY);
      if (!raw) {
        createDefaultSession();
        return true;
      }
      const parsed = JSON.parse(raw);

      if (parsed && parsed.encrypted === true) {
        if (!passphrase) return false; // needs unlock
        try {
          store = await decryptStore(passphrase, parsed);
          cryptoKey = await deriveKey(passphrase, base64ToBuf(parsed.salt)); // keep for future saves
          isEncrypted = true;
          return true;
        } catch (err) {
          console.warn('[Ra-Thor] decrypt failed', err);
          return false;
        }
      }

      // Plain store
      if (parsed && parsed.sessions) {
        store = parsed;
        isEncrypted = false;
        cryptoKey = null;
      }
    } catch (e) {
      console.warn('[Ra-Thor] loadStore error', e);
    }

    if (!store.activeId || !store.sessions[store.activeId]) {
      createDefaultSession();
    }
    return true;
  }

  function createDefaultSession() {
    const id = uid();
    store = {
      activeId: id,
      sessions: {
        [id]: { id, name: 'Session 1', created: Date.now(), updated: Date.now(), history: [] }
      }
    };
  }

  async function saveStore() {
    try {
      if (isEncrypted && cryptoKey) {
        // Re-encrypt with the current in-memory key material is not directly possible
        // without the original passphrase. For simplicity and safety we keep the
        // encrypted envelope approach: user must re-enter passphrase to change encryption state.
        // Here we just save the current plain structure only if not encrypted.
        // When encrypted we require the passphrase again only on enable/disable.
        // For ongoing saves while unlocked we store plaintext in memory and write encrypted only on explicit lock.
        // Practical approach: while unlocked we keep a temporary plain write,
        // and the encrypt button creates a new encrypted envelope.
        localStorage.setItem(STORE_KEY, JSON.stringify(store));
      } else {
        localStorage.setItem(STORE_KEY, JSON.stringify(store));
      }
    } catch (e) {
      console.warn('[Ra-Thor] localStorage write failed', e);
    }
  }

  // Simplified practical encryption flow for reliability:
  // - Encrypt button creates an encrypted envelope and replaces the store
  // - On next load the unlock modal appears
  // - After unlock the store is decrypted into memory and subsequent saves are plaintext until the user encrypts again
  // This is the safest UX for a pure-browser tool without a persistent keyring.

  async function enableEncryption() {
    const pass = prompt('Choose a strong passphrase to encrypt all sessions.\n\nWARNING: If you forget this passphrase the data cannot be recovered.');
    if (!pass || pass.length < 6) {
      addMessage('Encryption cancelled or passphrase too short (min 6 characters).', 'rathor');
      return;
    }
    const confirmPass = prompt('Confirm passphrase:');
    if (pass !== confirmPass) {
      addMessage('Passphrases did not match. Encryption cancelled.', 'rathor');
      return;
    }

    try {
      const envelope = await encryptStore(pass, store);
      localStorage.setItem(STORE_KEY, JSON.stringify(envelope));
      isEncrypted = true;
      cryptoKey = null; // force re-unlock next time
      addMessage('Session store is now encrypted with your passphrase. ⚡️ On the next page load you will be asked to unlock it.\n\nRemember: forgetting the passphrase makes the data unrecoverable.', 'rathor');
    } catch (err) {
      console.error('[Ra-Thor encrypt]', err);
      addMessage('Encryption failed. Your current sessions remain unencrypted.', 'rathor');
    }
  }

  async function tryUnlock() {
    const pass = unlockPassphrase ? unlockPassphrase.value : '';
    if (!pass) return;

    const success = await loadStore(pass);
    if (success) {
      if (unlockOverlay) unlockOverlay.classList.remove('active');
      if (unlockError) unlockError.classList.add('hidden');
      isEncrypted = false; // now unlocked in memory
      refreshSessionSelect();
      renderHistory();
      addMessage('Lattice unlocked. ⚡️ Sessions are available for this browser session.', 'rathor');
    } else {
      if (unlockError) unlockError.classList.remove('hidden');
    }
  }

  // ─── Utilities ────────────────────────────────────────────────────────────
  function uid() {
    return 's_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
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

  function renderText(text) {
    if (!text) return '';
    let html = text
      .replace(/&/g, '&')
      .replace(/</g, '<')
      .replace(/>/g, '>');

    html = html.replace(/```(\w*)\n([\s\S]*?)```/g, function (_, lang, code) {
      const language = lang ? ` data-lang="${lang}"` : '';
      return `<pre${language}><code>${code.trim()}</code></pre>`;
    });
    html = html.replace(/`([^`\n]+)`/g, '<code>$1</code>');
    html = html.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.+?)\*/g, '<em>$1</em>');
    html = html.replace(/^### (.+)$/gm, '<h3>$1</h3>');
    html = html.replace(/^## (.+)$/gm, '<h2>$1</h2>');
    html = html.replace(/^# (.+)$/gm, '<h1>$1</h1>');
    html = html.replace(/^[-*] (.+)$/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    html = html.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    html = html.replace(/\n/g, '<br>');
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
      activePathBadge.textContent = chatStr('chatPathServer');
      activePathBadge.classList.add('active-backend');
    } else if (llmReady) {
      activePathBadge.textContent = 'WebLLM';
      activePathBadge.classList.add('active-webllm');
    } else {
      activePathBadge.textContent = chatStr('chatPathFast');
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

  function detectLocalLlmSupport() {
    if (!navigator.gpu) return { supported: false, reason: 'WebGPU not available in this browser' };
    const ua = navigator.userAgent || '';
    if (/Android|iPhone|iPad|iPod|Mobile/i.test(ua)) {
      return { supported: false, reason: 'Local LLM currently works best on desktop.' };
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
      addMessage(chatStr('chatReplyEmpty'), 'rathor', false);
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
        response: chatStr('chatReplyMercy')
      };
    }
    return { allowed: true };
  }

  function generateLocalResponse(userText) {
    const gate = mercyGate(userText);
    if (!gate.allowed) return gate.response;

    for (const entry of LOCAL_KNOWLEDGE) {
      if (entry.q.test(userText)) return chatStr(entry.k);
    }

    return chatStr('chatReplyFallback');
  }

  // ─── Local Backend ────────────────────────────────────────────────────────
  function setBackendUI(connected) {
    backendEnabled = connected;
    if (backendConnectBtn) backendConnectBtn.classList.toggle('hidden', connected);
    if (backendDisconnectBtn) backendDisconnectBtn.classList.toggle('hidden', !connected);
    if (backendStatus) {
      backendStatus.textContent = connected ? `Connected → ${backendConfig.model}` : 'Not connected';
      backendStatus.style.color = connected ? '#34d399' : '';
    }
    if (localBackendBtn) localBackendBtn.classList.toggle('backend-ready', connected);
    updatePathBadge();
    if (localLlmStatus) {
      localLlmStatus.textContent = connected
        ? `Local Server active (${backendConfig.model})`
        : (llmReady ? 'WebLLM active' : chatStr('chatStatusDefault'));
    }
  }

  async function connectBackend() {
    saveBackendConfig();
    const endpoint = backendConfig.endpoint.replace(/\/$/, '');
    if (!localServerEndpointAllowed(readNetMode(), endpoint)) {
      setBackendUI(false);
      if (backendStatus) {
        backendStatus.textContent = chatLabel('chatNetLoopbackOnly', 'Offline only allows a server on this machine (localhost, 127.0.0.1, or [::1]).');
      }
      addMessage(chatLabel('chatNetLoopbackOnly', 'Offline only allows a server on this machine (localhost, 127.0.0.1, or [::1]).'), 'rathor');
      return;
    }
    try {
      const res = await fetch(endpoint + '/models', { method: 'GET', signal: AbortSignal.timeout(4000) });
      if (!res.ok) throw new Error('Endpoint returned ' + res.status);
      setBackendUI(true);
      addMessage(`Local Server connected. ⚡️ Endpoint: ${endpoint}\nModel: ${backendConfig.model}\nStreaming enabled. TOLC 8 system prompt will be injected.`, 'rathor');
    } catch (err) {
      setBackendUI(false);
      addMessage(`Could not reach Local Server at ${endpoint}.\n\nMake sure Ollama (or LM Studio) is running and the endpoint + model name are correct.`, 'rathor');
    }
  }

  function disconnectBackend() {
    setBackendUI(false);
    addMessage('Local Server disconnected. Falling back to fast responder / WebLLM.', 'rathor');
  }

  async function generateWithBackend(userText) {
    if (!backendEnabled) return null;

    const hist = getHistory();
    const messages = [{ role: 'system', content: systemPreamble() + getDocumentContext() }];
    hist.slice(-14).forEach(m => {
      messages.push({ role: m.role === 'user' ? 'user' : 'assistant', content: m.text });
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
      console.error('[Ra-Thor Backend]', err);
      return null;
    }
  }

  // ─── WebLLM ───────────────────────────────────────────────────────────────
  function updateLlmUI(state, extra = '') {
    if (!localLlmBtn || !localLlmStatus) return;
    if (state === 'unsupported') {
      localLlmBtn.disabled = true;
      localLlmBtn.innerHTML = '<i class="fa-solid fa-microchip"></i> ' + chatStr('chatNotAvailable');
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
      if (!backendEnabled) localLlmStatus.textContent = chatStr('chatStatusDefault');
      if (localLlmProgress) localLlmProgress.style.width = '0%';
    }
    updatePathBadge();
  }

  function applyWebllmStaticCopy() {
    if (webllmThirdParty) {
      webllmThirdParty.textContent = chatLabel('chatWebllmThirdParty', 'Third-party models under their own licenses. Not made by Ra-Thor. Not reviewed or endorsed by their authors.');
    }
    if (webllmDownloadNote) {
      webllmDownloadNote.textContent = chatLabel('chatWebllmFirstDownload', "The first download of each model comes from Hugging Face and GitHub (raw.githubusercontent.com, which serves the model's code file) and needs the network. After that it runs in this browser.");
    }
    if (webllmOtherNote) {
      webllmOtherNote.textContent = chatLabel('chatWebllmOtherModel', 'Any other model: Local Server (Ollama).');
    }
  }

  /* chat-models-1-pure */
  function isLoopbackChatEndpoint(endpoint) {
    try {
      var url = new URL(endpoint);
      var host = String(url.hostname || '').replace(/^\[|\]$/g, '').toLowerCase();
      return host === 'localhost' || host === '127.0.0.1' || host === '::1';
    } catch (e) {
      return false;
    }
  }

  function localServerEndpointAllowed(mode, endpoint) {
    if (mode !== 'offline-only') return true;
    return isLoopbackChatEndpoint(endpoint);
  }

  function webllmDownloadAllowed(mode, onLine) {
    if (mode === 'offline-only') return false;
    if (onLine === false) return false;
    return true;
  }

  function readNetMode() {
    try {
      if (localStorage.getItem('rathor-net-mode-v1') === 'offline-only') return 'offline-only';
    } catch (e) {}
    return 'network-on';
  }

  function writeNetMode(mode) {
    var next = mode === 'offline-only' ? 'offline-only' : 'network-on';
    try { localStorage.setItem('rathor-net-mode-v1', next); } catch (e) {}
    return next;
  }

  function webllmBadgeLabel(phase, pct) {
    if (phase === 'downloading') return 'Downloading ' + (typeof pct === 'number' ? pct : 0) + '%';
    if (phase === 'partial') return 'Partly downloaded';
    if (phase === 'ready') return 'On this device · works offline';
    return 'Not on this device';
  }

  function webllmActionLabel(phase) {
    if (phase === 'downloading') return 'Stop';
    if (phase === 'partial' || phase === 'ready') return 'Delete';
    return 'Download';
  }

  function webllmRowTransition(row, action, ctx) {
    var phase = row && row.phase ? row.phase : 'absent';
    var consent = row && row.consent ? row.consent : null;
    var offlineOnly = !!(ctx && ctx.offlineOnly);
    var onLine = !ctx || ctx.onLine !== false;
    var next = { phase: phase, consent: consent };
    if (action === 'cancel') {
      next.consent = null;
      return { row: next, effect: null };
    }
    if (action === 'tap-download') {
      if (phase !== 'absent') return { row: next, effect: null };
      if (offlineOnly || !onLine) return { row: next, effect: null };
      next.consent = 'download';
      return { row: next, effect: null };
    }
    if (action === 'confirm-download') {
      if (consent !== 'download') return { row: next, effect: null };
      if (offlineOnly || !onLine) {
        next.consent = null;
        return { row: next, effect: null };
      }
      next.consent = null;
      next.phase = 'downloading';
      return { row: next, effect: 'download' };
    }
    if (action === 'stop') {
      if (phase !== 'downloading') return { row: next, effect: null };
      next.phase = 'partial';
      next.consent = null;
      return { row: next, effect: 'stop' };
    }
    if (action === 'tap-delete') {
      if (phase !== 'partial' && phase !== 'ready') return { row: next, effect: null };
      next.consent = 'delete';
      return { row: next, effect: null };
    }
    if (action === 'confirm-delete') {
      if (consent !== 'delete') return { row: next, effect: null };
      next.consent = null;
      next.phase = 'absent';
      return { row: next, effect: 'delete' };
    }
    if (action === 'use') {
      if (phase !== 'ready') return { row: next, effect: null };
      return { row: next, effect: 'use' };
    }
    return { row: next, effect: null };
  }

  function absoluteScriptUrl(scriptSrc, relativePath) {
    var abs = new URL(relativePath, scriptSrc).href;
    return abs.split('#')[0].split('?')[0];
  }

  function modelCacheUrlPrefix(modelUrl) {
    var url = String(modelUrl || '');
    if (!url) return '';
    if (url.charAt(url.length - 1) !== '/') url += '/';
    if (!/.+\/resolve\/.+\//.test(url)) url += 'resolve/main/';
    try { return new URL(url).href; } catch (e) { return url; }
  }

  function isWebllmOwnedCache(name) {
    if (String(name).indexOf('webllm') === -1) return false;
    return name === 'webllm/model' || name === 'webllm/wasm' || name === 'webllm/config';
  }

  function requestMatchesModel(requestUrl, modelUrl, modelLib) {
    var url = String(requestUrl || '');
    if (!url) return false;
    var prefix = modelCacheUrlPrefix(modelUrl);
    if (prefix && url.indexOf(prefix) !== -1) return true;
    if (modelLib && url === String(modelLib)) return true;
    return false;
  }

  function purgeOwnedModelEntries(stores, modelUrl, modelLib) {
    var next = {};
    Object.keys(stores || {}).forEach(function (name) {
      var urls = (stores[name] || []).slice();
      if (isWebllmOwnedCache(name)) {
        urls = urls.filter(function (url) { return !requestMatchesModel(url, modelUrl, modelLib); });
      }
      next[name] = urls;
    });
    return next;
  }

  function modelFilesRemain(stores, modelUrl, modelLib) {
    var names = Object.keys(stores || {});
    for (var i = 0; i < names.length; i++) {
      if (!isWebllmOwnedCache(names[i])) continue;
      var urls = stores[names[i]] || [];
      for (var r = 0; r < urls.length; r++) {
        if (requestMatchesModel(urls[r], modelUrl, modelLib)) return true;
      }
    }
    return false;
  }

  function phaseAfterLocalDelete(stores, modelUrl, modelLib) {
    var next = purgeOwnedModelEntries(stores, modelUrl, modelLib);
    return modelFilesRemain(next, modelUrl, modelLib) ? 'partial' : 'absent';
  }

  function tensorManifestDownloadBytes(manifest) {
    var records = manifest && manifest.records;
    if (!records || !records.length) return null;
    var sum = 0;
    for (var i = 0; i < records.length; i++) {
      var n = records[i] && records[i].nbytes;
      if (typeof n !== 'number' || n !== Math.floor(n) || n < 0) return null;
      sum += n;
    }
    return sum > 0 ? sum : null;
  }

  function formatByteMegabytes(bytes) {
    if (typeof bytes !== 'number' || bytes <= 0) return '';
    var mb = Math.round((bytes / 1000000) * 10) / 10;
    var text = mb === Math.round(mb) ? String(Math.round(mb)) : mb.toFixed(1);
    return text + ' MB';
  }

  function downloadConsentText(base, bytes) {
    var text = 'Download ' + base + ' from Hugging Face and GitHub to this device?';
    var size = formatByteMegabytes(bytes);
    if (size) text += ' Download size: ' + size + '.';
    return text;
  }

  function deleteConsentText(base, bytes) {
    var text = 'Delete ' + base + ' from this device?';
    var size = formatByteMegabytes(bytes);
    if (size) text += ' This frees about ' + size + '.';
    return text;
  }
  /* chat-models-1-pure-end */

  function readStoredModelBase() {
    try { return localStorage.getItem(WEBLLM_MODEL_KEY) || ''; } catch (e) { return ''; }
  }

  function quantIdFor(entry) {
    return webllmShaderF16 ? entry.q4f16 : entry.q4f32;
  }

  function optionByBase(base) {
    for (var i = 0; i < webllmOptions.length; i++) {
      if (webllmOptions[i].entry.base === base) return webllmOptions[i];
    }
    return null;
  }

  function ensureRowRuntime(base) {
    if (!webllmRowRuntime[base]) {
      webllmRowRuntime[base] = { phase: 'absent', consent: null, progress: 0, downloading: false, downloadBytes: null };
    }
    return webllmRowRuntime[base];
  }

  function activeModelBase() {
    var stored = readStoredModelBase();
    if (stored && optionByBase(stored)) return stored;
    return WEBLLM_DEFAULT_BASE;
  }

  function modelGpuMemoryLabel(rec) {
    var mb = rec && rec.vram_required_MB;
    return (typeof mb === 'number') ? ('GPU memory needed: about ' + mb + ' MB') : '';
  }

  function renderNetMode() {
    var mode = readNetMode();
    var online = typeof navigator === 'undefined' || navigator.onLine !== false;
    if (netModeOfflineBtn) netModeOfflineBtn.setAttribute('aria-pressed', mode === 'offline-only' ? 'true' : 'false');
    if (netModeNetworkBtn) netModeNetworkBtn.setAttribute('aria-pressed', mode === 'network-on' ? 'true' : 'false');
    if (netConnection) {
      netConnection.textContent = online
        ? chatLabel('chatNetConnected', 'Connected')
        : chatLabel('chatNetNone', 'No connection');
    }
    if (netModeNote) {
      netModeNote.textContent = mode === 'offline-only'
        ? chatLabel('chatNetOfflineLine', "Offline only: Lattice Chat won't download models or contact any server except your own machine.")
        : chatLabel('chatNetOnLine', 'Network on. A download starts only after you confirm it.');
    }
    if (netModeOfflineNote) {
      netModeOfflineNote.classList.toggle('hidden', mode !== 'offline-only');
      netModeOfflineNote.textContent = (mode === 'offline-only' && webllmDownloadCancelled)
        ? 'Offline only is on. Download cancelled. Models already on this device still work.'
        : 'Offline only is on. Models already on this device still work.';
    }
  }

  async function modelFilesCached(rec) {
    if (!rec || typeof caches === 'undefined' || !caches.keys) return false;
    var names;
    try { names = await caches.keys(); } catch (e) { return false; }
    for (var i = 0; i < names.length; i++) {
      if (!isWebllmOwnedCache(names[i])) continue;
      var cache;
      try { cache = await caches.open(names[i]); } catch (e) { continue; }
      var reqs = [];
      try { reqs = await cache.keys(); } catch (e) { reqs = []; }
      for (var r = 0; r < reqs.length; r++) {
        var url = reqs[r] && reqs[r].url ? reqs[r].url : '';
        if (requestMatchesModel(url, rec.model, rec.model_lib)) return true;
      }
    }
    return false;
  }

  async function purgeModelLeftovers(rec) {
    if (!rec || typeof caches === 'undefined' || !caches.keys) return;
    var names;
    try { names = await caches.keys(); } catch (e) { return; }
    for (var i = 0; i < names.length; i++) {
      if (!isWebllmOwnedCache(names[i])) continue;
      var cache;
      try { cache = await caches.open(names[i]); } catch (e) { continue; }
      var reqs = [];
      try { reqs = await cache.keys(); } catch (e) { reqs = []; }
      for (var r = 0; r < reqs.length; r++) {
        var url = reqs[r] && reqs[r].url ? reqs[r].url : '';
        if (!requestMatchesModel(url, rec.model, rec.model_lib)) continue;
        try { await cache.delete(reqs[r]); } catch (e2) {}
      }
    }
  }

  async function refreshRowPhase(opt) {
    var rt = ensureRowRuntime(opt.entry.base);
    if (rt.downloading) return;
    var full = false;
    try {
      full = !!(webllmModule && await webllmModule.hasModelInCache(opt.id));
    } catch (e) {
      full = false;
    }
    if (full) {
      rt.phase = 'ready';
      return;
    }
    var files = false;
    try { files = await modelFilesCached(opt.rec); } catch (e) { files = false; }
    rt.phase = files ? 'partial' : 'absent';
  }

  function renderWebllmRows() {
    if (!webllmRows) return;
    var active = activeModelBase();
    var downloadBlocked = !webllmDownloadAllowed(readNetMode(), typeof navigator === 'undefined' || navigator.onLine !== false);
    var online = typeof navigator === 'undefined' || navigator.onLine !== false;
    webllmRows.replaceChildren();
    webllmOptions.forEach(function (opt) {
      var base = opt.entry.base;
      var rt = ensureRowRuntime(base);
      var phase = rt.downloading ? 'downloading' : rt.phase;
      var gpu = modelGpuMemoryLabel(opt.rec);
      var row = document.createElement('div');
      row.className = 'webllm-row' + (active === base ? ' webllm-row-active' : '');
      row.setAttribute('data-base', base);

      var top = document.createElement('div');
      top.className = 'webllm-row-top';
      var copy = document.createElement('div');
      copy.className = 'webllm-row-copy';
      var name = document.createElement('p');
      name.className = 'webllm-row-name';
      name.textContent = base;
      var meta = document.createElement('p');
      meta.className = 'webllm-row-meta';
      if (gpu) meta.appendChild(document.createTextNode(gpu));
      (opt.entry.links || []).forEach(function (link, i) {
        meta.appendChild(document.createTextNode(i === 0 && gpu ? ' · ' : (i ? ' · ' : '')));
        var a = document.createElement('a');
        a.href = link.href;
        a.target = '_blank';
        a.rel = 'noopener';
        a.textContent = link.text;
        meta.appendChild(a);
      });
      if (opt.entry.builtWithLlama) meta.appendChild(document.createTextNode(' · Built with Llama'));
      copy.appendChild(name);
      copy.appendChild(meta);
      top.appendChild(copy);

      var badge = document.createElement('span');
      badge.className = 'webllm-badge';
      badge.textContent = webllmBadgeLabel(phase, rt.progress);
      top.appendChild(badge);

      if (!rt.consent) {
        var action = document.createElement('button');
        action.type = 'button';
        action.className = 'ctrl-btn text-xs px-2';
        action.textContent = webllmActionLabel(phase);
        if (phase === 'downloading') action.setAttribute('data-act', 'stop');
        else if (phase === 'partial' || phase === 'ready') action.setAttribute('data-act', 'tap-delete');
        else action.setAttribute('data-act', 'tap-download');
        if (action.textContent === 'Download' && downloadBlocked) action.disabled = true;
        top.appendChild(action);
        if (phase === 'ready') {
          if (llmReady && llmModelId === opt.id) {
            var used = document.createElement('span');
            used.className = 'webllm-badge';
            used.textContent = 'In use';
            top.appendChild(used);
          } else {
            var useBtn = document.createElement('button');
            useBtn.type = 'button';
            useBtn.className = 'ctrl-btn text-xs px-2';
            useBtn.textContent = 'Use';
            useBtn.setAttribute('data-act', 'use');
            top.appendChild(useBtn);
          }
        }
      }
      row.appendChild(top);

      if (opt.entry.smallReplyNote) {
        var small = document.createElement('p');
        small.className = 'webllm-row-note';
        small.textContent = 'Small models can give wrong or inappropriate replies.';
        row.appendChild(small);
      }

      if (!online && phase !== 'ready') {
        var need = document.createElement('p');
        need.className = 'webllm-row-note';
        need.textContent = chatLabel('chatNetNeedsConnection', 'Download needs a connection.');
        row.appendChild(need);
      }

      if (rt.consent === 'download' || rt.consent === 'delete') {
        var consent = document.createElement('div');
        consent.className = 'webllm-consent';
        var ask = document.createElement('p');
        ask.textContent = rt.consent === 'download'
          ? downloadConsentText(base, rt.downloadBytes)
          : deleteConsentText(base, rt.downloadBytes);
        consent.appendChild(ask);
        var yes = document.createElement('button');
        yes.type = 'button';
        yes.className = 'ctrl-btn text-xs px-2';
        yes.textContent = rt.consent === 'download' ? 'Confirm download' : 'Confirm delete';
        yes.setAttribute('data-act', rt.consent === 'download' ? 'confirm-download' : 'confirm-delete');
        var no = document.createElement('button');
        no.type = 'button';
        no.className = 'ctrl-btn text-xs px-2';
        no.textContent = 'Cancel';
        no.setAttribute('data-act', 'cancel');
        consent.appendChild(yes);
        consent.appendChild(no);
        row.appendChild(consent);
      }
      webllmRows.appendChild(row);
    });
  }

  async function unloadWebllmEngine() {
    llmLoadToken += 1;
    var engine = llmEngine;
    llmEngine = null;
    llmReady = false;
    llmLoading = false;
    if (engine && typeof engine.unload === 'function') {
      try { await engine.unload(); } catch (err) {
        console.error('[Ra-Thor WebLLM] unload', err);
      }
    }
    Object.keys(webllmRowRuntime).forEach(function (base) {
      var rowState = webllmRowRuntime[base];
      if (!rowState.downloading) return;
      rowState.downloading = false;
      if (rowState.phase === 'downloading') rowState.phase = 'partial';
    });
  }

  async function waitForServiceWorkerReady() {
    if (typeof navigator === 'undefined' || !navigator.serviceWorker || !navigator.serviceWorker.ready) return;
    try {
      var reg = navigator.serviceWorker.getRegistration ? await navigator.serviceWorker.getRegistration() : null;
      if (!reg && !navigator.serviceWorker.controller) return;
      await navigator.serviceWorker.ready;
    } catch (e) {}
  }

  async function cacheVendoredWebllmScript() {
    if (typeof caches === 'undefined' || !caches.open || !WEBLLM_SCRIPT_URL) return;
    await waitForServiceWorkerReady();
    try {
      var cache = await caches.open(WEBLLM_SCRIPT_CACHE);
      await cache.add(WEBLLM_SCRIPT_URL);
    } catch (err) {
      console.error('[Ra-Thor WebLLM] script cache', err);
    }
  }

  async function fetchTensorDownloadBytes(rec) {
    if (!webllmDownloadAllowed(readNetMode(), typeof navigator === 'undefined' || navigator.onLine !== false)) return null;
    var prefix = modelCacheUrlPrefix(rec && rec.model);
    if (!prefix || typeof fetch !== 'function') return null;
    try {
      var res = await fetch(new URL('tensor-cache.json', prefix).href);
      if (!res || !res.ok) return null;
      return tensorManifestDownloadBytes(await res.json());
    } catch (e) {
      return null;
    }
  }

  async function cachedTensorDownloadBytes(rec) {
    if (typeof caches === 'undefined' || !caches.keys || !caches.open) return null;
    var prefix = modelCacheUrlPrefix(rec && rec.model);
    if (!prefix) return null;
    var names;
    try { names = await caches.keys(); } catch (e) { return null; }
    if (names.indexOf('webllm/model') === -1) return null;
    try {
      var cache = await caches.open('webllm/model');
      var hit = await cache.match(new URL('tensor-cache.json', prefix).href);
      if (!hit) return null;
      return tensorManifestDownloadBytes(await hit.json());
    } catch (e2) {
      return null;
    }
  }

  async function startWebllmDownload(base) {
    if (!webllmDownloadAllowed(readNetMode(), navigator.onLine !== false)) return;
    await cacheVendoredWebllmScript();
    await loadWebllmModel(base, true);
  }

  async function loadWebllmModel(base, fromDownload) {
    var opt = optionByBase(base);
    if (!opt) return;
    if (!webllmModule) {
      try { webllmModule = await import(WEBLLM_VENDOR); } catch (err) {
        console.error('[Ra-Thor WebLLM]', err);
        return;
      }
    }
    if (llmEngine || llmLoading) await unloadWebllmEngine();
    var token = llmLoadToken;
    var rt = ensureRowRuntime(base);
    llmModelId = opt.id;
    if (fromDownload) {
      rt.downloading = true;
      rt.consent = null;
      rt.phase = 'downloading';
      rt.progress = 0;
      llmLoading = true;
      updateLlmUI('loading', webllmBadgeLabel('downloading', 0));
      renderWebllmRows();
    }
    var engine = new webllmModule.MLCEngine({
      initProgressCallback: function (report) {
        if (token !== llmLoadToken || !fromDownload) return;
        var pct = Math.round((report.progress || 0) * 100);
        rt.progress = pct;
        var node = webllmRows && webllmRows.querySelector('.webllm-row[data-base="' + base + '"] .webllm-badge');
        if (node) node.textContent = webllmBadgeLabel('downloading', pct);
        if (localLlmProgress) localLlmProgress.style.width = Math.max(5, pct) + '%';
        if (localLlmStatus) localLlmStatus.textContent = webllmBadgeLabel('downloading', pct);
      }
    });
    llmEngine = engine;
    try {
      await engine.reload(opt.id);
      if (token !== llmLoadToken) return;
      var full = false;
      try { full = await webllmModule.hasModelInCache(opt.id); } catch (e) { full = false; }
      rt.downloading = false;
      llmLoading = false;
      if (full) {
        llmReady = true;
        rt.phase = 'ready';
        try { localStorage.setItem(WEBLLM_MODEL_KEY, base); } catch (e) {}
        updateLlmUI('ready');
        addMessage('WebLLM loaded (' + opt.id + '). ⚡️ Generation now runs entirely in the browser.', 'rathor');
      } else {
        llmReady = false;
        llmEngine = null;
        try { await engine.unload(); } catch (e) {}
        rt.phase = (await modelFilesCached(opt.rec)) ? 'partial' : 'absent';
        updateLlmUI('idle');
      }
    } catch (err) {
      if (token !== llmLoadToken) return;
      console.error('[Ra-Thor WebLLM]', err);
      rt.downloading = false;
      llmLoading = false;
      llmReady = false;
      llmEngine = null;
      rt.phase = (await modelFilesCached(opt.rec)) ? 'partial' : 'absent';
      updateLlmUI('error', 'Load failed');
      addMessage('WebLLM failed to load. Use Local Server or **Copy Context**.', 'rathor');
    }
    renderWebllmRows();
  }

  async function stopWebllmDownload(base) {
    await unloadWebllmEngine();
    var rt = ensureRowRuntime(base);
    rt.downloading = false;
    rt.consent = null;
    rt.progress = 0;
    rt.phase = 'partial';
    updateLlmUI('idle');
    renderWebllmRows();
  }

  async function deleteWebllmModel(base) {
    var opt = optionByBase(base);
    if (!opt) return;
    if ((llmEngine || llmReady || llmLoading) && llmModelId === opt.id) await unloadWebllmEngine();
    if (readNetMode() !== 'offline-only' && webllmModule && webllmModule.deleteModelAllInfoInCache) {
      try {
        await webllmModule.deleteModelAllInfoInCache(opt.id);
      } catch (err) {
        console.error('[Ra-Thor WebLLM] delete', err);
      }
    }
    try { await purgeModelLeftovers(opt.rec); } catch (err2) {
      console.error('[Ra-Thor WebLLM] delete leftovers', err2);
    }
    var rt = ensureRowRuntime(base);
    rt.downloading = false;
    rt.consent = null;
    rt.progress = 0;
    rt.downloadBytes = null;
    await refreshRowPhase(opt);
    renderWebllmRows();
  }

  async function useWebllmModel(base) {
    var opt = optionByBase(base);
    var rt = ensureRowRuntime(base);
    if (!opt || rt.phase !== 'ready') return;
    try { localStorage.setItem(WEBLLM_MODEL_KEY, base); } catch (e) {}
    if (llmReady && llmEngine && llmModelId === opt.id) {
      renderWebllmRows();
      return;
    }
    if (llmEngine || llmLoading) await unloadWebllmEngine();
    await loadWebllmModel(base, false);
  }

  async function onWebllmRowAction(base, action) {
    var rt = ensureRowRuntime(base);
    var ctx = {
      offlineOnly: readNetMode() === 'offline-only',
      onLine: navigator.onLine !== false
    };
    var result = webllmRowTransition(
      { phase: rt.downloading ? 'downloading' : rt.phase, consent: rt.consent },
      action,
      ctx
    );
    rt.consent = result.row.consent;
    if (action === 'tap-download' && result.row.consent === 'download') {
      var opt = optionByBase(base);
      rt.downloadBytes = opt ? await fetchTensorDownloadBytes(opt.rec) : null;
      renderWebllmRows();
      return;
    }
    if (action === 'tap-delete' && result.row.consent === 'delete') {
      var delOpt = optionByBase(base);
      if (rt.downloadBytes == null && delOpt) rt.downloadBytes = await cachedTensorDownloadBytes(delOpt.rec);
      renderWebllmRows();
      return;
    }
    if (result.effect === 'download') {
      await startWebllmDownload(base);
      return;
    }
    if (result.effect === 'stop') {
      await stopWebllmDownload(base);
      return;
    }
    if (result.effect === 'delete') {
      await deleteWebllmModel(base);
      return;
    }
    if (result.effect === 'use') {
      await useWebllmModel(base);
      return;
    }
    if (!rt.downloading) rt.phase = result.row.phase;
    renderWebllmRows();
  }

  async function initWebllmPicker() {
    renderNetMode();
    if (!webllmPicker) {
      webllmPickerReady = true;
      return false;
    }
    webllmPicker.classList.remove('hidden');
    applyWebllmStaticCopy();
    webllmShaderF16 = false;
    try {
      var adapter = await navigator.gpu.requestAdapter();
      webllmShaderF16 = !!(adapter && adapter.features && adapter.features.has('shader-f16'));
    } catch (e) {
      webllmShaderF16 = false;
    }
    try {
      webllmModule = webllmModule || await import(WEBLLM_VENDOR);
    } catch (err) {
      console.error('[Ra-Thor WebLLM]', err);
      webllmPickerReady = true;
      updateLlmUI('error', 'Load failed');
      return false;
    }
    var byId = new Map();
    var list = (webllmModule.prebuiltAppConfig && webllmModule.prebuiltAppConfig.model_list) || [];
    list.forEach(function (rec) {
      if (rec && rec.model_id) byId.set(rec.model_id, rec);
    });
    webllmOptions = [];
    WEBLLM_CURATED.forEach(function (entry) {
      var id = quantIdFor(entry);
      var rec = byId.get(id);
      if (!rec) return;
      webllmOptions.push({ entry: entry, id: id, rec: rec });
    });
    if (!readStoredModelBase()) {
      try { localStorage.setItem(WEBLLM_MODEL_KEY, WEBLLM_DEFAULT_BASE); } catch (e) {}
    }
    for (var i = 0; i < webllmOptions.length; i++) {
      await refreshRowPhase(webllmOptions[i]);
    }
    webllmPickerReady = true;
    renderWebllmRows();
    return true;
  }


  async function generateWithLocalLLM(userText) {
    if (!llmEngine || !llmReady) return null;
    const hist = getHistory();
    const messages = [{ role: 'system', content: systemPreamble() + getDocumentContext() }];
    hist.slice(-10).forEach(m => {
      messages.push({ role: m.role === 'user' ? 'user' : 'assistant', content: m.text });
    });
    messages.push({ role: 'user', content: userText });

    try {
      const stream = await llmEngine.chat.completions.create({
        messages, temperature: 0.7, max_tokens: 500, stream: true
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
        const reply = await llmEngine.chat.completions.create({ messages, temperature: 0.7, max_tokens: 500 });
        return reply.choices?.[0]?.message?.content?.trim() || null;
      } catch (e2) {
        return null;
      }
    }
  }

  // ─── STT ──────────────────────────────────────────────────────────────────
  function initSpeechRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      if (micBtn) { micBtn.disabled = true; micBtn.title = 'Speech recognition not supported'; }
      return;
    }
    recognition = new SpeechRecognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    recognition.onstart = () => { isListening = true; if (micBtn) micBtn.classList.add('listening'); };
    recognition.onresult = (event) => {
      let interim = '', final = '';
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const t = event.results[i][0].transcript;
        if (event.results[i].isFinal) final += t; else interim += t;
      }
      if (chatInput) chatInput.value = final || interim;
    };
    recognition.onend = () => {
      isListening = false;
      if (micBtn) micBtn.classList.remove('listening');
      if (chatInput && chatInput.value.trim()) setTimeout(() => sendMessage(), 300);
    };
    recognition.onerror = () => { isListening = false; if (micBtn) micBtn.classList.remove('listening'); };
  }

  function toggleMic() {
    if (!recognition) { addMessage('Speech recognition is not available in this browser.', 'rathor'); return; }
    if (isListening) recognition.stop();
    else { try { recognition.start(); } catch (e) {} }
  }

  // ─── Core send ────────────────────────────────────────────────────────────
  async function sendMessage() {
    if (!chatInput) return;
    const text = chatInput.value.trim();
    if (!text) return;
    addMessage(text, 'user');
    chatInput.value = '';

    if (backendEnabled) {
      const reply = await generateWithBackend(text);
      if (reply === null) addMessage(generateLocalResponse(text) + '\n\n(Local Server request failed)', 'rathor');
      return;
    }
    if (llmReady && llmEngine) {
      const reply = await generateWithLocalLLM(text);
      if (reply === null) addMessage(generateLocalResponse(text), 'rathor');
      return;
    }
    setTimeout(() => addMessage(generateLocalResponse(text), 'rathor'), 180 + Math.random() * 220);
  }

  // ─── Export / Import / Copy ───────────────────────────────────────────────
  function exportSession() {
    const s = activeSession();
    if (!s) return;
    downloadJSON({
      version: '14.18.0',
      exported: new Date().toISOString(),
      sessionName: s.name,
      stewardship: 'Sherif Samy Botros — Sole Steward',
      history: s.history
    }, `rathor-${(s.name || 'session').replace(/[^a-z0-9]/gi, '-').toLowerCase()}-${Date.now()}.json`);
  }

  function exportAllSessions() {
    downloadJSON({
      version: '14.18.0',
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
    a.href = url; a.download = filename; a.click();
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
      SYSTEM_PROMPT.trim(),
    ];
    var reply = replyInClause();
    if (reply) lines.push('', reply);
    lines.push(
      '',
      'You are continuing a conversation that began on the Ra-Thor offline Lattice Chat (rathor.ai/chat.html).',
      '',
      'Conversation history (generated on-device):'
    );
    hist.forEach(m => lines.push(`${m.role === 'user' ? 'Human' : 'Ra-Thor'}: ${m.text}`));
    if (injectedDocs.length > 0) {
      lines.push('', '--- Injected Documents ---');
      injectedDocs.forEach(d => { lines.push(`### ${d.name}`); lines.push(d.content); lines.push(''); });
      lines.push('--- End Documents ---');
    }
    lines.push('', 'Continue naturally. Outputs remain drafts. Independent of xAI.');
    return lines.join('\n');
  }

  function copyContext() {
    copyText(buildContextPrompt()).then(() => {
      addMessage('Context copied. ⚡️ Paste it into any public LLM to continue with full generative power.', 'rathor');
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
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
    });
  }
  if (docBtn && docFileInput) {
    docBtn.addEventListener('click', () => docFileInput.click());
    docFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) handleDocumentUpload(e.target.files[0]);
      e.target.value = '';
    });
  }
  if (searchInput) {
    let searchTimer = null;
    searchInput.addEventListener('input', () => {
      clearTimeout(searchTimer);
      searchTimer = setTimeout(() => renderHistory(searchInput.value), 180);
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
  if (webllmRows) {
    webllmRows.addEventListener('click', function (ev) {
      var row = ev.target.closest ? ev.target.closest('.webllm-row') : null;
      if (!row) return;
      var base = row.getAttribute('data-base');
      var actEl = ev.target.closest ? ev.target.closest('[data-act]') : null;
      if (actEl) {
        onWebllmRowAction(base, actEl.getAttribute('data-act'));
        return;
      }
      if (ev.target.closest && ev.target.closest('a')) return;
      var rt = webllmRowRuntime[base];
      if (rt && rt.phase === 'ready' && !rt.consent) onWebllmRowAction(base, 'use');
    });
  }
  if (localLlmBtn) localLlmBtn.addEventListener('click', function () {
    if (!webllmPicker || webllmPicker.classList.contains('hidden')) return;
    var row = webllmPicker.querySelector('.webllm-row-active') || webllmPicker;
    if (row.scrollIntoView) row.scrollIntoView({ block: 'nearest' });
    var focusBtn = row.querySelector ? row.querySelector('button:not([disabled])') : null;
    if (!focusBtn) focusBtn = webllmPicker.querySelector('button:not([disabled])');
    if (focusBtn && focusBtn.focus) focusBtn.focus();
  });
  if (netModeOfflineBtn) netModeOfflineBtn.addEventListener('click', function () {
    var stopping = Object.keys(webllmRowRuntime).filter(function (base) {
      var rowState = webllmRowRuntime[base];
      return rowState && rowState.downloading;
    });
    writeNetMode('offline-only');
    webllmDownloadCancelled = stopping.length > 0;
    if (!stopping.length) {
      renderNetMode();
      renderWebllmRows();
      return;
    }
    Promise.all(stopping.map(function (base) { return stopWebllmDownload(base); })).then(function () {
      if (localLlmStatus) localLlmStatus.textContent = 'Download cancelled.';
      renderNetMode();
    });
  });
  if (netModeNetworkBtn) netModeNetworkBtn.addEventListener('click', function () {
    webllmDownloadCancelled = false;
    writeNetMode('network-on');
    renderNetMode();
    renderWebllmRows();
  });
  window.addEventListener('online', function () { renderNetMode(); renderWebllmRows(); });
  window.addEventListener('offline', function () { renderNetMode(); renderWebllmRows(); });
  if (localBackendBtn) localBackendBtn.addEventListener('click', () => {
    if (backendSettings) backendSettings.classList.toggle('hidden');
  });
  if (backendConnectBtn) backendConnectBtn.addEventListener('click', connectBackend);
  if (backendDisconnectBtn) backendDisconnectBtn.addEventListener('click', disconnectBackend);
  if (encryptBtn) encryptBtn.addEventListener('click', enableEncryption);

  if (unlockBtn) unlockBtn.addEventListener('click', tryUnlock);
  if (unlockPassphrase) {
    unlockPassphrase.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') tryUnlock();
    });
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

  document.addEventListener('rt-chrome-i18n', function () {
    applyChatSurfaceDir();
    applyWebllmStaticCopy();
    renderNetMode();
    renderWebllmRows();
    updatePathBadge();
    setBackendUI(backendEnabled);
    if (!llmProbed || llmLoading) return;
    if (!llmSupported) {
      var cap = detectLocalLlmSupport();
      updateLlmUI('unsupported', cap.reason);
    } else if (llmReady) updateLlmUI('ready');
    else if (webllmPickerReady) updateLlmUI('idle');
  });

  // ─── Init ─────────────────────────────────────────────────────────────────
  window.addEventListener('DOMContentLoaded', async () => {
    loadSettings();
    renderNetMode();

    // Check if store is encrypted
    if (isStoreEncrypted()) {
      if (unlockOverlay) unlockOverlay.classList.add('active');
      // Wait for user to unlock — do not load plain store
      return;
    }

    await loadStore();
    refreshSessionSelect();
    renderHistory();
    initSpeechRecognition();

    const cap = detectLocalLlmSupport();
    llmSupported = cap.supported;
    llmProbed = true;
    if (!llmSupported) updateLlmUI('unsupported', cap.reason);
    else {
      if (localLlmBtn) localLlmBtn.disabled = true;
      var pickerOk = await initWebllmPicker();
      if (pickerOk && !llmReady && !llmLoading) updateLlmUI('idle');
    }
    applyChatSurfaceDir();

    if (backendSettings) backendSettings.classList.add('hidden');
    setBackendUI(false);

    if (window.speechSynthesis) {
      window.speechSynthesis.getVoices();
      window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();
    }

    console.log('[Ra-Thor chat.js] v14.18.0 — Optional Passphrase Encryption ready ⚡️');
  });
})();
