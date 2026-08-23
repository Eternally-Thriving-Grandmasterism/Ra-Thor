/* js/rathor-feedback.js — human-readable 3s status across rathor.ai
 * Workspace 14.15.6 · TOLC 8 · info@Rathor.ai
 * On-device only. No analytics. Respects prefers-reduced-motion.
 *
 * Speaks what happened — not chrome chrome.
 * State changes, already-installed, theme, language, copy, export,
 * outbound hops, offline. Ordinary same-origin nav stays quiet.
 */
(function () {
  'use strict';

  if (window.rathorSay && window.rathorSay.__rtFull) return;

  if (!document.getElementById('rt-hf-css')) {
    var css = document.createElement('style');
    css.id = 'rt-hf-css';
    css.textContent =
      '#rt-human-feedback{position:fixed;left:50%;bottom:1.15rem;transform:translateX(-50%) translateY(12px);z-index:10050;max-width:min(26rem,calc(100vw - 1.5rem));padding:.72rem .95rem .78rem;border-radius:1rem;border:1px solid var(--rt-line,rgba(240,211,106,.34));background:var(--rt-bg-elev,#0d0b08);color:var(--rt-ink,#f6f1e4);box-shadow:0 18px 44px -20px rgba(201,162,39,.38);opacity:0;pointer-events:none;transition:opacity .22s ease,transform .22s ease}' +
      '#rt-human-feedback.rt-hf-show{opacity:1;transform:translateX(-50%) translateY(0)}' +
      '#rt-human-feedback .rt-hf-title{margin:0;font-size:.78rem;font-weight:700;letter-spacing:.02em;color:var(--rt-gold,#f0d36a)}' +
      '#rt-human-feedback .rt-hf-body{margin:.18rem 0 0;font-size:.78rem;line-height:1.4;color:var(--rt-muted,rgba(246,241,228,.68))}' +
      '#rt-human-feedback[data-tone="ok"]{border-color:rgba(52,211,153,.45)}' +
      '#rt-human-feedback[data-tone="ok"] .rt-hf-title{color:#6ee7b7}' +
      '#rt-human-feedback[data-tone="hold"]{border-color:rgba(251,191,36,.5)}' +
      '@media (prefers-reduced-motion:reduce){#rt-human-feedback{transition:none!important;transform:translateX(-50%) translateY(0)}}';
    (document.head || document.documentElement).appendChild(css);
  }

  var hideTimer = null;
  var host = null;
  var lastKey = '';
  var lastAt = 0;
  var pending = null;

  var LANG_NAMES = {
    en: 'English', ar: 'العربية', es: 'Español', fr: 'Français', nl: 'Nederlands',
    de: 'Deutsch', zh: '简体中文', ja: '日本語', pt: 'Português', ru: 'Русский', hi: 'हिन्दी'
  };

  var BY_ID = {
    'new-session-btn': { title: 'New session', body: 'Blank thread on this device. Previous sessions stay in the list.', tone: 'ok' },
    'rename-session-btn': { title: 'Rename', body: 'Name is stored only in this browser.', tone: 'ok' },
    'delete-session-btn': { title: 'Delete session', body: 'That thread is removed from this device only.', tone: 'hold' },
    'export-session-btn': { title: 'Export', body: 'A JSON file downloads to this device. Nothing is uploaded.', tone: 'ok' },
    'export-all-btn': { title: 'Export all', body: 'Every local session downloads as JSON. Nothing leaves for a server.', tone: 'ok' },
    'import-session-btn': { title: 'Import', body: 'Choose a local JSON file. It stays in this browser.', tone: 'ok' },
    'copy-context-btn': { title: 'Copied', body: 'System prompt + history are on the clipboard. Paste into any model.', tone: 'ok' },
    'copy-context-btn-alt': { title: 'Copied', body: 'System prompt + history are on the clipboard. Paste into any model.', tone: 'ok' },
    'voice-settings-btn': { title: 'Voice', body: 'Speech settings stay on this device.', ms: 2500 },
    'voice-save': { title: 'Voice saved', body: 'Playback settings stored locally.', tone: 'ok' },
    'voice-cancel': { title: 'Voice unchanged', body: 'No settings written.', ms: 2200 },
    'unlock-btn': { title: 'Unlock', body: 'Passphrase never leaves this device. Web Crypto stays local.', tone: 'ok' },
    'doc-btn': { title: 'Add a document', body: 'Text is injected into this session only. Not uploaded.', tone: 'ok' },
    'mic-btn': { title: 'Microphone', body: 'Speech-to-text stays in this browser when the OS allows it.' },
    'send-btn': { title: 'Sent', body: 'Stored in this session on this device.', tone: 'ok', ms: 2200 },
    'backend-connect-btn': { title: 'Connect', body: 'Optional local backend. The page still works offline without it.' },
    'backend-disconnect-btn': { title: 'Disconnected', body: 'Back to on-device Lattice Chat.', tone: 'ok' },
    'oo-rerun': { title: 'Re-running tick', body: 'Local ONE Organism pass. No network required.' }
  };

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function ensureHost() {
    if (host && host.parentNode) return host;
    host = document.getElementById('rt-human-feedback');
    if (host) return host;
    if (!document.body) return null;
    host = document.createElement('div');
    host.id = 'rt-human-feedback';
    host.setAttribute('role', 'status');
    host.setAttribute('aria-live', 'polite');
    host.setAttribute('aria-atomic', 'true');
    host.setAttribute('aria-hidden', 'true');
    document.body.appendChild(host);
    return host;
  }

  function hide() {
    if (!host) return;
    host.classList.remove('rt-hf-show');
    host.setAttribute('aria-hidden', 'true');
    host.removeAttribute('data-tone');
  }

  function say(opts) {
    if (typeof opts === 'string') opts = { body: opts };
    opts = opts || {};
    var title = opts.title || '';
    var body = opts.body || opts.message || '';
    if (!title && !body) return;
    var ms = opts.ms == null ? 3000 : Number(opts.ms);
    var tone = opts.tone || 'info';
    var key = tone + '|' + title + '|' + body;
    var now = Date.now();
    if (key === lastKey && now - lastAt < 900) return;
    lastKey = key;
    lastAt = now;

    var el = ensureHost();
    if (!el) {
      pending = opts;
      document.addEventListener('DOMContentLoaded', function () {
        if (pending) { var p = pending; pending = null; say(p); }
      }, { once: true });
      return;
    }
    el.dataset.tone = tone;
    el.innerHTML =
      (title ? '<p class="rt-hf-title">' + esc(title) + '</p>' : '') +
      (body ? '<p class="rt-hf-body">' + esc(body) + '</p>' : '');
    el.setAttribute('aria-hidden', 'false');
    el.classList.add('rt-hf-show');
    clearTimeout(hideTimer);
    if (ms > 0) hideTimer = setTimeout(hide, ms);
  }

  function once(key, opts) {
    try {
      if (sessionStorage.getItem('rt-hf-' + key) === '1') return;
      sessionStorage.setItem('rt-hf-' + key, '1');
    } catch (e) {}
    say(opts);
  }

  function normPath(p) {
    if (!p) return '/';
    p = String(p).split('?')[0].split('#')[0];
    if (p.charAt(p.length - 1) === '/' && p.length > 1) p = p.slice(0, -1);
    if (p === '/index.html') return '/';
    return p || '/';
  }

  function herePath() {
    return normPath(location.pathname || '/');
  }

  function pageName(path) {
    var map = {
      '/': 'Home',
      '/chat.html': 'Lattice Chat',
      '/Launch-Ra-Thor.html': 'Launch',
      '/micro-moment.html': 'Moments',
      '/sovereign-shard.html': 'Shard',
      '/web-forge.html': 'Forge',
      '/contact.html': 'Contact',
      '/privacy.html': 'Privacy',
      '/go-x.html': 'X hop',
      '/thanks.html': 'Thanks',
      '/offline.html': 'Offline'
    };
    return map[normPath(path)] || normPath(path);
  }

  function hostLabel(href) {
    try {
      var u = new URL(href, location.href);
      if (u.protocol === 'mailto:') return u.pathname || 'info@Rathor.ai';
      return u.hostname.replace(/^www\./, '');
    } catch (e) {
      return '';
    }
  }

  function fromOnclick(node) {
    var raw = (node.getAttribute && node.getAttribute('onclick')) || '';
    if (raw.indexOf('exportFullShard') !== -1) return { title: 'Exporting shard', body: 'Local JSON download. Demo state only — not a live node.', tone: 'ok' };
    if (raw.indexOf('performTick') !== -1) return { title: 'Shard tick', body: 'Local demo clock advanced on this device.' };
    if (raw.indexOf('participateQuantumSwarm') !== -1) return { title: 'Swarm demo', body: 'Simulated participation. No network swarm is joined.' };
    if (raw.indexOf('toggleOfflineMode') !== -1) return { title: 'Offline toggle', body: 'Demo flag flipped locally. The site already works without a server.' };
    if (raw.indexOf('reconcileWithConductor') !== -1) return { title: 'Reconcile demo', body: 'Local preview only. No remote conductor is contacted.' };
    if (raw.indexOf('runTolc24Evaluation') !== -1) return { title: 'Governance demo', body: 'Simulated 24-gate pass. Not a PATSAGi warranty.' };
    if (raw.indexOf('clearLog') !== -1) return { title: 'Log cleared', body: 'On-screen history wiped. Device storage is unchanged.', ms: 2200 };
    if (raw.indexOf('applyPreset') !== -1) return { title: 'Preset applied', body: 'Gate weights updated in this tab only.', tone: 'ok' };
    if (raw.indexOf('generateShard') !== -1) return { title: 'Forging locally', body: 'Weights stay in this tab. Not a live conductor node.' };
    if (raw.indexOf('resetWeights') !== -1) return { title: 'Weights reset', body: 'Back to Balanced Genesis on this device.' };
    if (raw.indexOf('exportShard') !== -1) return { title: 'JSON download', body: 'Shard preview saved on this device.', tone: 'ok' };
    if (raw.indexOf('exportHTML') !== -1) return { title: 'HTML download', body: 'Standalone preview file. Still a local demo.', tone: 'ok' };
    return null;
  }

  function infer(node) {
    if (!node || node.nodeType !== 1) return null;
    if (node.closest && node.closest('[data-rt-say-skip]')) return null;
    if (node.hasAttribute && node.hasAttribute('data-rt-say')) {
      return {
        title: node.getAttribute('data-rt-say-title') || '',
        body: node.getAttribute('data-rt-say') || '',
        tone: node.getAttribute('data-rt-say-tone') || 'info',
        ms: parseInt(node.getAttribute('data-rt-say-ms') || '3000', 10)
      };
    }

    var id = node.id || '';
    if (id && BY_ID[id]) return BY_ID[id];

    if (node.classList && node.classList.contains('lang-tab')) {
      var code = node.getAttribute('data-lang') || '';
      return {
        title: LANG_NAMES[code] || code || 'Language',
        body: 'Language stored on this device only. Not sent to rathor.ai.',
        tone: 'ok'
      };
    }

    var oc = fromOnclick(node);
    if (oc) return oc;

    if (node.matches && node.matches('a[href]')) {
      var href = node.getAttribute('href') || '';
      if (href.indexOf('mailto:') === 0) {
        return { title: 'Mail', body: 'Opening your mail app to ' + (href.replace(/^mailto:/, '').split('?')[0] || 'info@Rathor.ai') + '.' };
      }
      if (node.target === '_blank' || /^(https?:)?\/\//i.test(href)) {
        try {
          var dest = new URL(href, location.href);
          if (dest.origin !== location.origin) {
            return {
              title: 'Leaving rathor.ai',
              body: 'Opening ' + hostLabel(href) + ' in a new tab. Separate policy. Not affiliation.'
            };
          }
        } catch (e) {}
      }
      if (normPath(href) === herePath() && node.getAttribute('aria-current') === 'page') {
        return {
          title: 'Already here',
          body: 'You are on ' + pageName(href) + '. Nothing new loaded.',
          ms: 2200
        };
      }
    }

    if (node.classList && node.classList.contains('msg-copy')) {
      return { title: 'Copied', body: 'That message is on the clipboard.', tone: 'ok', ms: 2200 };
    }
    return null;
  }

  function bindClicks() {
    document.addEventListener('click', function (e) {
      var node = e.target;
      if (!node) return;
      if (node.closest) {
        node = node.closest('[data-rt-say], button, a, [onclick], .lang-tab, .msg-copy, input[type="file"]') || node;
      }
      var msg = infer(node);
      if (!msg || (!msg.title && !msg.body)) return;
      say(msg);
    }, true);
  }

  function bindSystem() {
    window.addEventListener('offline', function () {
      say({ title: 'Offline', body: 'No network. Cached pages and on-device tools still work.', tone: 'hold' });
    });
    window.addEventListener('online', function () {
      say({ title: 'Back online', body: 'Network returned. Computations still stay in this browser.', tone: 'ok' });
    });
    document.addEventListener('copy', function () {
      if (Date.now() - lastAt < 900) return;
      say({ title: 'Copied', body: 'On the clipboard of this device.', tone: 'ok', ms: 2200 });
    });
    document.addEventListener('change', function (e) {
      var el = e.target;
      if (!el) return;
      if (el.id === 'session-select') {
        say({ title: 'Session switched', body: 'That thread is local to this device.', tone: 'ok', ms: 2200 });
      }
      if (el.id === 'vid-file' && el.files && el.files[0]) {
        say({ title: 'Video ready', body: el.files[0].name + ' stays in this browser. Nothing is uploaded.', tone: 'ok' });
      }
      if (el.id === 'import-file-input' && el.files && el.files[0]) {
        say({ title: 'Importing', body: el.files[0].name + ' will stay in this browser.', tone: 'ok' });
      }
      if (el.id === 'doc-file-input' && el.files && el.files[0]) {
        say({ title: 'Document attached', body: el.files[0].name + ' is injected locally. Not uploaded.', tone: 'ok' });
      }
    }, true);
  }

  say.__rtFull = true;
  window.rathorSay = say;
  window.rathorSayHide = hide;
  window.rathorSayOnce = once;

  function boot() {
    bindClicks();
    bindSystem();
    if (pending) { var p = pending; pending = null; say(p); }
  }

  if (document.body) boot();
  else document.addEventListener('DOMContentLoaded', boot);
})();
