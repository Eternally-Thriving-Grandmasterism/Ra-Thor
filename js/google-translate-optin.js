/* js/google-translate-optin.js
 * Google Translate = new tab, not a widget.
 * Workspace 14.15.6 · info@Rathor.ai
 * Site COEP require-corp blocks translate.google.com inject.
 * Offline packs remain the default. This link leaves the device.
 */
(function () {
  'use strict';
  if (window.__rtGTranslate) return;
  window.__rtGTranslate = true;

  function pack(key, fallback) {
    var lang = currentLang();
    var packs = window.translations || {};
    var t = packs[lang] || packs.en || {};
    var en = packs.en || {};
    var val = (t[key] != null && t[key] !== '') ? t[key] : en[key];
    return val != null ? val : fallback;
  }

  function currentLang() {
    try { return localStorage.getItem('rathor-lang') || 'en'; } catch (e) { return 'en'; }
  }

  function pagePath() {
    var p = location.pathname || '/';
    if (p === '/index.html' || p === '') p = '/';
    return p;
  }

  function googleHref(lang) {
    lang = lang || currentLang() || 'en';
    var u = 'https://rathor.ai' + pagePath();
    return 'https://translate.google.com/translate?sl=en&tl=' +
      encodeURIComponent(lang) +
      '&u=' + encodeURIComponent(u);
  }

  function applyDir(node, text) {
    if (!node) return;
    var rtl = typeof window.rtIsRtlText === 'function' && window.rtIsRtlText(text);
    node.setAttribute('dir', rtl ? 'rtl' : 'ltr');
  }

  function sync() {
    var a = document.getElementById('rt-gtranslate-open');
    var note = document.getElementById('rt-gtranslate-note');
    var wrap = document.getElementById('rt-gtranslate');
    var lang = currentLang();
    if (a) {
      var btn = pack('gTranslateBtn', 'Translate with Google');
      a.textContent = btn;
      a.setAttribute('href', googleHref(lang));
      a.setAttribute('data-i18n', 'gTranslateBtn');
      applyDir(a, btn);
    }
    if (note) {
      var txt = pack(
        'gTranslateNote',
        'Opens Google Translate in a new tab. Needs the network. Not the offline pack.'
      );
      note.textContent = txt;
      note.setAttribute('data-i18n', 'gTranslateNote');
      applyDir(note, txt);
    }
    if (wrap) applyDir(wrap, (note && note.textContent) || '');
  }

  function mount() {
    if (document.getElementById('rt-gtranslate')) {
      sync();
      return;
    }
    var wrap = document.createElement('aside');
    wrap.id = 'rt-gtranslate';
    wrap.className = 'rt-gtranslate';
    wrap.setAttribute('aria-label', 'Google Translate in a new tab');
    wrap.innerHTML =
      '<a class="rt-gtranslate-btn" id="rt-gtranslate-open" target="_blank" rel="noopener"></a>' +
      '<p class="rt-gtranslate-note" id="rt-gtranslate-note"></p>';
    var nav = document.getElementById('rt-family-nav');
    if (nav && nav.parentNode) nav.parentNode.insertBefore(wrap, nav.nextSibling);
    else document.body.insertBefore(wrap, document.body.firstChild);
    sync();
  }

  window.rtGTranslateSync = sync;

  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);
  document.addEventListener('click', function (e) {
    var btn = e.target && e.target.closest && e.target.closest('.lang-tab, [data-lang]');
    if (btn) setTimeout(sync, 0);
  }, true);
})();
