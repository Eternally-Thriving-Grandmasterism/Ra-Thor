/* js/google-translate-optin.js
 * Google Translate = new tab, not a widget.
 * Workspace 14.15.6 · info@Rathor.ai
 * Offline packs stay the default. This link leaves the device.
 * Do not inject a widget. Do not inject a translate.google.com script.
 * Do not remove COEP from /chat.html.
 */
(function () {
  'use strict';
  if (window.__rtGTranslate) return;
  window.__rtGTranslate = true;

  var TL_KEY = 'rathor-gtranslate-tl';

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

  function rememberTl(lang) {
    if (!lang || lang === 'en') return;
    try { localStorage.setItem(TL_KEY, lang); } catch (e) {}
  }

  function storedTl() {
    try {
      var saved = localStorage.getItem(TL_KEY) || '';
      if (saved && saved !== 'en') return saved;
    } catch (e) {}
    return '';
  }

  function pageUrl() {
    var p = location.pathname || '/';
    if (p === '/index.html' || p === '' || p === '/chat.html') p = '/';
    return 'https://rathor.ai' + p;
  }

  function googleHref(lang) {
    lang = lang || currentLang() || 'en';
    var u = encodeURIComponent(pageUrl());
    if (lang !== 'en') {
      rememberTl(lang);
      return 'https://translate.google.com/translate?sl=en&tl=' +
        encodeURIComponent(lang) +
        '&u=' + u;
    }
    var saved = storedTl();
    if (saved) {
      return 'https://translate.google.com/translate?sl=en&tl=' +
        encodeURIComponent(saved) +
        '&u=' + u;
    }
    return 'https://translate.google.com/website?sl=en&u=' + u;
  }

  function sync() {
    var a = document.getElementById('rt-gtranslate-open');
    var note = document.getElementById('rt-gtranslate-note');
    var lang = currentLang();
    if (a) {
      a.textContent = pack('gTranslateBtn', 'Translate with Google');
      a.setAttribute('href', googleHref(lang));
    }
    if (note) {
      note.textContent = pack(
        'gTranslateNote',
        'Opens Google Translate in a new tab. Needs the network. Not the offline pack.'
      );
    }
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
