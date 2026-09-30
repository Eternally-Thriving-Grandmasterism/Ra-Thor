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

  // First non-English language the device prefers ('' if none).
  // Read locally to build the link. Never sent anywhere by this script.
  function deviceTl() {
    var list = [];
    try {
      list = (navigator.languages && navigator.languages.length) ?
        navigator.languages : [navigator.language];
    } catch (e) {}
    for (var i = 0; i < list.length; i++) {
      var tag = String(list[i] || '').toLowerCase();
      var base = tag.split('-')[0];
      if (!base || base === 'en') continue;
      if (base === 'zh') {
        return (tag.indexOf('tw') !== -1 || tag.indexOf('hant') !== -1 ||
          tag.indexOf('hk') !== -1) ? 'zh-TW' : 'zh-CN';
      }
      return base;
    }
    return '';
  }

  function isEnglishTl(tl) {
    tl = String(tl || '').toLowerCase();
    return !tl || tl === 'en' || tl.indexOf('en-') === 0;
  }

  // The page source is English. Google's proxy answers HTTP 400
  // "Can't translate this page" when target == source, and the /website
  // picker takes the target from the UI language (en on English phones).
  // So never build a link whose target is English. Returns '' when there is
  // no non-English target; sync() then shows the language-button hint.
  function googleHref(lang) {
    lang = lang || currentLang() || 'en';
    var tl;
    if (lang !== 'en') {
      rememberTl(lang);
      tl = lang;
    } else {
      tl = storedTl() || deviceTl();
    }
    if (isEnglishTl(tl)) return '';
    return 'https://translate.google.com/translate?sl=en&tl=' +
      encodeURIComponent(tl) + '&u=' + encodeURIComponent(pageUrl());
  }

  function underProxy() {
    try { return /\.translate\.goog$/i.test(location.hostname || ''); } catch (e) { return false; }
  }

  function sync() {
    var a = document.getElementById('rt-gtranslate-open');
    var note = document.getElementById('rt-gtranslate-note');
    var lang = currentLang();
    var href = googleHref(lang);
    if (a) {
      if (href) {
        a.textContent = pack('gTranslateBtn', 'Translate with Google');
        a.setAttribute('href', href);
        a.setAttribute('target', '_blank');
        a.setAttribute('rel', 'noopener');
        a.removeAttribute('data-rt-gtranslate-hint');
      } else {
        // Same pill, same strip height: point at the site's own language buttons.
        a.textContent = pack('gTranslateHint', 'Choose a language on this page');
        if (document.getElementById('lang-selector')) a.setAttribute('href', '#lang-selector');
        else a.removeAttribute('href');
        a.removeAttribute('target');
        a.removeAttribute('rel');
        a.setAttribute('data-rt-gtranslate-hint', '1');
      }
    }
    if (note) {
      note.textContent = pack(
        'gTranslateNote',
        'Opens Google Translate in a new tab. Needs the network. Not the offline pack.'
      );
    }
  }

  function mount() {
    if (underProxy()) return; // already inside Google's proxy: no nested proxy link
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
  window.rtGTranslateHref = googleHref;

  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);
  document.addEventListener('click', function (e) {
    var btn = e.target && e.target.closest && e.target.closest('.lang-tab, [data-lang]');
    if (btn) setTimeout(sync, 0);
  }, true);
})();
