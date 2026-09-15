/* js/google-translate-optin.js
 * Click-only Google Translate for rathor.ai family pages.
 * Workspace 14.15.6 · info@Rathor.ai
 * Never loads translate.google.com until the visitor asks.
 * Offline packs remain the default. This widget is Google, on the network.
 */
(function () {
  'use strict';
  if (window.__rtGTranslate) return;
  window.__rtGTranslate = true;

  var loaded = false;
  var failed = false;

  function pack(key, fallback) {
    var lang = 'en';
    try { lang = localStorage.getItem('rathor-lang') || 'en'; } catch (e) {}
    var packs = window.translations || {};
    var t = packs[lang] || packs.en || {};
    var en = packs.en || {};
    var val = (t[key] != null && t[key] !== '') ? t[key] : en[key];
    return val != null ? val : fallback;
  }

  function offline() {
    try { return navigator.onLine === false; } catch (e) { return false; }
  }

  function noteEl() { return document.getElementById('rt-gtranslate-note'); }

  function setNote(text) {
    var el = noteEl();
    if (el) el.textContent = text;
  }

  function mount() {
    if (document.getElementById('rt-gtranslate')) return;
    var wrap = document.createElement('aside');
    wrap.id = 'rt-gtranslate';
    wrap.className = 'rt-gtranslate';
    wrap.setAttribute('aria-label', 'Google Translate opt-in');
    wrap.innerHTML =
      '<button type="button" class="rt-gtranslate-btn" id="rt-gtranslate-open"></button>' +
      '<p class="rt-gtranslate-note" id="rt-gtranslate-note"></p>' +
      '<div id="google_translate_element" hidden></div>';
    var nav = document.getElementById('rt-family-nav');
    if (nav && nav.parentNode) nav.parentNode.insertBefore(wrap, nav.nextSibling);
    else document.body.insertBefore(wrap, document.body.firstChild);

    var btn = document.getElementById('rt-gtranslate-open');
    btn.textContent = pack('gTranslateBtn', 'Translate with Google');
    setNote(pack('gTranslateNote', 'Uses Google. Needs the network. This is not the offline language pack.'));
    btn.addEventListener('click', open);
  }

  function open() {
    var slot = document.getElementById('google_translate_element');
    if (offline()) {
      setNote(pack('gTranslateOffline', 'Google Translate needs the network. Offline language packs on this page still work.'));
      return;
    }
    if (failed) {
      setNote(pack('gTranslateBlocked', 'Google Translate is blocked or unavailable. Offline packs still work.'));
      return;
    }
    if (loaded) {
      if (slot) slot.hidden = false;
      return;
    }
    setNote(pack('gTranslateLoading', 'Loading Google Translate. This leaves the device and is not the offline pack.'));
    if (slot) slot.hidden = false;

    window.googleTranslateElementInit = function () {
      try {
        if (!window.google || !google.translate || !google.translate.TranslateElement) {
          throw new Error('missing google.translate');
        }
        new google.translate.TranslateElement({
          pageLanguage: 'en',
          autoDisplay: false
        }, 'google_translate_element');
        loaded = true;
        setNote(pack('gTranslateReady', 'This widget is Google’s. It needs the network. The offline pack remains the default.'));
      } catch (e) {
        failed = true;
        setNote(pack('gTranslateBlocked', 'Google Translate failed to start. Offline packs still work.'));
      }
    };

    var s = document.createElement('script');
    s.src = 'https://translate.google.com/translate_a/element.js?cb=googleTranslateElementInit';
    s.async = true;
    s.onerror = function () {
      failed = true;
      setNote(pack('gTranslateBlocked', 'Could not reach Google (offline or blocked). Offline language packs still work.'));
    };
    document.head.appendChild(s);
  }

  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);
})();
