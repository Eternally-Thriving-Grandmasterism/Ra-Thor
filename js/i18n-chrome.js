/* js/i18n-chrome.js
 * Offline chrome i18n + per-node dir.
 * Workspace 14.15.6 · info@Rathor.ai
 * Visitor essays apply in js/i18n-essay.js (loaded after this file).
 * Missing key → English. Never blank. Never invent METR.
 * Recent Updates lines are chrome.
 * dir=rtl only when the applied string for that node is actually RTL.
 * Family pills and language tabs stay LTR. html[dir] follows chrome only.
 */
(function (root) {
  'use strict';
  if (root.__rtI18nChrome) return;
  root.__rtI18nChrome = true;

  var RTL_RE = /[\u0590-\u08FF\uFB1D-\uFDFF\uFE70-\uFEFF]/;

  var CHROME = {
    navHome: 1, navChat: 1, navEmploy: 1, navLaunch: 1, navMoments: 1,
    navShard: 1, navForge: 1, navContact: 1, navPrivacy: 1,
    followTitle: 1, followX: 1, followLinkedIn: 1, followFacebook: 1,
    headline: 1, fusion: 1, kicker: 1,
    weekTitle: 1, weekLineWrap: 1, weekLineRa: 1, weekLinePowrush: 1, weekLineResearch: 1, weekMore: 1,
    grokTitle: 1, grokSubtitle: 1, grokCta: 1,
    xTitle: 1, xSubtitle: 1, xCta: 1,
    vibeTitle: 1, vibeSubtitle: 1, vibeCta: 1,
    employTitle: 1, employSubtitle: 1, employCta: 1,
    contactInquiry: 1,
    gTranslateBtn: 1, gTranslateNote: 1,
    installTitle: 1, installStatus: 1, installCta: 1, demoNote: 1
  };

  var PACK_V = '20260923c';

  var NAV_BY_HREF = {
    '/': 'navHome',
    '/index.html': 'navHome',
    '/chat.html': 'navChat',
    '/employ.html': 'navEmploy',
    '/Launch-Ra-Thor.html': 'navLaunch',
    '/micro-moment.html': 'navMoments',
    '/sovereign-shard.html': 'navShard',
    '/web-forge.html': 'navForge',
    '/contact.html': 'navContact',
    '/privacy.html': 'navPrivacy'
  };

  var LANG_TABS = [
    ['en', 'English'], ['ar', 'العربية'], ['es', 'Español'], ['fr', 'Français'],
    ['nl', 'Nederlands'], ['de', 'Deutsch'], ['zh', '简体中文'], ['ja', '日本語'],
    ['pt', 'Português'], ['ru', 'Русский'], ['hi', 'हिन्दी'], ['it', 'Italiano'],
    ['ko', '한국어'], ['uk', 'Українська'], ['pl', 'Polski'], ['tr', 'Türkçe'],
    ['vi', 'Tiếng Việt'], ['id', 'Bahasa Indonesia'], ['sv', 'Svenska'], ['th', 'ไทย'],
    ['el', 'Ελληνικά'], ['fa', 'فارسی'], ['he', 'עברית']
  ];

  function savedLang() {
    try { return localStorage.getItem('rathor-lang') || 'en'; } catch (e) { return 'en'; }
  }

  function stampKnownChrome() {
    var nav = document.getElementById('rt-family-nav');
    if (nav) {
      var links = nav.querySelectorAll('a[href]');
      for (var i = 0; i < links.length; i++) {
        var key = NAV_BY_HREF[links[i].getAttribute('href')];
        if (key) links[i].setAttribute('data-i18n', key);
      }
    }
    var follows = document.querySelectorAll('.rt-follow a[href], a#follow-x, a#follow-linkedin, a#follow-facebook');
    for (var f = 0; f < follows.length; f++) {
      var href = follows[f].getAttribute('href') || '';
      var fkey = '';
      if (href.indexOf('x.com/AlphaProMega') !== -1) fkey = 'followX';
      else if (href.indexOf('linkedin.com/in/sherif-botros') !== -1) fkey = 'followLinkedIn';
      else if (href.indexOf('facebook.com/') !== -1) fkey = 'followFacebook';
      if (fkey) follows[f].setAttribute('data-i18n', fkey);
    }
    var labels = document.querySelectorAll('.rt-follow-label, #follow-title');
    for (var n = 0; n < labels.length; n++) labels[n].setAttribute('data-i18n', 'followTitle');
  }

  function ensureLangSelector() {
    if (!document.body || document.getElementById('lang-selector')) return;
    if (document.body.getAttribute('data-rt-family') === 'off') return;
    var lang = savedLang();
    var sel = document.createElement('div');
    sel.id = 'lang-selector';
    sel.setAttribute('data-rt-lang-injected', '1');
    sel.setAttribute('dir', 'ltr');
    for (var i = 0; i < LANG_TABS.length; i++) {
      var b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('data-lang', LANG_TABS[i][0]);
      b.className = 'lang-tab';
      if (LANG_TABS[i][0] === lang) b.className += ' active';
      b.textContent = LANG_TABS[i][1];
      sel.appendChild(b);
    }
    var nav = document.getElementById('rt-family-nav');
    if (nav && nav.parentNode) nav.parentNode.insertBefore(sel, nav.nextSibling);
    else {
      var main = document.querySelector('main') || document.body;
      main.insertBefore(sel, main.firstChild);
    }
    if (sel.getAttribute('data-rt-chrome-bound') === '1') return;
    sel.setAttribute('data-rt-chrome-bound', '1');
    sel.addEventListener('click', function (e) {
      var btn = e.target && e.target.closest && e.target.closest('[data-lang]');
      if (!btn || !sel.contains(btn)) return;
      var code = btn.getAttribute('data-lang');
      if (typeof root.switchLanguage === 'function') root.switchLanguage(code);
      else loadPackAndApply(code);
    });
  }

  function loadPackAndApply(lang) {
    lang = lang || 'en';
    if (root.translations && root.translations[lang]) {
      applyChromeI18n(lang);
      return;
    }
    var s = document.createElement('script');
    s.src = '/i18n/' + lang + '.js?v=' + PACK_V;
    s.onload = function () { applyChromeI18n(lang); };
    s.onerror = function () { if (lang !== 'en') loadPackAndApply('en'); };
    (document.head || document.documentElement).appendChild(s);
  }

  function isRtlText(s) {
    return typeof s === 'string' && RTL_RE.test(s);
  }

  function isChromeKey(key) {
    return !!(key && CHROME[key]);
  }

  function isLongCopyKey(key) {
    if (!key) return true;
    if (/^faqA\d+$/.test(key)) return true;
    if (/^faqQ\d+$/.test(key)) return true;
    if (key.indexOf('footer') === 0) return true;
    return false;
  }

  function packOf(lang) {
    var packs = root.translations || {};
    return packs[lang] || {};
  }

  function pick(lang, key) {
    var pack = packOf(lang);
    var en = packOf('en');
    var val = pack[key];
    if (val != null && String(val) !== '') {
      return { val: val, fallback: false };
    }
    if (en[key] != null && String(en[key]) !== '') {
      return { val: en[key], fallback: lang !== 'en' };
    }
    return { val: null, fallback: true };
  }

  function setDir(el, rtl, lang) {
    if (!el || !el.setAttribute) return;
    el.setAttribute('dir', rtl ? 'rtl' : 'ltr');
    el.setAttribute('lang', rtl ? (lang || 'ar') : 'en');
  }

  function lockProse() {
    var nodes = document.querySelectorAll('article, .rt-prose, #faq, .faq-accordion, [data-rt-prose]');
    for (var i = 0; i < nodes.length; i++) {
      setDir(nodes[i], false, 'en');
      nodes[i].classList.remove('rtl');
    }
  }

  function applyChromeI18n(lang) {
    lang = lang || 'en';
    stampKnownChrome();
    var applied = 0;
    var rtlHits = 0;
    var fallbackHits = 0;
    var nodes = document.querySelectorAll('[data-i18n], [data-lock-i18n]');
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      var key = el.getAttribute('data-i18n') || el.getAttribute('data-lock-i18n');
      if (!key) continue;
      if (isLongCopyKey(key) || !isChromeKey(key)) {
        setDir(el, false, 'en');
        continue;
      }
      var picked = pick(lang, key);
      if (picked.val == null) continue;
      applied++;
      if (el.hasAttribute('data-i18n-html')) el.innerHTML = picked.val;
      else el.textContent = picked.val;
      var rtl = !picked.fallback && isRtlText(picked.val);
      setDir(el, rtl, lang);
      if (rtl) rtlHits++;
      if (picked.fallback) fallbackHits++;
    }

    var kicker = document.querySelector('.kicker');
    if (kicker) {
      var k = pick(lang, 'kicker');
      if (k.val) {
        kicker.textContent = k.val;
        var kRtl = !k.fallback && isRtlText(k.val);
        setDir(kicker, kRtl, lang);
        applied++;
        if (kRtl) rtlHits++;
        if (k.fallback) fallbackHits++;
      }
    }

    document.querySelectorAll('.lang-tab').forEach(function (b) {
      b.classList.toggle('active', b.getAttribute('data-lang') === lang);
    });

    lockProse();

    var html = document.documentElement;
    var main = document.querySelector('main, #rt-family-main');
    var mostFallback = applied > 0 && fallbackHits >= Math.ceil(applied / 2);
    var mostRtl = applied > 0 && rtlHits > applied / 2 && !mostFallback;
    if (html) {
      html.setAttribute('lang', lang || 'en');
      /* html[dir] follows chrome only when most applied chrome strings are actually RTL. */
      html.setAttribute('dir', mostRtl ? 'rtl' : 'ltr');
    }
    if (main) {
      if (mostFallback || !mostRtl) setDir(main, false, 'en');
      else setDir(main, true, lang);
    }
    lockProse();

    var faq = document.getElementById('faq');
    if (faq) {
      faq.classList.remove('rtl');
      setDir(faq, false, 'en');
    }

    var family = document.getElementById('rt-family-nav');
    if (family) family.setAttribute('dir', 'ltr');
    var langBar = document.getElementById('lang-selector');
    if (langBar) langBar.setAttribute('dir', 'ltr');

    try { localStorage.setItem('rathor-lang', lang); } catch (e) {}
    if (typeof root.rtGTranslateSync === 'function') {
      try { root.rtGTranslateSync(); } catch (e2) {}
    }
    try {
      document.dispatchEvent(new CustomEvent('rt-chrome-i18n', { detail: { lang: lang } }));
    } catch (e3) {}
  }

  function applyNodeDirFromText(el, lang) {
    if (!el) return false;
    var text = el.textContent || '';
    var rtl = isRtlText(text);
    setDir(el, rtl, lang);
    return rtl;
  }

  root.rtIsRtlText = isRtlText;
  root.rtIsChromeKey = isChromeKey;
  root.rtIsLongCopyKey = isLongCopyKey;
  root.rtPickChrome = pick;
  root.rtApplyChromeI18n = applyChromeI18n;
  root.rtApplyNodeDirFromText = applyNodeDirFromText;
  root.rtLockProseDir = lockProse;
  root.rtEnsureLangSelector = ensureLangSelector;

  function bootChrome() {
    ensureLangSelector();
    var lang = savedLang();
    if (root.translations && root.translations[lang]) applyChromeI18n(lang);
    else if (root.translations && root.translations.en) {
      applyChromeI18n('en');
      /* Pack is not on the page yet. Do not replace a saved language with en. */
      if (lang && lang !== 'en') {
        try { localStorage.setItem('rathor-lang', lang); } catch (e) {}
      }
    }
  }
  document.addEventListener('rathor-nav-ready', bootChrome);
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', bootChrome);
  else bootChrome();
})(typeof window !== 'undefined' ? window : this);
