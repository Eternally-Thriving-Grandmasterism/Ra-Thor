/* js/i18n-chrome.js
 * Offline chrome-only i18n + per-node dir.
 * Workspace 14.15.6 · info@Rathor.ai
 * Long copy (Employ body, Privacy body, FAQ answers, week research footnote)
 * stays English in git. Missing key → English. Never blank. Never invent METR.
 * dir=rtl only when the applied string for that node is actually RTL.
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
    weekTitle: 1, weekLineRa: 1, weekLinePowrush: 1, weekMore: 1,
    grokTitle: 1, grokSubtitle: 1, grokCta: 1,
    xTitle: 1, xSubtitle: 1, xCta: 1,
    vibeTitle: 1, vibeSubtitle: 1, vibeCta: 1,
    employTitle: 1, employSubtitle: 1, employCta: 1,
    contactInquiry: 1,
    gTranslateBtn: 1, gTranslateNote: 1,
    installTitle: 1, installStatus: 1, installCta: 1, demoNote: 1
  };

  function isRtlText(s) {
    return typeof s === 'string' && RTL_RE.test(s);
  }

  function isChromeKey(key) {
    return !!(key && CHROME[key]);
  }

  function isLongCopyKey(key) {
    if (!key) return true;
    if (key === 'weekLineResearch') return true;
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
})(typeof window !== 'undefined' ? window : this);
