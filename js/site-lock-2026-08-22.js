/**
 * Ra-Thor site lock 2026-08-22 (Nth-degree lattice map)
 * 2026-08-24: session cards name Ra-Thor + Grok as a gated working method.
 * 2026-09-01: boot watch-footer-lock — Science watches in footer only.
 * 2026-09-07: expand language buttons + RTL for ar/fa/he.
 * Cargo truth: workspace 14.15.6 • Lattice Chat v14.18
 * Contact: info@Rathor.ai — independent of xAI.
 */
(function () {
  'use strict';

  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  function lockKey(el) {
    var key = el.getAttribute('data-i18n') || el.getAttribute('data-lock-i18n');
    if (key) return key;
    var id = el.id || '';
    if (/^faq-q\d+$/.test(id)) return 'faqQ' + id.slice(5);
    if (/^faq-a\d+$/.test(id)) return 'faqA' + id.slice(5);
    return null;
  }

  function applyLockI18n(lang) {
    try {
      lang = lang || localStorage.getItem('rathor-lang') || 'en';
    } catch (e) {
      lang = 'en';
    }
    var packs = window.translations || {};
    var t = packs[lang] || packs.en;
    var en = packs.en || {};
    if (!t && !en) return;
    document.querySelectorAll('[data-i18n], [data-lock-i18n]').forEach(function (el) {
      var key = lockKey(el);
      if (!key) return;
      var val = (t && t[key] !== undefined) ? t[key] : en[key];
      if (val === undefined) return;
      if (el.hasAttribute('data-i18n-html') || key.indexOf('faqA') === 0 || key.indexOf('footer') === 0) el.innerHTML = val;
      else el.textContent = val;
    });
    var kicker = document.querySelector('.kicker');
    if (kicker && (t.kicker || en.kicker)) kicker.textContent = t.kicker || en.kicker;
    var rtl = (lang === 'ar' || lang === 'fa' || lang === 'he');
    document.documentElement.setAttribute('lang', lang || 'en');
    document.documentElement.setAttribute('dir', rtl ? 'rtl' : 'ltr');
    var faqSection = document.getElementById('faq');
    if (faqSection) {
      if (rtl) { faqSection.classList.add('rtl'); faqSection.setAttribute('dir', 'rtl'); }
      else { faqSection.classList.remove('rtl'); faqSection.setAttribute('dir', 'ltr'); }
    }
  }

  function expandLangButtons() {
    var sel = document.getElementById('lang-selector');
    if (!sel || sel.getAttribute('data-expanded') === '1') return;
    var extra = [
      ['it', 'Italiano'],
      ['ko', '\ud55c\uad6d\uc5b4'],
      ['uk', '\u0423\u043a\u0440\u0430\u0457\u043d\u0441\u044c\u043a\u0430'],
      ['pl', 'Polski'],
      ['tr', 'T\u00fcrk\u00e7e'],
      ['vi', 'Ti\u1ebfng Vi\u1ec7t'],
      ['id', 'Bahasa Indonesia'],
      ['sv', 'Svenska'],
      ['th', '\u0e44\u0e17\u0e22'],
      ['el', '\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac'],
      ['fa', '\u0641\u0627\u0631\u0633\u06cc'],
      ['he', '\u05e2\u05d1\u05e8\u05d9\u05ea']
    ];
    extra.forEach(function (pair) {
      if (sel.querySelector('[data-lang="' + pair[0] + '"]')) return;
      var b = document.createElement('button');
      b.setAttribute('data-lang', pair[0]);
      b.className = 'lang-tab px-5 sm:px-6 py-3 rounded-3xl border border-amber-300 text-amber-300 text-sm sm:text-base';
      b.textContent = pair[1];
      sel.appendChild(b);
    });
    sel.setAttribute('data-expanded', '1');
  }

  function hookLanguageSwitch() {
    if (window.__rathorLockHooked) return;
    window.__rathorLockHooked = true;
    var orig = window.switchLanguage;
    if (typeof orig === 'function') {
      window.switchLanguage = async function (lang) {
        await orig(lang);
        applyLockI18n(lang);
      };
    }
    window.applyLockI18n = applyLockI18n;
  }

  function bootScript(needle, src, immediately) {
    if (document.querySelector('script[src*="' + needle + '"]')) return;
    var s = document.createElement('script');
    s.src = src;
    if (!immediately) s.defer = true;
    (document.head || document.documentElement).appendChild(s);
  }

  function wireSessionCards() {
    var map = {
      'grok-title': 'grokTitle',
      'grok-subtitle': 'grokSubtitle',
      'grok-cta': 'grokCta',
      'x-title': 'xTitle',
      'x-subtitle': 'xSubtitle',
      'x-cta': 'xCta',
      'vibe-title': 'vibeTitle',
      'vibe-subtitle': 'vibeSubtitle',
      'vibe-cta': 'vibeCta'
    };
    Object.keys(map).forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      el.setAttribute('data-i18n', map[id]);
    });
    var grid = document.querySelector('.grid.md\\:grid-cols-3') || document.querySelector('div.grid.max-w-5xl');
    if (!grid) {
      var grids = document.querySelectorAll('div.grid');
      for (var g = 0; g < grids.length; g++) {
        if (grids[g].querySelectorAll('a.card-hover').length >= 3) { grid = grids[g]; break; }
      }
    }
    if (grid) {
      var btns = grid.querySelectorAll('.rt-btn-primary');
      var keys = ['grokCta', 'xCta', 'vibeCta'];
      var ids = ['grok-cta', 'x-cta', 'vibe-cta'];
      for (var i = 0; i < btns.length && i < 3; i++) {
        if (!btns[i].id) btns[i].id = ids[i];
        btns[i].setAttribute('data-i18n', keys[i]);
      }
    }
  }

  ready(function () {
    var meta = document.querySelector('meta[name="description"]');
    if (meta) {
      meta.setAttribute(
        'content',
        'Ra-Thor is an independent software lattice from Autonomicity Games Inc. Two public flagships: this monorepo and Powrush-MMO. Optional Grok sessions work under PATSAGi Councils. Not affiliated with xAI.'
      );
    }

    var kicker = document.querySelector('.kicker');
    if (kicker) kicker.setAttribute('data-i18n', 'kicker');

    var fusion = document.getElementById('fusion-hero');
    if (fusion) fusion.setAttribute('data-i18n', 'fusion');

    wireSessionCards();
    expandLangButtons();

    var status = document.querySelector('.lattice-status');
    if (status) {
      status.querySelectorAll('div').forEach(function (n) {
        if (n.textContent && n.textContent.indexOf('Capable') !== -1) n.setAttribute('data-i18n', 'statusTolc');
        if (n.textContent && n.textContent.indexOf('Powrush') !== -1) n.setAttribute('data-i18n', 'statusPowrush');
      });
    }

    hookLanguageSwitch();
    window.addEventListener('load', function () {
      expandLangButtons();
      hookLanguageSwitch();
      wireSessionCards();
      try { applyLockI18n(localStorage.getItem('rathor-lang') || 'en');
      } catch (e) { applyLockI18n('en'); }
    });

    bootScript('pwa-install', '/js/pwa-install.js', true);
    bootScript('family-nav-2026-08-22', '/js/family-nav-2026-08-22.js');
    bootScript('science-map-lock', '/js/science-map-lock.js');
    bootScript('watch-footer-lock', '/js/watch-footer-lock.js');

    console.info('[Ra-Thor] site-lock-2026-09-07 language expand + RTL ar/fa/he');
  });
})();
