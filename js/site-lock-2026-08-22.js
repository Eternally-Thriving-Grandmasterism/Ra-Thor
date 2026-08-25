/**
 * Ra-Thor site lock 2026-08-22 (Nth-degree lattice map)
 * 2026-08-24: session cards name Ra-Thor + Grok as a gated working method.
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

    var status = document.querySelector('.lattice-status');
    if (status) {
      status.querySelectorAll('div').forEach(function (n) {
        if (n.textContent && n.textContent.indexOf('Capable') !== -1) n.setAttribute('data-i18n', 'statusTolc');
        if (n.textContent && n.textContent.indexOf('Powrush') !== -1) n.setAttribute('data-i18n', 'statusPowrush');
      });
    }

    if (!document.getElementById('living-surfaces')) {
      var cta = document.getElementById('rathor-v14-cta');
      if (cta) {
        cta.insertAdjacentHTML(
          'afterend',
          '<section id="living-surfaces" class="max-w-4xl mx-auto px-6 pb-8">' +
            '<h2 class="text-2xl sm:text-3xl font-semibold tracking-tight text-center mb-6 text-white" data-i18n="surfacesTitle">On this device</h2>' +
            '<div class="grid sm:grid-cols-2 gap-4">' +
            '<a href="/chat.html" class="card-hover rt-card-uniform block rounded-2xl p-5"><p class="font-semibold text-amber-100" data-i18n="surfaceChat">Offline Lattice Chat</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceChatNote">Private sessions on this device. Optional passphrase. No account. No backend we control.</p></a>' +
            '<a href="/Launch-Ra-Thor.html" class="card-hover rt-card-uniform block rounded-2xl p-5"><p class="font-semibold text-amber-100" data-i18n="surfaceMap">Launch map</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceMapNote">Public map of Chat, Shard, Forge, and research cards.</p></a>' +
            '<a href="/sovereign-shard.html" class="card-hover rt-card-uniform block rounded-2xl p-5"><p class="font-semibold text-amber-100" data-i18n="surfaceShard">Sovereign Shard</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceShardNote">Local demonstration of the mercy gates on this device.</p></a>' +
            '<a href="/web-forge.html" class="card-hover rt-card-uniform block rounded-2xl p-5"><p class="font-semibold text-amber-100">Web-Forge</p><p class="text-xs text-white/60 mt-2">Local gate-weight generator. Demonstration only.</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/WHITEPAPER_v4.1.md" target="_blank" rel="noopener" class="card-hover rt-card-uniform block rounded-2xl p-5"><p class="font-semibold text-amber-100" data-i18n="surfacePaper">Whitepaper v4.1</p><p class="text-xs text-white/60 mt-2" data-i18n="surfacePaperNote">Architecture and the Powrush-MMO delivery record.</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor" target="_blank" rel="noopener" class="card-hover rt-card-uniform block rounded-2xl p-5"><p class="font-semibold text-amber-100" data-i18n="surfaceRepo">Open the monorepo</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceRepoNote">Source, councils, conductor, and public fixtures.</p></a>' +
            '</div></section>'
        );
      }
    }

    hookLanguageSwitch();
    window.addEventListener('load', function () {
      hookLanguageSwitch();
      wireSessionCards();
      try { applyLockI18n(localStorage.getItem('rathor-lang') || 'en'); }
      catch (e) { applyLockI18n('en'); }
    });

    bootScript('pwa-install', '/js/pwa-install.js', true);
    bootScript('family-nav-2026-08-22', '/js/family-nav-2026-08-22.js');
    bootScript('science-map-lock', '/js/science-map-lock.js');

    console.info('[Ra-Thor] site-lock-2026-08-24 desert-uniform surfaces');
  });
})();
