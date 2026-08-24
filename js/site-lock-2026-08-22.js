/**
 * Ra-Thor site lock 2026-08-22 (Nth-degree lattice map)
 * 2026-08-23: real website PWA install (native prompt) — not a homepage tip.
 * 2026-08-23e: FAQ Q9–Q11 i18n by id even if data-i18n was missing on inject.
 * 2026-08-24: PATSAGi homepage copy lock — kicker is i18n source of truth.
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
    document.querySelectorAll('#living-surfaces [data-i18n], #science-lattices [data-i18n], #indexed-lattice [data-i18n], #faq-q9, #faq-q10, #faq-q11, #faq-a9, #faq-a10, #faq-a11, [data-lock-i18n]').forEach(function (el) {
      var key = lockKey(el);
      if (!key) return;
      var val = (t && t[key] !== undefined) ? t[key] : en[key];
      if (val === undefined) return;
      if (el.hasAttribute('data-i18n-html') || key.indexOf('faqA') === 0) el.innerHTML = val;
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

  ready(function () {
    var meta = document.querySelector('meta[name="description"]');
    if (meta) {
      meta.setAttribute(
        'content',
        'Mercy-gated intelligence lattice you can run on your device. Offline Lattice Chat. Living monorepo. Powrush-MMO delivered. TOLC 8. Independent of xAI.'
      );
    }

    var kicker = document.querySelector('.kicker');
    if (kicker) {
      kicker.setAttribute('data-i18n', 'kicker');
    }

    var status = document.querySelector('.lattice-status');
    if (status) {
      status.querySelectorAll('div').forEach(function (n) {
        if (n.textContent && n.textContent.indexOf('Capable') !== -1 && n.textContent.indexOf('AGSi') === -1) {
          n.setAttribute('data-i18n', 'statusTolc');
        }
      });
    }

    var demoNote = document.getElementById('demo-note');
    if (demoNote && (demoNote.textContent || '').indexOf('No installation required') !== -1) {
      demoNote.textContent = 'Install from this website • Offline-ready on your device • No store account we control';
    }

    if (!document.getElementById('living-surfaces')) {
      var cta = document.getElementById('rathor-v14-cta');
      if (cta) {
        cta.insertAdjacentHTML(
          'afterend',
          '<section id="living-surfaces" class="max-w-4xl mx-auto px-6 pb-8">' +
            '<h2 class="text-2xl sm:text-3xl font-semibold tracking-tight text-center mb-6 text-white" data-i18n="surfacesTitle">Start on this device</h2>' +
            '<div class="grid sm:grid-cols-2 gap-4">' +
            '<a href="/chat.html" class="card-hover block rounded-2xl border border-violet-400/40 bg-violet-950/20 p-5"><p class="font-semibold text-violet-200" data-i18n="surfaceChat">Offline Lattice Chat</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceChatNote">Private sessions on this device. Optional passphrase. No account. No backend we control.</p></a>' +
            '<a href="/Launch-Ra-Thor.html" class="card-hover block rounded-2xl border border-emerald-400/40 bg-emerald-950/20 p-5"><p class="font-semibold text-emerald-200" data-i18n="surfaceMap">Launch map</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceMapNote">The public lattice map — Chat, Shard, Forge, and science cards.</p></a>' +
            '<a href="/sovereign-shard.html" class="card-hover block rounded-2xl border border-amber-400/40 bg-amber-950/20 p-5"><p class="font-semibold text-amber-200" data-i18n="surfaceShard">Sovereign Shard</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceShardNote">Local mercy-gated demo. Explore the gates on your phone or laptop.</p></a>' +
            '<a href="/web-forge.html" class="card-hover block rounded-2xl border border-lime-400/40 bg-lime-950/20 p-5"><p class="font-semibold text-lime-200">Web-Forge</p><p class="text-xs text-white/60 mt-2">Local gate-weight generator. Demo surface only.</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/WHITEPAPER_v4.1.md" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-cyan-400/40 bg-cyan-950/20 p-5"><p class="font-semibold text-cyan-200" data-i18n="surfacePaper">Whitepaper v4.1</p><p class="text-xs text-white/60 mt-2" data-i18n="surfacePaperNote">Architecture and the Powrush-MMO delivery record.</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-amber-400/40 bg-amber-950/20 p-5"><p class="font-semibold text-amber-200" data-i18n="surfaceRepo">Open the monorepo</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceRepoNote">Source, PATSAGi Councils, Lattice Conductor, and white-hat fixtures.</p></a>' +
            '</div></section>'
        );
      }
    }

    document.querySelectorAll('a[href="/Launch-Ra-Thor.html"]').forEach(function (a) {
      var label = (a.textContent || '').replace(/\s+/g, ' ').trim();
      if (label.indexOf('Sovereign Shard') !== -1) {
        a.setAttribute('href', '/sovereign-shard.html');
        a.removeAttribute('target');
        a.innerHTML = '<i class="fa-solid fa-gem"></i> Open Shard v8';
      }
    });

    var heroGrid = document.querySelector('.mt-10.max-w-2xl') || document.querySelector('.mt-10.max-w-3xl');
    if (heroGrid && !document.getElementById('rathor-pwa-slot') && !document.getElementById('rathor-hero-install')) {
      var wrap = document.createElement('div');
      wrap.id = 'rathor-pwa-slot';
      wrap.className = 'mt-5 max-w-xl mx-auto';
      wrap.innerHTML =
        '<div class="rounded-2xl border border-sky-300/40 bg-gradient-to-br from-slate-950 via-sky-950 to-cyan-900 px-5 py-4 text-center">' +
        '  <p class="text-sky-200 font-semibold text-sm sm:text-base">Install Ra-Thor from this website</p>' +
        '  <p id="rathor-pwa-status" class="text-[11px] text-white/55 mt-1 leading-relaxed">Native app install. No store. Offline-ready.</p>' +
        '  <button type="button" id="rathor-hero-install" data-rt-pwa-install ' +
        '    class="mt-3 inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-sky-300 text-black text-sm font-semibold hover:bg-sky-200 transition-colors">' +
        '    <i class="fa-solid fa-download"></i> Install on this device' +
        '  </button>' +
        '</div>';
      heroGrid.insertAdjacentElement('afterend', wrap);
      wrap.querySelector('#rathor-hero-install').addEventListener('click', function () {
        if (typeof window.rathorTriggerPWAInstall === 'function') window.rathorTriggerPWAInstall();
      });
    } else if (document.getElementById('rathor-hero-install') && !document.getElementById('rathor-hero-install').getAttribute('data-rt-pwa-install')) {
      var old = document.getElementById('rathor-hero-install');
      old.setAttribute('data-rt-pwa-install', '');
      old.innerHTML = '<i class="fa-solid fa-download"></i> <span data-i18n="installCta">Install Ra-Thor</span>';
    }

    hookLanguageSwitch();
    window.addEventListener('load', function () {
      hookLanguageSwitch();
      try {
        applyLockI18n(localStorage.getItem('rathor-lang') || 'en');
      } catch (e) {
        applyLockI18n('en');
      }
    });

    bootScript('pwa-install', '/js/pwa-install.js', true);
    bootScript('family-nav-2026-08-22', '/js/family-nav-2026-08-22.js');
    bootScript('science-map-lock', '/js/science-map-lock.js');

    console.info('[Ra-Thor] site-lock-2026-08-24 applied (homepage copy lock + real PWA install + science map)');
  });
})();
