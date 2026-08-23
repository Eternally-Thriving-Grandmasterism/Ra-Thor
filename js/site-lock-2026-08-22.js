/**
 * Ra-Thor site lock 2026-08-22 (Nth-degree lattice map)
 * 2026-08-23: PWA boot + Shard/Launch labels + science-map restore.
 * Cargo truth: workspace 14.15.6 • Lattice Chat v14.18
 * Contact: info@Rathor.ai — independent of xAI.
 */
(function () {
  'use strict';

  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  function applyLockI18n(lang) {
    try {
      lang = lang || localStorage.getItem('rathor-lang') || 'en';
    } catch (e) {
      lang = 'en';
    }
    var packs = window.translations || {};
    var t = packs[lang] || packs.en;
    if (!t) return;
    document.querySelectorAll('#living-surfaces [data-i18n], #science-lattices [data-i18n], #indexed-lattice [data-i18n], #faq-q9, #faq-q10, #faq-q11, [data-lock-i18n]').forEach(function (el) {
      var key = el.getAttribute('data-i18n') || el.getAttribute('data-lock-i18n');
      if (!key || t[key] === undefined) return;
      if (el.hasAttribute('data-i18n-html')) el.innerHTML = t[key];
      else el.textContent = t[key];
    });
    var faqA9 = document.getElementById('faq-a9');
    var faqA10 = document.getElementById('faq-a10');
    var faqA11 = document.getElementById('faq-a11');
    if (faqA9 && t.faqA9) faqA9.innerHTML = t.faqA9;
    if (faqA10 && t.faqA10) faqA10.innerHTML = t.faqA10;
    if (faqA11 && t.faqA11) faqA11.innerHTML = t.faqA11;
    var kicker = document.querySelector('.kicker');
    if (kicker && t.kicker) kicker.textContent = t.kicker;
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

  function bootScript(needle, src) {
    if (document.querySelector('script[src*="' + needle + '"]')) return;
    var s = document.createElement('script');
    s.src = src;
    s.defer = true;
    document.head.appendChild(s);
  }

  ready(function () {
    var meta = document.querySelector('meta[name="description"]');
    if (meta) {
      meta.setAttribute(
        'content',
        'Mercy-gated AGSi lattice • Workspace 14.15.6 • Lattice Chat v14.18 • TOLC 8 • ONE Organism • Whitepaper v4.1 • Independent project — not affiliated with xAI'
      );
    }

    var kicker = document.querySelector('.kicker');
    if (kicker) {
      kicker.setAttribute('data-i18n', 'kicker');
      kicker.textContent =
        'Powrush-MMO is a working dual-repo world simulator, built with Ra-Thor on Grok. The public monorepo is fully indexed: workspace 14.15.6, AGSi phase, Lattice Chat v14.18, and sister science lattices under TOLC 8.';
    }

    var status = document.querySelector('.lattice-status');
    if (status) {
      status.querySelectorAll('div').forEach(function (n) {
        if (n.textContent && n.textContent.indexOf('Capable') !== -1 && n.textContent.indexOf('AGSi') === -1) {
          n.setAttribute('data-i18n', 'statusTolc');
          n.textContent = 'TOLC 8 • ONE Organism • AGSi Phase • Capable · Bounded · Corrigible';
        }
      });
    }

    if (!document.getElementById('living-surfaces')) {
      var cta = document.getElementById('rathor-v14-cta');
      if (cta) {
        cta.insertAdjacentHTML(
          'afterend',
          '<section id="living-surfaces" class="max-w-4xl mx-auto px-6 pb-8">' +
            '<h2 class="text-2xl sm:text-3xl font-semibold tracking-tight text-center mb-6 text-white" data-i18n="surfacesTitle">Living Surfaces</h2>' +
            '<div class="grid sm:grid-cols-2 gap-4">' +
            '<a href="/chat.html" class="card-hover block rounded-2xl border border-violet-400/40 bg-violet-950/20 p-5"><p class="font-semibold text-violet-200" data-i18n="surfaceChat">Offline Lattice Chat v14.18</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceChatNote">Multi-session, offline-first, optional passphrase encryption. No account. No backend we control.</p></a>' +
            '<a href="/Launch-Ra-Thor.html" class="card-hover block rounded-2xl border border-emerald-400/40 bg-emerald-950/20 p-5"><p class="font-semibold text-emerald-200">Launch · Lattice map</p><p class="text-xs text-white/60 mt-2">Thunder lattice map • science cards • Chat / Shard / Forge</p></a>' +
            '<a href="/sovereign-shard.html" class="card-hover block rounded-2xl border border-amber-400/40 bg-amber-950/20 p-5"><p class="font-semibold text-amber-200">Sovereign Shard v8</p><p class="text-xs text-white/60 mt-2">Local TOLC 8 demo — not a live Conductor node.</p></a>' +
            '<a href="/web-forge.html" class="card-hover block rounded-2xl border border-lime-400/40 bg-lime-950/20 p-5"><p class="font-semibold text-lime-200">Web-Forge</p><p class="text-xs text-white/60 mt-2">Local gate-weight generator. Demo surface only.</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/WHITEPAPER_v4.1.md" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-cyan-400/40 bg-cyan-950/20 p-5"><p class="font-semibold text-cyan-200" data-i18n="surfacePaper">Whitepaper v4.1</p><p class="text-xs text-white/60 mt-2" data-i18n="surfacePaperNote">ONE Organism architecture and Powrush-MMO delivery record</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-amber-400/40 bg-amber-950/20 p-5"><p class="font-semibold text-amber-200" data-i18n="surfaceRepo">Open the monorepo</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceRepoNote">200+ crates • PATSAGi Councils • Lattice Conductor v14 • white-hat fixtures</p></a>' +
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
    if (heroGrid && !document.getElementById('rathor-hero-install')) {
      var wrap = document.createElement('div');
      wrap.className = 'mt-3 flex justify-center';
      wrap.innerHTML = '<button type="button" id="rathor-hero-install" class="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-amber-300/40 text-amber-200/90 text-xs hover:bg-amber-400/10 transition-colors"><i class="fa-solid fa-download"></i> Add to Home Screen</button>';
      heroGrid.insertAdjacentElement('afterend', wrap);
      wrap.querySelector('#rathor-hero-install').addEventListener('click', function () {
        if (typeof window.rathorTriggerPWAInstall === 'function') window.rathorTriggerPWAInstall();
      });
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

    bootScript('family-nav-2026-08-22', '/js/family-nav-2026-08-22.js');
    bootScript('pwa-install', '/js/pwa-install.js');
    bootScript('science-map-lock', '/js/science-map-lock.js');

    console.info('[Ra-Thor] site-lock-2026-08-22 applied (PWA + coherent nav + science map restore)');
  });
})();
