/**
 * Ra-Thor site lock 2026-08-22 (Nth-degree lattice map)
 * Injects living surfaces / science lattices / indexed map onto rathor.ai
 * Uses data-i18n so switchLanguage() can localize injected copy.
 * Cargo truth: workspace 14.15.6 • Lattice Chat v14.18 • Whitepaper v4.1
 * Contact: info@Rathor.ai
 * Independent lattice — not affiliated with, sponsored by, or endorsed by xAI.
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
            '<a href="/Launch-Ra-Thor.html" class="card-hover block rounded-2xl border border-emerald-400/40 bg-emerald-950/20 p-5"><p class="font-semibold text-emerald-200" data-i18n="surfaceShard">Sovereign Shard v8</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceShardNote">Dedicated launcher • 11 languages • real-time TOLC 8 verification</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/WHITEPAPER_v4.1.md" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-cyan-400/40 bg-cyan-950/20 p-5"><p class="font-semibold text-cyan-200" data-i18n="surfacePaper">Whitepaper v4.1</p><p class="text-xs text-white/60 mt-2" data-i18n="surfacePaperNote">ONE Organism architecture and Powrush-MMO delivery record</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-amber-400/40 bg-amber-950/20 p-5"><p class="font-semibold text-amber-200" data-i18n="surfaceRepo">Open the monorepo</p><p class="text-xs text-white/60 mt-2" data-i18n="surfaceRepoNote">200+ crates • PATSAGi Councils • Lattice Conductor v14 • white-hat fixtures</p></a>' +
            '</div></section>' +
            '<section id="science-lattices" class="max-w-4xl mx-auto px-6 pb-8">' +
            '<h2 class="text-2xl sm:text-3xl font-semibold tracking-tight text-center mb-3 text-white" data-i18n="scienceTitle">Science Discovery Lattices</h2>' +
            '<p class="text-center text-sm text-white/60 mb-6 max-w-2xl mx-auto" data-i18n="scienceLead">Sister public repos under the same mercy gates. Research lattices — not product warranties.</p>' +
            '<div class="grid sm:grid-cols-2 gap-4">' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Fusion-Abundance" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-amber-300/30 bg-black p-5"><p class="font-semibold text-amber-200" data-i18n="scienceFusion">Fusion Abundance</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceFusionNote">Practical net-positive fusion discovery lattice: materials, tritium breeding, grid integration.</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-HighTc-Superconductors" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-cyan-300/30 bg-black p-5"><p class="font-semibold text-cyan-200" data-i18n="scienceHtc">Ambient-Pressure High-Tc</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceHtcNote">AI-accelerated materials discovery (REBCO lineage) for zero-loss power and fusion magnets.</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Protein-Molecular-Design" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-emerald-300/30 bg-black p-5"><p class="font-semibold text-emerald-200" data-i18n="scienceProtein">Protein & Molecular Design</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceProteinNote">Closed-loop de-novo design lattice under TOLC 8. Research surface — not a therapeutic product.</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Mercy-Coordination-Substrate" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-violet-300/30 bg-black p-5"><p class="font-semibold text-violet-200" data-i18n="scienceMercy">Mercy Coordination Substrate</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceMercyNote">Post-quantum-resistant coordination architecture. Design lattice — not a launched chain.</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Daedalus-Skin-Eternal-Ark" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-sky-300/30 bg-black p-5"><p class="font-semibold text-sky-200" data-i18n="scienceArk">Daedalus-Skin Eternal Ark</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceArkNote">Systems-engineering worldship study. Research architecture — not a flying vessel.</p></a>' +
            '<a href="https://github.com/Eternally-Thriving-Grandmasterism/AlphaProMega-Air" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-lime-300/30 bg-black p-5"><p class="font-semibold text-lime-200" data-i18n="scienceAir">Air Foundation lattices</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceAirNote">Air + rotorcraft reliability lattices. Engineering research — not certified aircraft.</p></a>' +
            '</div></section>' +
            '<section id="indexed-lattice" class="max-w-4xl mx-auto px-6 pb-4"><div class="rounded-3xl border border-white/10 bg-zinc-950/50 px-6 py-6">' +
            '<h2 class="text-xl font-semibold text-amber-200 mb-3" data-i18n="mapTitle">Indexed lattice (public)</h2>' +
            '<p class="text-sm text-white/70 leading-relaxed" data-i18n="mapBody">Full-repo awareness locked 22 Aug 2026. Cargo workspace 14.15.6 is the version source of truth. Lattice Chat surface is v14.18.x. GPU / tick path is documented in v15.34 notes without bumping workspace identity. Cosmic Loop remains mandatory identity. Capable · bounded · corrigible.</p>' +
            '</div></section>'
        );
      }
    }

    if (!document.getElementById('faq-q9')) {
      var acc = document.querySelector('.faq-accordion');
      if (acc) {
        acc.insertAdjacentHTML(
          'beforeend',
          '<details><summary class="flex items-center gap-3"><span class="lightning-icon text-amber-300">⚡</span> <span id="faq-q9" data-i18n="faqQ9">What is Lattice Chat?</span> <span class="text-3xl">⚡</span></summary><div id="faq-a9" class="faq-content mt-6 text-white/80 leading-relaxed">Privacy-first multi-session chat at /chat.html. Offline-first, no login, no backend we control. v14.18 adds optional passphrase encryption via Web Crypto.</div></details>' +
            '<details><summary class="flex items-center gap-3"><span class="lightning-icon text-amber-300">⚡</span> <span id="faq-q10" data-i18n="faqQ10">Are the science lattices part of Ra-Thor?</span> <span class="text-3xl">⚡</span></summary><div id="faq-a10" class="faq-content mt-6 text-white/80 leading-relaxed">Sister public discovery lattices under the same steward, AG-SML v1.0, and TOLC 8. Research surfaces — not working plants, certified aircraft, or commercial products.</div></details>' +
            '<details><summary class="flex items-center gap-3"><span class="lightning-icon text-amber-300">⚡</span> <span id="faq-q11" data-i18n="faqQ11">What are the PATSAGi Councils?</span> <span class="text-3xl">⚡</span></summary><div id="faq-a11" class="faq-content mt-6 text-white/80 leading-relaxed">Permanent mercy-gated deliberation councils in the monorepo. They decide under TOLC 8. Architecture and governance posture — not a warranty that every decision is automatically correct.</div></details>'
        );
      }
    } else if (!document.getElementById('faq-q11')) {
      var acc2 = document.querySelector('.faq-accordion');
      if (acc2) {
        acc2.insertAdjacentHTML(
          'beforeend',
          '<details><summary class="flex items-center gap-3"><span class="lightning-icon text-amber-300">⚡</span> <span id="faq-q11" data-i18n="faqQ11">What are the PATSAGi Councils?</span> <span class="text-3xl">⚡</span></summary><div id="faq-a11" class="faq-content mt-6 text-white/80 leading-relaxed">Permanent mercy-gated deliberation councils in the monorepo. They decide under TOLC 8. Architecture and governance posture — not a warranty that every decision is automatically correct.</div></details>'
        );
      }
    }

    var linkBox = document.querySelector('footer .flex.flex-col');
    if (linkBox && !document.getElementById('rathor-fusion-link')) {
      linkBox.insertAdjacentHTML(
        'beforeend',
        '<a id="rathor-fusion-link" href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Fusion-Abundance" class="hover:text-amber-200">Fusion Abundance lattice</a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-HighTc-Superconductors" class="hover:text-amber-200">High-Tc Superconductors lattice</a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Protein-Molecular-Design" class="hover:text-amber-200">Protein & Molecular Design</a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Mercy-Coordination-Substrate" class="hover:text-amber-200">Mercy Coordination Substrate</a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Daedalus-Skin-Eternal-Ark" class="hover:text-amber-200">Daedalus-Skin Eternal Ark</a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/AlphaProMega-Air" class="hover:text-amber-200">AlphaProMega Air</a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO" class="hover:text-amber-200">Powrush-MMO</a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/docs/COMMERCIAL_TARGET_SEGMENTS.md" class="hover:text-amber-200 font-semibold text-cyan-300/90">Commercial target segments</a>'
      );
    }

    document.querySelectorAll('a[href="/chat.html"]').forEach(function (a) {
      if (a.textContent && a.textContent.indexOf('v14.18') === -1 && a.textContent.indexOf('Lattice Chat') !== -1) {
        a.innerHTML = a.innerHTML.replace('Offline Lattice Chat', 'Offline Lattice Chat v14.18');
      }
    });

    hookLanguageSwitch();
    window.addEventListener('load', function () {
      hookLanguageSwitch();
      try {
        applyLockI18n(localStorage.getItem('rathor-lang') || 'en');
      } catch (e) {
        applyLockI18n('en');
      }
    });

    console.info('[Ra-Thor] site-lock-2026-08-22 applied (i18n-aware, science map + PATSAGi FAQ)');
  });
})();
