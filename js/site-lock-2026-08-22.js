/**
 * Ra-Thor site lock 2026-08-22
 * Injects living surfaces / science lattices / indexed map onto rathor.ai
 * Contact: info@Rathor.ai
 */
(function () {
  'use strict';
  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }
  ready(function () {
    var meta = document.querySelector('meta[name="description"]');
    if (meta) {
      meta.setAttribute('content',
        'Mercy-gated AGSi lattice • Workspace 14.15.6 • Lattice Chat v14.18 • TOLC 8 • ONE Organism • Whitepaper v4.1 • Independent project — not affiliated with xAI');
    }
    var kicker = document.querySelector('.kicker');
    if (kicker) {
      kicker.textContent =
        'Powrush-MMO is a working dual-repo world simulator, built with Ra-Thor on Grok. The public monorepo is fully indexed: workspace 14.15.6, AGSi phase, Lattice Chat v14.18, and sister science lattices under TOLC 8.';
    }
    var status = document.querySelector('.lattice-status');
    if (status) {
      status.querySelectorAll('div').forEach(function (n) {
        if (n.textContent && n.textContent.indexOf('Capable') !== -1 && n.textContent.indexOf('AGSi') === -1) {
          n.textContent = 'TOLC 8 • ONE Organism • AGSi Phase • Capable · Bounded · Corrigible';
        }
      });
    }
    if (!document.getElementById('living-surfaces')) {
      var cta = document.getElementById('rathor-v14-cta');
      if (cta) {
        cta.insertAdjacentHTML('afterend',
          '<section id="living-surfaces" class="max-w-4xl mx-auto px-6 pb-8">' +
          '<h2 class="text-2xl sm:text-3xl font-semibold tracking-tight text-center mb-6 text-white">Living Surfaces</h2>' +
          '<div class="grid sm:grid-cols-2 gap-4">' +
          '<a href="/chat.html" class="card-hover block rounded-2xl border border-violet-400/40 bg-violet-950/20 p-5"><p class="font-semibold text-violet-200">Offline Lattice Chat v14.18</p><p class="text-xs text-white/60 mt-2">Multi-session, offline-first, optional passphrase encryption. No account. No backend we control.</p></a>' +
          '<a href="/Launch-Ra-Thor.html" class="card-hover block rounded-2xl border border-emerald-400/40 bg-emerald-950/20 p-5"><p class="font-semibold text-emerald-200">Sovereign Shard v8</p><p class="text-xs text-white/60 mt-2">Dedicated launcher • 11 languages • real-time TOLC 8 verification</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/WHITEPAPER_v4.1.md" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-cyan-400/40 bg-cyan-950/20 p-5"><p class="font-semibold text-cyan-200">Whitepaper v4.1</p><p class="text-xs text-white/60 mt-2">ONE Organism architecture and Powrush-MMO delivery record</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-amber-400/40 bg-amber-950/20 p-5"><p class="font-semibold text-amber-200">Open the monorepo</p><p class="text-xs text-white/60 mt-2">200+ crates • PATSAGi Councils • Lattice Conductor v14 • white-hat fixtures</p></a>' +
          '</div></section>' +
          '<section id="science-lattices" class="max-w-4xl mx-auto px-6 pb-8">' +
          '<h2 class="text-2xl sm:text-3xl font-semibold tracking-tight text-center mb-3 text-white">Science Discovery Lattices</h2>' +
          '<p class="text-center text-sm text-white/60 mb-6 max-w-2xl mx-auto">Sister public repos under the same mercy gates. Research lattices — not product warranties.</p>' +
          '<div class="grid sm:grid-cols-2 gap-4">' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Fusion-Abundance" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-amber-300/30 bg-black p-5"><p class="font-semibold text-amber-200">Fusion Abundance</p><p class="text-xs text-white/60 mt-2">Practical net-positive fusion discovery lattice: materials, tritium breeding, grid integration.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-HighTc-Superconductors" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-cyan-300/30 bg-black p-5"><p class="font-semibold text-cyan-200">Ambient-Pressure High-Tc</p><p class="text-xs text-white/60 mt-2">AI-accelerated materials discovery (REBCO lineage) for zero-loss power and fusion magnets.</p></a>' +
          '</div></section>' +
          '<section id="indexed-lattice" class="max-w-4xl mx-auto px-6 pb-4"><div class="rounded-3xl border border-white/10 bg-zinc-950/50 px-6 py-6">' +
          '<h2 class="text-xl font-semibold text-amber-200 mb-3">Indexed lattice (public)</h2>' +
          '<p class="text-sm text-white/70 leading-relaxed">Full-repo awareness locked 22 Aug 2026. Cargo workspace 14.15.6 is the version source of truth. Lattice Chat surface is v14.18.0. GPU / tick path is documented in v15.34 notes without bumping workspace identity. Cosmic Loop remains mandatory identity.</p>' +
          '</div></section>'
        );
      }
    }
    if (!document.getElementById('faq-q9')) {
      var acc = document.querySelector('.faq-accordion');
      if (acc) {
        acc.insertAdjacentHTML('beforeend',
          '<details><summary class="flex items-center gap-3"><span class="lightning-icon text-amber-300">⚡</span> <span id="faq-q9">What is Lattice Chat?</span> <span class="text-3xl">⚡</span></summary><div class="faq-content mt-6 text-white/80 leading-relaxed">Privacy-first multi-session chat at /chat.html. Offline-first, no login, no backend we control. v14.18.0 adds optional passphrase encryption via Web Crypto.</div></details>' +
          '<details><summary class="flex items-center gap-3"><span class="lightning-icon text-amber-300">⚡</span> <span id="faq-q10">Are the fusion and superconductor repos part of Ra-Thor?</span> <span class="text-3xl">⚡</span></summary><div class="faq-content mt-6 text-white/80 leading-relaxed">Sister public discovery lattices under the same steward, AG-SML v1.0, and TOLC 8. Research surfaces — not a working plant or commercial superconductor product.</div></details>'
        );
      }
    }
    var linkBox = document.querySelector('footer .flex.flex-col');
    if (linkBox && !document.getElementById('rathor-fusion-link')) {
      linkBox.insertAdjacentHTML('beforeend',
        '<a id="rathor-fusion-link" href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Fusion-Abundance" class="hover:text-amber-200">Fusion Abundance lattice</a>' +
        '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-HighTc-Superconductors" class="hover:text-amber-200">High-Tc Superconductors lattice</a>' +
        '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/docs/COMMERCIAL_TARGET_SEGMENTS.md" class="hover:text-amber-200 font-semibold text-cyan-300/90">Commercial target segments</a>'
      );
    }
    document.querySelectorAll('a[href="/chat.html"]').forEach(function (a) {
      if (a.textContent && a.textContent.indexOf('v14.18') === -1) {
        a.innerHTML = a.innerHTML.replace('Offline Lattice Chat', 'Offline Lattice Chat v14.18');
      }
    });
    console.info('[Ra-Thor] site-lock-2026-08-22 applied');
  });
})();
