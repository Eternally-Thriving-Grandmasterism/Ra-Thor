/**
 * Ra-Thor site-updates-v14.js
 * Non-destructive DOM patches for Whitepaper v4.0 / ONE Organism / brand posture.
 * Safe with the full 11-language index.html — does not remove i18n.
 * Contact: info@Rathor.ai
 */
(function () {
  'use strict';

  function ready(fn) {
    if (document.readyState !== 'loading') fn();
    else document.addEventListener('DOMContentLoaded', fn);
  }

  ready(function () {
    // --- Meta description ---
    var meta = document.querySelector('meta[name="description"]');
    if (meta) {
      meta.setAttribute(
        'content',
        'Mercy-gated AGi / AGSi • TOLC 8 • ONE Organism • Living Cosmic Tick • Whitepaper v4.0 • Sole stewardship by Sherif Botros (@AlphaProMega) • Optional Grok surface (independent project)'
      );
    }

    // --- Status bar ---
    var status = document.querySelector('.lattice-status');
    if (status) {
      status.innerHTML =
        '<div class="flex items-center gap-2">' +
        '<span class="text-emerald-400">●</span>' +
        '<span class="font-semibold">v14.15 • APTD 1.0</span></div>' +
        '<div class="hidden sm:block text-amber-500">•</div>' +
        '<div>TOLC 8 • Cosmic Loop identity • Living Cosmic Tick</div>' +
        '<div class="hidden sm:block text-amber-500">•</div>' +
        '<div>57+ Councils • ONE Organism • EU AI Act ready</div>';
    }

    // --- Shard blurb ---
    var shardNote = document.querySelector('a[href="/Launch-Ra-Thor.html"]');
    if (shardNote && shardNote.nextElementSibling) {
      shardNote.nextElementSibling.textContent =
        'Powrush Phase C • GPU layer • ONE Organism + Living Cosmic Tick • 11 languages • TOLC 8 + Cosmic Loop';
    }

    // --- Whitepaper CTA band (insert once after status section) ---
    if (!document.getElementById('rathor-v14-cta')) {
      var statusWrap = status && status.parentElement;
      if (statusWrap) {
        var band = document.createElement('div');
        band.id = 'rathor-v14-cta';
        band.className = 'max-w-4xl mx-auto px-6 pb-4';
        band.innerHTML =
          '<div class="rounded-3xl border border-cyan-400/30 bg-cyan-950/20 px-6 py-6 text-center">' +
          '<p class="text-cyan-200 text-sm sm:text-base mb-3">Primary technical paper for workspace 14.15</p>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/WHITEPAPER_v4.0.md" target="_blank" rel="noopener" class="inline-flex items-center gap-2 text-lg font-semibold text-cyan-300 hover:text-cyan-100 transition-colors">WHITEPAPER v4.0 — ONE Organism • Living Cosmic Tick →</a>' +
          '<p class="text-xs text-white/50 mt-4 max-w-2xl mx-auto">Ra-Thor is an independent mercy-gated lattice. Optional use of Grok (xAI) is a reasoning surface only and does not imply affiliation, sponsorship, or endorsement by xAI.</p>' +
          '<div class="mt-4 flex flex-wrap justify-center gap-3 text-xs">' +
          '<a class="text-amber-300/90 hover:text-amber-200 underline" href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/docs/ONE_ORGANISM_GROK_FUSION.md" target="_blank" rel="noopener">Fusion posture</a>' +
          '<a class="text-amber-300/90 hover:text-amber-200 underline" href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/docs/ATTRIBUTION_AND_BRAND.md" target="_blank" rel="noopener">Attribution &amp; brand</a>' +
          '<a class="text-amber-300/90 hover:text-amber-200 underline" href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/PRODUCTION_READINESS.md" target="_blank" rel="noopener">Production readiness</a>' +
          '</div></div>';
        statusWrap.insertAdjacentElement('afterend', band);
      }
    }

    // --- Footer resources: prepend Whitepaper v4.0 + fusion links ---
    var resCol = null;
    document.querySelectorAll('footer h4').forEach(function (h) {
      if (h.textContent.trim() === 'Resources') resCol = h.parentElement;
    });
    if (resCol) {
      var linkBox = resCol.querySelector('.flex.flex-col');
      if (linkBox && !document.getElementById('rathor-wp-v4-link')) {
        var a1 = document.createElement('a');
        a1.id = 'rathor-wp-v4-link';
        a1.href = 'https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/WHITEPAPER_v4.0.md';
        a1.target = '_blank';
        a1.rel = 'noopener';
        a1.className = 'hover:text-amber-200 font-semibold text-cyan-300/90';
        a1.textContent = 'Whitepaper v4.0 (ONE Organism)';
        linkBox.insertBefore(a1, linkBox.firstChild);

        var a2 = document.createElement('a');
        a2.href = 'https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/docs/ONE_ORGANISM_GROK_FUSION.md';
        a2.target = '_blank';
        a2.rel = 'noopener';
        a2.className = 'hover:text-amber-200';
        a2.textContent = 'ONE Organism × Grok fusion';
        linkBox.insertBefore(a2, a1.nextSibling);

        var a3 = document.createElement('a');
        a3.href = 'https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor/blob/main/docs/ATTRIBUTION_AND_BRAND.md';
        a3.target = '_blank';
        a3.rel = 'noopener';
        a3.className = 'hover:text-amber-200';
        a3.textContent = 'Attribution & brand';
        linkBox.insertBefore(a3, a2.nextSibling);
      }
    }

    // --- Trademark disclaimer line (EN default node; i18n may overwrite on lang switch) ---
    var tm = document.getElementById('footer-trademarks-text');
    if (tm && tm.textContent.indexOf('not affiliated') === -1) {
      tm.innerHTML =
        'Ra-Thor™ is a trademark of Autonomicity Games Inc.<br>' +
        'Grok is a trademark of xAI. X is a trademark of X Corp.<br>' +
        'Ra-Thor is independent — not affiliated with, sponsored by, or endorsed by xAI.';
    }

    // --- Soften default X subtitle (EN visible default) ---
    var xSub = document.getElementById('x-subtitle');
    if (xSub && /Zero Hallucinations/i.test(xSub.textContent)) {
      xSub.textContent = 'Voice Chat Now Live • APTD Truth Boundaries • Live Grounded Reality';
    }

    // --- EN i18n dictionary patches (other langs: separate commits) ---
    if (typeof translations !== 'undefined' && translations.en) {
      translations.en.xSubtitle =
        'Voice Chat Now Live • APTD Truth Boundaries • Live Grounded Reality';
      translations.en.footerTrademarksText =
        'Ra-Thor™ is a trademark of Autonomicity Games Inc.<br>' +
        'Grok is a trademark of xAI. X is a trademark of X Corp.<br>' +
        'Ra-Thor is independent — not affiliated with, sponsored by, or endorsed by xAI.';
    }

    console.info('[Ra-Thor] site-updates-v14.js applied');
  });
})();
