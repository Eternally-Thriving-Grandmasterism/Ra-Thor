/* science-map-lock.js — Home science lattices + public works + FAQ
   2026-08-24: official corporate copy; FAQ 9–25; session honesty
   Contact: info@Rathor.ai
*/
(function () {
  if (typeof document === 'undefined') return;

  function currentLang() {
    try {
      return localStorage.getItem('rathor-lang') || (navigator.language || 'en').split('-')[0] || 'en';
    } catch (e) {
      return 'en';
    }
  }

  function pack() {
    var lang = currentLang();
    var all = window.translations || {};
    return all[lang] || all.en || {};
  }

  function reapplyLang() {
    var lang = currentLang();
    try {
      if (typeof window.applyTranslations === 'function') window.applyTranslations(lang);
      else if (typeof window.applyLockI18n === 'function') window.applyLockI18n(lang);
    } catch (e) {}
  }

  function faqItem(idNum, qEn, aEn) {
    return (
      '<details><summary>' +
        '<span class="lightning-icon text-amber-300" aria-hidden="true">⚡</span>' +
        '<span id="faq-q' + idNum + '" class="faq-q" data-i18n="faqQ' + idNum + '">' + qEn + '</span>' +
      '</summary>' +
      '<div id="faq-a' + idNum + '" data-i18n="faqA' + idNum + '" data-i18n-html class="faq-content text-white/80 leading-relaxed">' + aEn + '</div>' +
      '</details>'
    );
  }

  function normalizeFaqRow(qEl) {
    if (!qEl) return;
    var id = qEl.id || '';
    var n = id.replace('faq-q', '');
    if (!qEl.getAttribute('data-i18n')) qEl.setAttribute('data-i18n', 'faqQ' + n);
    if (qEl.className.indexOf('faq-q') === -1) qEl.className = (qEl.className + ' faq-q').trim();
    var aEl = document.getElementById('faq-a' + n);
    if (aEl && !aEl.getAttribute('data-i18n')) aEl.setAttribute('data-i18n', 'faqA' + n);
  }

  function workCard(href, titleKey, titleEn, noteKey, noteEn, border) {
    return (
      '<a href="' + href + '" target="_blank" rel="noopener" class="card-hover block rounded-2xl border ' + border + ' bg-black p-5">' +
        '<p class="font-semibold text-amber-100" data-i18n="' + titleKey + '">' + titleEn + '</p>' +
        '<p class="text-xs text-white/60 mt-2" data-i18n="' + noteKey + '">' + noteEn + '</p>' +
      '</a>'
    );
  }

  function injectFaq() {
    var acc = document.querySelector('.faq-accordion');
    if (!acc) return;
    var t = pack();
    var en = (window.translations && window.translations.en) || t;
    for (var n = 9; n <= 25; n++) {
      if (document.getElementById('faq-q' + n)) continue;
      var q = t['faqQ' + n] || en['faqQ' + n] || '';
      var a = t['faqA' + n] || en['faqA' + n] || '';
      if (!q) continue;
      acc.insertAdjacentHTML('beforeend', faqItem(n, q, a));
    }
    for (var i = 9; i <= 25; i++) normalizeFaqRow(document.getElementById('faq-q' + i));
  }

  function go() {
    var living = document.getElementById('living-surfaces');
    var insertAfter = living || document.getElementById('rathor-v14-cta');

    if (insertAfter && !document.getElementById('public-works')) {
      insertAfter.insertAdjacentHTML(
        'afterend',
        '<section id="public-works" class="max-w-4xl mx-auto px-6 pb-8">' +
          '<h2 class="text-2xl sm:text-3xl font-semibold tracking-tight text-center mb-3 text-white" data-i18n="worksTitle">Public works</h2>' +
          '<p class="text-center text-sm text-white/60 mb-6 max-w-2xl mx-auto" data-i18n="worksLead">A constellation around two flagships. Linked repositories are public source. Applied science and aviation repositories are research and engineering studies — not plants, drugs, chains, ships, or certified aircraft.</p>' +
          '<p class="text-xs uppercase tracking-wide text-amber-300/80 mb-3" data-i18n="worksFlagships">Flagships</p>' +
          '<div class="grid sm:grid-cols-2 gap-4 mb-6">' +
          workCard('https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor', 'workRaThor', 'Ra-Thor', 'workRaThorNote', 'Core monorepo: mercy gates, PATSAGi Councils, Lattice Conductor, Offline Lattice Chat.', 'border-amber-300/40') +
          workCard('https://github.com/Eternally-Thriving-Grandmasterism/Powrush-MMO', 'workPowrush', 'Powrush-MMO', 'workPowrushNote', 'Dual-repo world simulator. Software — not a live massively-multiplayer service promise.', 'border-amber-300/40') +
          '</div>' +
          '<p class="text-xs uppercase tracking-wide text-amber-300/80 mb-3" data-i18n="worksCore">Core stack</p>' +
          '<div class="grid sm:grid-cols-2 gap-4 mb-6">' +
          workCard('https://github.com/Eternally-Thriving-Grandmasterism/MercyOS', 'workMercyOS', 'MercyOS', 'workMercyOSNote', 'Operating-system expression of the same gated lattice. Public source. Not a consumer OS product.', 'border-white/20') +
          workCard('https://github.com/Eternally-Thriving-Grandmasterism/Mercy-Coordination-Substrate', 'workSubstrate', 'Mercy Coordination Substrate', 'workSubstrateNote', 'Design lattice for post-quantum-oriented coordination. Architecture — not a launched, audited chain.', 'border-white/20') +
          workCard('https://github.com/Eternally-Thriving-Grandmasterism/NEXi', 'workNexi', 'NEXi', 'workNexiNote', 'Supporting lattice crate. Public source in the same family.', 'border-white/20') +
          workCard('https://github.com/Eternally-Thriving-Grandmasterism/ESAO', 'workEsao', 'ESAO', 'workEsaoNote', 'Supporting engine in the same family. Public source.', 'border-white/20') +
          workCard('https://github.com/Eternally-Thriving-Grandmasterism/rathor-grok-proxy', 'workProxy', 'rathor-grok-proxy', 'workProxyNote', 'Practical integration surface for optional Grok sessions. Demonstration plumbing — not an xAI product.', 'border-white/20') +
          workCard('https://github.com/Eternally-Thriving-Grandmasterism', 'worksOrg', 'Full organization on GitHub →', 'worksRelated', 'Related public repositories under the same steward.', 'border-white/20') +
          '</div></section>'
      );
    }

    var works = document.getElementById('public-works');
    var scienceAnchor = works || living;
    if (scienceAnchor && !document.getElementById('science-lattices')) {
      scienceAnchor.insertAdjacentHTML(
        'afterend',
        '<section id="science-lattices" class="max-w-4xl mx-auto px-6 pb-8">' +
          '<h2 class="text-2xl sm:text-3xl font-semibold tracking-tight text-center mb-3 text-white" data-i18n="scienceTitle">Applied research lattices</h2>' +
          '<p class="text-center text-sm text-white/60 mb-6 max-w-2xl mx-auto" data-i18n="scienceLead">Sister public repositories under the same steward, license, and gates. Open research and systems-engineering studies — not plants, drugs, chains, ships, or certified aircraft.</p>' +
          '<div class="grid sm:grid-cols-2 gap-4">' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Fusion-Abundance" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-amber-300/30 bg-black p-5"><p class="font-semibold text-amber-200" data-i18n="scienceFusion">Fusion Abundance</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceFusionNote">Public research lattice on fusion pathways. Study surface — not a working plant.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-HighTc-Superconductors" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-cyan-300/30 bg-black p-5"><p class="font-semibold text-cyan-200" data-i18n="scienceHtc">Ambient-Pressure High-Tc</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceHtcNote">Public research lattice on high-Tc candidates. Study surface — not a commercial superconductor.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Protein-Molecular-Design" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-emerald-300/30 bg-black p-5"><p class="font-semibold text-emerald-200" data-i18n="scienceProtein">Protein & Molecular Design</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceProteinNote">Public research lattice for de-novo design. Not a therapeutic. Not a wet-lab product.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Daedalus-Skin-Eternal-Ark" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-sky-300/30 bg-black p-5"><p class="font-semibold text-sky-200" data-i18n="scienceArk">Daedalus-Skin Eternal Ark</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceArkNote">Public systems-engineering worldship study. Architecture — not a flying vessel.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/AlphaProMega-Air" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-lime-300/30 bg-black p-5"><p class="font-semibold text-lime-200" data-i18n="scienceAir">Air Foundation lattices</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceAirNote">Public engineering research on airframe and rotorcraft reliability. Not certified aircraft.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/AlphaProMega-Rotorcraft-Immortality" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-lime-300/30 bg-black p-5"><p class="font-semibold text-lime-200">Rotorcraft reliability</p><p class="text-xs text-white/60 mt-2">Public engineering study of rotary-wing single-point-of-failure reduction. Not certified aircraft.</p></a>' +
          '</div></section>'
      );
    }

    injectFaq();
    reapplyLang();
  }

  if (document.body) go();
  else document.addEventListener('DOMContentLoaded', go);
  window.addEventListener('load', function () {
    injectFaq();
    reapplyLang();
  });
})();
