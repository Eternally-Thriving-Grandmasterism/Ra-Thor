/* science-map-lock.js — restore Home science lattices + FAQ if a thin site-lock ran
   2026-08-23e: FAQ Q9–Q11 carry data-i18n; single bolt; re-apply language after inject
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

  function reapplyLang() {
    var lang = currentLang();
    try {
      if (typeof window.applyTranslations === 'function') window.applyTranslations(lang);
      else if (typeof window.applyLockI18n === 'function') window.applyLockI18n(lang);
    } catch (e) {}
  }

  function faqItem(idNum, qKey, qEn, aEn) {
    return (
      '<details><summary>' +
        '<span class="lightning-icon text-amber-300" aria-hidden="true">⚡</span>' +
        '<span id="faq-q' + idNum + '" class="faq-q" data-i18n="' + qKey + '">' + qEn + '</span>' +
      '</summary>' +
      '<div id="faq-a' + idNum + '" data-i18n="faqA' + idNum + '" class="faq-content text-white/80 leading-relaxed">' + aEn + '</div>' +
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
    var sum = qEl.parentNode;
    if (sum && sum.tagName === 'SUMMARY') {
      var extras = sum.querySelectorAll('span.text-3xl');
      for (var i = 0; i < extras.length; i++) extras[i].parentNode.removeChild(extras[i]);
    }
  }

  function go() {
    var living = document.getElementById('living-surfaces');
    if (living && !document.getElementById('science-lattices')) {
      living.insertAdjacentHTML(
        'afterend',
        '<section id="science-lattices" class="max-w-4xl mx-auto px-6 pb-8">' +
          '<h2 class="text-2xl sm:text-3xl font-semibold tracking-tight text-center mb-3 text-white" data-i18n="scienceTitle">Science Discovery Lattices</h2>' +
          '<p class="text-center text-sm text-white/60 mb-6 max-w-2xl mx-auto" data-i18n="scienceLead">Sister public repos under the same mercy gates. Research lattices — not product warranties.</p>' +
          '<div class="grid sm:grid-cols-2 gap-4">' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Fusion-Abundance" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-amber-300/30 bg-black p-5"><p class="font-semibold text-amber-200" data-i18n="scienceFusion">Fusion Abundance</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceFusionNote">Practical net-positive fusion discovery. Research surface — not a working plant.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-HighTc-Superconductors" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-cyan-300/30 bg-black p-5"><p class="font-semibold text-cyan-200" data-i18n="scienceHtc">Ambient-Pressure High-Tc</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceHtcNote">AI-accelerated materials discovery. Research surface — not a commercial superconductor product.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Protein-Molecular-Design" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-emerald-300/30 bg-black p-5"><p class="font-semibold text-emerald-200" data-i18n="scienceProtein">Protein & Molecular Design</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceProteinNote">Closed-loop de-novo design under TOLC 8. Research surface — not a therapeutic product.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Mercy-Coordination-Substrate" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-violet-300/30 bg-black p-5"><p class="font-semibold text-violet-200" data-i18n="scienceMercy">Mercy Coordination Substrate</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceMercyNote">Post-quantum-resistant coordination architecture. Design lattice — not a launched chain.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Daedalus-Skin-Eternal-Ark" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-sky-300/30 bg-black p-5"><p class="font-semibold text-sky-200" data-i18n="scienceArk">Daedalus-Skin Eternal Ark</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceArkNote">Systems-engineering worldship study. Research architecture — not a flying vessel.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/AlphaProMega-Air" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-lime-300/30 bg-black p-5"><p class="font-semibold text-lime-200" data-i18n="scienceAir">Air Foundation lattices</p><p class="text-xs text-white/60 mt-2" data-i18n="scienceAirNote">Air + rotorcraft reliability lattices. Engineering research — not certified aircraft.</p></a>' +
          '</div></section>'
      );
    }

    var acc = document.querySelector('.faq-accordion');
    if (acc && !document.getElementById('faq-q9')) {
      acc.insertAdjacentHTML(
        'beforeend',
        faqItem(9, 'faqQ9', 'What is Lattice Chat?',
          'A privacy-first, multi-session, hybrid-bridge chat surface at /chat.html. Offline-first, no login, no embedded API keys, no backend we control. v14.18.x adds optional passphrase encryption of the session store via Web Crypto (PBKDF2 + AES-GCM).') +
        faqItem(10, 'faqQ10', 'Are the science lattices part of Ra-Thor?',
          'They are sister public discovery lattices under the same steward, AG-SML v1.0, and TOLC 8. Research surfaces — not claims of a working plant, certified aircraft, or commercial product.') +
        faqItem(11, 'faqQ11', 'What are the PATSAGi Councils?',
          'Permanent mercy-gated deliberation councils in the monorepo. They decide under TOLC 8. Architecture and governance posture — not a warranty that every decision is automatically correct.')
      );
    }

    normalizeFaqRow(document.getElementById('faq-q9'));
    normalizeFaqRow(document.getElementById('faq-q10'));
    normalizeFaqRow(document.getElementById('faq-q11'));
    reapplyLang();
  }

  if (document.body) go();
  else document.addEventListener('DOMContentLoaded', go);
  window.addEventListener('load', function () {
    normalizeFaqRow(document.getElementById('faq-q9'));
    normalizeFaqRow(document.getElementById('faq-q10'));
    normalizeFaqRow(document.getElementById('faq-q11'));
    reapplyLang();
  });
})();
