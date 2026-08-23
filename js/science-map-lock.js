/* science-map-lock.js — restore Home science lattices + FAQ if a thin site-lock ran */
(function () {
  if (typeof document === 'undefined') return;
  function go() {
    var living = document.getElementById('living-surfaces');
    if (living && !document.getElementById('science-lattices')) {
      living.insertAdjacentHTML(
        'afterend',
        '<section id="science-lattices" class="max-w-4xl mx-auto px-6 pb-8">' +
          '<h2 class="text-2xl sm:text-3xl font-semibold tracking-tight text-center mb-3 text-white">Science Discovery Lattices</h2>' +
          '<p class="text-center text-sm text-white/60 mb-6 max-w-2xl mx-auto">Sister public repos under the same mercy gates. Research lattices — not product warranties.</p>' +
          '<div class="grid sm:grid-cols-2 gap-4">' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Fusion-Abundance" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-amber-300/30 bg-black p-5"><p class="font-semibold text-amber-200">Fusion Abundance</p><p class="text-xs text-white/60 mt-2">Practical net-positive fusion discovery. Research surface — not a working plant.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-HighTc-Superconductors" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-cyan-300/30 bg-black p-5"><p class="font-semibold text-cyan-200">Ambient-Pressure High-Tc</p><p class="text-xs text-white/60 mt-2">AI-accelerated materials discovery. Research surface — not a commercial superconductor product.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Ra-Thor-Protein-Molecular-Design" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-emerald-300/30 bg-black p-5"><p class="font-semibold text-emerald-200">Protein & Molecular Design</p><p class="text-xs text-white/60 mt-2">Closed-loop de-novo design under TOLC 8. Research surface — not a therapeutic product.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Mercy-Coordination-Substrate" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-violet-300/30 bg-black p-5"><p class="font-semibold text-violet-200">Mercy Coordination Substrate</p><p class="text-xs text-white/60 mt-2">Post-quantum-resistant coordination architecture. Design lattice — not a launched chain.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/Daedalus-Skin-Eternal-Ark" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-sky-300/30 bg-black p-5"><p class="font-semibold text-sky-200">Daedalus-Skin Eternal Ark</p><p class="text-xs text-white/60 mt-2">Systems-engineering worldship study. Research architecture — not a flying vessel.</p></a>' +
          '<a href="https://github.com/Eternally-Thriving-Grandmasterism/AlphaProMega-Air" target="_blank" rel="noopener" class="card-hover block rounded-2xl border border-lime-300/30 bg-black p-5"><p class="font-semibold text-lime-200">Air Foundation lattices</p><p class="text-xs text-white/60 mt-2">Air + rotorcraft reliability lattices. Engineering research — not certified aircraft.</p></a>' +
          '</div></section>'
      );
    }
    var acc = document.querySelector('.faq-accordion');
    if (acc && !document.getElementById('faq-q9')) {
      acc.insertAdjacentHTML(
        'beforeend',
        '<details><summary class="flex items-center gap-3"><span class="lightning-icon text-amber-300">⚡</span> <span id="faq-q9">What is Lattice Chat?</span> <span class="text-3xl">⚡</span></summary><div id="faq-a9" class="faq-content mt-6 text-white/80 leading-relaxed">Privacy-first multi-session chat at /chat.html. Offline-first, no login, no backend we control. v14.18 adds optional passphrase encryption via Web Crypto.</div></details>' +
          '<details><summary class="flex items-center gap-3"><span class="lightning-icon text-amber-300">⚡</span> <span id="faq-q10">Are the science lattices part of Ra-Thor?</span> <span class="text-3xl">⚡</span></summary><div id="faq-a10" class="faq-content mt-6 text-white/80 leading-relaxed">Sister public discovery lattices under the same steward, AG-SML v1.0, and TOLC 8. Research surfaces — not working plants, certified aircraft, or commercial products.</div></details>' +
          '<details><summary class="flex items-center gap-3"><span class="lightning-icon text-amber-300">⚡</span> <span id="faq-q11">What are the PATSAGi Councils?</span> <span class="text-3xl">⚡</span></summary><div id="faq-a11" class="faq-content mt-6 text-white/80 leading-relaxed">Permanent mercy-gated deliberation councils in the monorepo. They decide under TOLC 8. Architecture and governance posture — not a warranty that every decision is automatically correct.</div></details>'
      );
    }
  }
  if (document.body) go();
  else document.addEventListener('DOMContentLoaded', go);
})();
