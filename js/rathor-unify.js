/* js/rathor-unify.js — strip leftover rainbow card gradients on public surfaces */
(function () {
  'use strict';
  function run() {
    var nodes = document.querySelectorAll('a, button, div, section');
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      var cls = el.className;
      if (typeof cls !== 'string' || !cls) continue;
      if (/from-slate-950|from-blue-900|from-purple-900|to-cyan-900|to-indigo-900|via-sky-950|via-violet-900/.test(cls)) {
        el.className = cls
          .replace(/bg-gradient-to-br/g, '')
          .replace(/from-\S+/g, '')
          .replace(/via-\S+/g, '')
          .replace(/to-\S+/g, '')
          .replace(/border-(sky|cyan|purple|violet|emerald)-\S+/g, 'border-amber-300/30')
          + ' rt-card-uniform';
      }
    }
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run);
  else run();
})();
