/* js/rathor-unify.js — strip leftover rainbow card gradients + pin RTL scroll
 * Workspace 14.15.6 · info@Rathor.ai
 * 2026-08-23: never let off-canvas skip-links inflate standalone RTL.
 */
(function () {
  'use strict';

  function clipRoot() {
    var root = document.documentElement;
    var body = document.body;
    if (root) {
      root.style.overflowX = 'clip';
      root.style.maxWidth = '100%';
      root.style.width = '100%';
    }
    if (body) {
      body.style.overflowX = 'clip';
      body.style.maxWidth = '100%';
    }
  }

  function tameSkip() {
    var a = document.getElementById('rt-skip-family');
    if (!a || a === document.activeElement) return;
    var left = String(a.style.left || '');
    var pos = String(a.style.position || '');
    var offCanvas = /^-/.test(left) || left.indexOf('999') !== -1 || pos === 'absolute';
    if (!offCanvas) return;
    a.style.position = 'fixed';
    a.style.left = '0';
    a.style.top = '0';
    a.style.right = 'auto';
    a.style.width = '1px';
    a.style.height = '1px';
    a.style.padding = '0';
    a.style.margin = '-1px';
    a.style.overflow = 'hidden';
    a.style.clip = 'rect(0, 0, 0, 0)';
    a.style.clipPath = 'inset(50%)';
    a.style.background = 'transparent';
    a.style.color = 'transparent';
  }

  function pinScroll() {
    try {
      tameSkip();
      clipRoot();
      var y = window.scrollY || window.pageYOffset || 0;
      if (document.documentElement) document.documentElement.scrollLeft = 0;
      if (document.body) document.body.scrollLeft = 0;
      if (window.scrollTo) {
        try { window.scrollTo({ left: 0, top: y, behavior: 'instant' }); }
        catch (e1) { window.scrollTo(0, y); }
      }
    } catch (e) {}
  }

  function applyDir(lang) {
    var rtl = lang === 'ar';
    var root = document.documentElement;
    if (root) {
      root.setAttribute('dir', rtl ? 'rtl' : 'ltr');
      if (lang) root.setAttribute('lang', lang);
    }
    clipRoot();
    tameSkip();
    pinScroll();
    requestAnimationFrame(pinScroll);
    setTimeout(pinScroll, 40);
    setTimeout(pinScroll, 200);
  }

  window.rathorApplyDir = applyDir;
  window.rathorPinScroll = pinScroll;

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
    tameSkip();
    clipRoot();
    pinScroll();
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', run);
  else run();
})();
