/* js/rathor-theme.js — dark default, optional light, on-device only */
(function () {
  'use strict';
  var KEY = 'rathor-theme';

  function read() {
    try { return localStorage.getItem(KEY) || 'dark'; } catch (e) { return 'dark'; }
  }

  function apply(mode) {
    var next = mode === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem(KEY, next); } catch (e) {}
    var buttons = document.querySelectorAll('#rt-theme-toggle, [data-rt-theme-toggle]');
    for (var i = 0; i < buttons.length; i++) {
      var btn = buttons[i];
      btn.textContent = next === 'light' ? 'Dark' : 'Light';
      btn.setAttribute('aria-pressed', next === 'light' ? 'true' : 'false');
      btn.title = next === 'light' ? 'Switch to dark' : 'Switch to light';
    }
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', next === 'light' ? '#f3ead8' : '#050505');
    document.documentElement.style.colorScheme = next;
  }

  function ensureCss() {
    if (document.querySelector('link[href*="rathor-theme.css"]')) return;
    var l = document.createElement('link');
    l.rel = 'stylesheet';
    l.href = '/css/rathor-theme.css';
    document.head.appendChild(l);
  }

  window.rathorToggleTheme = function () {
    apply(read() === 'light' ? 'dark' : 'light');
  };

  ensureCss();
  apply(read());
  window.addEventListener('rathor-nav-ready', function () {
    apply(read());
  });
})();
