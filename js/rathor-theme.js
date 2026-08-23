/* js/rathor-theme.js — dark default, optional light, on-device only
 * Celestial orb: moon while night, sun while day.
 * Contact: info@Rathor.ai
 */
(function () {
  'use strict';
  var KEY = 'rathor-theme';

  var SUN =
    '<svg class="rt-celestial-sun" viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
      '<circle cx="12" cy="12" r="4.1"/>' +
      '<path d="M12 2.6v2.2M12 19.2v2.2M4.8 4.8l1.6 1.6M17.6 17.6l1.6 1.6M2.6 12h2.2M19.2 12h2.2M4.8 19.2l1.6-1.6M17.6 6.4l1.6-1.6" fill="none"/>' +
    '</svg>';

  var MOON =
    '<svg class="rt-celestial-moon" viewBox="0 0 24 24" aria-hidden="true" focusable="false">' +
      '<path d="M15.2 3.4A8.2 8.2 0 1 0 20.6 14 6.4 6.4 0 0 1 15.2 3.4z"/>' +
    '</svg>';

  function read() {
    try { return localStorage.getItem(KEY) || 'dark'; } catch (e) { return 'dark'; }
  }

  function paintButton(btn, next) {
    if (!btn) return;
    btn.classList.add('rt-theme-orb');
    btn.setAttribute('data-rt-theme-toggle', '1');
    btn.setAttribute('type', btn.getAttribute('type') || 'button');
    if (!btn.querySelector('.rt-celestial-sun') || !btn.querySelector('.rt-celestial-moon')) {
      btn.innerHTML = '<span class="rt-celestial-stack" aria-hidden="true">' + SUN + MOON + '</span>';
    }
    var goingLight = next === 'light';
    btn.setAttribute('aria-pressed', goingLight ? 'true' : 'false');
    btn.setAttribute('aria-label', goingLight ? 'Switch to dark mode' : 'Switch to light mode');
    btn.title = goingLight ? 'Night — switch to dark' : 'Day — switch to light';
    btn.dataset.theme = next;
  }

  function apply(mode) {
    var next = mode === 'light' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem(KEY, next); } catch (e) {}
    var buttons = document.querySelectorAll('#rt-theme-toggle, [data-rt-theme-toggle]');
    for (var i = 0; i < buttons.length; i++) paintButton(buttons[i], next);
    var meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.setAttribute('content', next === 'light' ? '#f3ead8' : '#050505');
    document.documentElement.style.colorScheme = next;
    try { window.dispatchEvent(new CustomEvent('rathor-theme-changed', { detail: { theme: next } })); } catch (e) {}
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

  window.rathorApplyTheme = apply;
  window.rathorCurrentTheme = read;

  ensureCss();
  apply(read());
  window.addEventListener('rathor-nav-ready', function () {
    apply(read());
  });
})();
