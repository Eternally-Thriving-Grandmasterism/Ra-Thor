/* js/week-window.js
 * Single constellation-week stamp for Home + constellation-week.html
 * Workspace 14.15.6 · info@Rathor.ai
 * Fail closed to the stamped window. Never blocks first paint.
 * Optional /js/week-window.json refresh. No GitHub API. Offline-safe.
 */
(function () {
  'use strict';

  var STAMP = {
    start: '2026-09-08',
    end: '2026-09-15',
    source: 'git dcdea8e5f'
  };

  window.RT_WEEK = STAMP;

  function valid(s) {
    return s && /^\d{4}-\d{2}-\d{2}$/.test(s.start) && /^\d{4}-\d{2}-\d{2}$/.test(s.end);
  }

  function paint(s) {
    if (!valid(s)) s = STAMP;
    window.RT_WEEK = s;
    var label = s.start + ' \u2192 ' + s.end;
    var roots = document.querySelectorAll('[data-week-start], [data-week-window]');
    for (var i = 0; i < roots.length; i++) {
      var el = roots[i];
      if (el.hasAttribute('data-week-start')) el.setAttribute('data-week-start', s.start);
      if (el.hasAttribute('data-week-end')) el.setAttribute('data-week-end', s.end);
    }
    var windows = document.querySelectorAll('[data-week-window]');
    for (var j = 0; j < windows.length; j++) {
      windows[j].textContent = label;
    }
    var starts = document.querySelectorAll('[data-week-paint-start]');
    for (var a = 0; a < starts.length; a++) starts[a].textContent = s.start;
    var ends = document.querySelectorAll('[data-week-paint-end]');
    for (var b = 0; b < ends.length; b++) ends[b].textContent = s.end;
  }

  function refreshJson() {
    if (typeof fetch !== 'function') return;
    fetch('/js/week-window.json', { credentials: 'same-origin' })
      .then(function (res) { return res && res.ok ? res.json() : null; })
      .then(function (data) { if (valid(data)) paint(data); })
      .catch(function () {});
  }

  function boot() {
    paint(STAMP);
    refreshJson();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }
})();
