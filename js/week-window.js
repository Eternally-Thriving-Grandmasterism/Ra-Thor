/* js/week-window.js
 * Display no-op. Home and constellation-week.html no longer paint a dated window.
 * Workspace 14.15.6 · info@Rathor.ai
 * Kept so existing script tags do not 404. Does not write a date range.
 * Does not fetch /js/week-window.json.
 */
(function () {
  'use strict';

  window.RT_WEEK = null;

  function clearWindows() {
    var windows = document.querySelectorAll('[data-week-window]');
    for (var i = 0; i < windows.length; i++) {
      windows[i].textContent = '';
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', clearWindows);
  } else {
    clearWindows();
  }
})();
