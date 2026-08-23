/* js/rathor-feedback.js — human-readable 3s status across rathor.ai
 * Workspace 14.15.6 · TOLC 8 · info@Rathor.ai
 * On-device only. No analytics. Respects prefers-reduced-motion.
 */
(function () {
  'use strict';

  if (window.rathorSay) return;

  var hideTimer = null;
  var host = null;
  var lastKey = '';
  var lastAt = 0;

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  function ensureHost() {
    if (host && host.parentNode) return host;
    host = document.getElementById('rt-human-feedback');
    if (host) return host;
    if (!document.body) return null;
    host = document.createElement('div');
    host.id = 'rt-human-feedback';
    host.setAttribute('role', 'status');
    host.setAttribute('aria-live', 'polite');
    host.setAttribute('aria-atomic', 'true');
    host.setAttribute('aria-hidden', 'true');
    document.body.appendChild(host);
    return host;
  }

  function hide() {
    if (!host) return;
    host.classList.remove('rt-hf-show');
    host.setAttribute('aria-hidden', 'true');
    host.removeAttribute('data-tone');
  }

  function say(opts) {
    if (typeof opts === 'string') opts = { body: opts };
    opts = opts || {};
    var title = opts.title || '';
    var body = opts.body || opts.message || '';
    if (!title && !body) return;
    var ms = opts.ms == null ? 3000 : Number(opts.ms);
    var tone = opts.tone || 'info';
    var key = tone + '|' + title + '|' + body;
    var now = Date.now();
    if (key === lastKey && now - lastAt < 900) return;
    lastKey = key;
    lastAt = now;

    var el = ensureHost();
    if (!el) {
      document.addEventListener('DOMContentLoaded', function () { say(opts); }, { once: true });
      return;
    }
    el.dataset.tone = tone;
    el.innerHTML =
      (title ? '<p class="rt-hf-title">' + esc(title) + '</p>' : '') +
      (body ? '<p class="rt-hf-body">' + esc(body) + '</p>' : '');
    el.setAttribute('aria-hidden', 'false');
    el.classList.add('rt-hf-show');
    clearTimeout(hideTimer);
    if (ms > 0) hideTimer = setTimeout(hide, ms);
  }

  function bindDeclarative() {
    document.addEventListener('click', function (e) {
      var node = e.target && e.target.closest ? e.target.closest('[data-rt-say]') : null;
      if (!node) return;
      var body = node.getAttribute('data-rt-say');
      var title = node.getAttribute('data-rt-say-title') || '';
      var tone = node.getAttribute('data-rt-say-tone') || 'info';
      var ms = parseInt(node.getAttribute('data-rt-say-ms') || '3000', 10);
      if (body) say({ title: title, body: body, tone: tone, ms: isNaN(ms) ? 3000 : ms });
    }, true);
  }

  window.rathorSay = say;
  window.rathorSayHide = hide;

  if (document.body) bindDeclarative();
  else document.addEventListener('DOMContentLoaded', bindDeclarative);
})();
