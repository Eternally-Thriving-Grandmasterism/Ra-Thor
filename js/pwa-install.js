/* js/pwa-install.js — Aug-8 lattice Chrome install, restored 2026-08-23
 * Capture BIP early. Register SW immediately. Wait for the event, then event.prompt().
 * Do not show the menu-coach card until the wait expires.
 * Workspace 14.15.6 · TOLC 8 · info@Rathor.ai
 */
(function () {
  'use strict';

  var DISMISS_KEY = 'rathor-pwa-install-dismissed';
  var DISMISS_DAYS = 14;
  var WAIT_MS = 12000;
  window.__rtPwa = window.__rtPwa || { ev: null };
  var deferredPrompt = window.__rtPwa.ev || null;
  var bannerEl = null;
  var waiting = false;

  try {
    var q = new URLSearchParams(location.search);
    if (q.has('pwa') || q.has('install') || q.get('source') === 'pwa') {
      localStorage.removeItem(DISMISS_KEY);
    }
  } catch (e) {}

  function ua() { return navigator.userAgent || ''; }
  function isStandalone() {
    return (
      (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches) ||
      window.navigator.standalone === true ||
      (document.referrer && document.referrer.indexOf('android-app://') !== -1)
    );
  }
  function isIos() { return /iphone|ipad|ipod/i.test(ua()); }
  function isEmbedded() {
    var s = ua();
    if (/\bwv\b/.test(s) || /WebView/i.test(s)) return true;
    if (/Grok\/|TwitterAndroid|FBAN|FBAV|Instagram|Line\/|WhatsApp/i.test(s)) return true;
    try { if (window.self !== window.top) return true; } catch (e) { return true; }
    return false;
  }
  function wasDismissedRecently() {
    try {
      var raw = localStorage.getItem(DISMISS_KEY);
      if (!raw) return false;
      var ts = parseInt(raw, 10);
      return !isNaN(ts) && Date.now() - ts < DISMISS_DAYS * 24 * 60 * 60 * 1000;
    } catch (e) { return false; }
  }
  function markDismissed() {
    try { localStorage.setItem(DISMISS_KEY, String(Date.now())); } catch (e) {}
  }
  function chromeIntent() {
    return 'intent://rathor.ai/#Intent;scheme=https;package=com.android.chrome;S.browser_fallback_url=https%3A%2F%2Frathor.ai%2F;end';
  }
  function setStatus(text) {
    var el = document.getElementById('rathor-pwa-status');
    if (el && text) el.textContent = text;
  }

  function registerServiceWorker() {
    if (!('serviceWorker' in navigator)) return Promise.resolve();
    return navigator.serviceWorker.register('/sw.js', { scope: '/', updateViaCache: 'none' }).then(function (reg) {
      try { if (reg && reg.update) reg.update(); } catch (e) {}
      return reg;
    }).catch(function () { return null; });
  }

  function hideBanner() {
    if (!bannerEl) return;
    bannerEl.style.opacity = '0';
    var el = bannerEl;
    bannerEl = null;
    setTimeout(function () { if (el && el.parentNode) el.parentNode.removeChild(el); }, 280);
  }

  function setButtons(label, disabled) {
    var nodes = document.querySelectorAll('#rathor-hero-install, #rathor-lattice-install, #rathor-pwa-install-btn, [data-rt-pwa-install]');
    for (var i = 0; i < nodes.length; i++) {
      nodes[i].disabled = !!disabled;
      if (nodes[i].id === 'rathor-pwa-install-btn') nodes[i].textContent = label;
      else nodes[i].innerHTML = '<i class="fa-solid fa-download"></i> ' + label;
    }
  }

  function firePrompt() {
    var ev = deferredPrompt || (window.__rtPwa && window.__rtPwa.ev);
    if (!ev || typeof ev.prompt !== 'function') return false;
    deferredPrompt = null;
    if (window.__rtPwa) window.__rtPwa.ev = null;
    try { ev.prompt(); } catch (e) { return false; }
    if (ev.userChoice) {
      ev.userChoice.then(function (choice) {
        hideBanner();
        if (choice && choice.outcome === 'accepted') {
          markDismissed();
          setButtons('Installed', true);
          setStatus('Installed on this device.');
        } else {
          setStatus('Install stayed on the device sheet. Tap again any time.');
        }
      }).catch(function () {});
    }
    return true;
  }

  function showBanner() {
    if (bannerEl || isStandalone() || wasDismissedRecently()) return;
    bannerEl = document.createElement('div');
    bannerEl.id = 'rathor-pwa-banner';
    bannerEl.setAttribute('role', 'dialog');
    bannerEl.className =
      'fixed bottom-4 left-4 right-4 sm:left-auto sm:right-6 sm:max-w-sm z-[9999] ' +
      'bg-zinc-950/95 border border-amber-300/40 rounded-2xl shadow-2xl p-4 sm:p-5 backdrop-blur-md';
    bannerEl.innerHTML =
      '<div class="flex items-start gap-3">' +
      '  <img src="/icons/ra-thor-icon-192.png" alt="" width="48" height="48" class="rounded-xl shrink-0 w-12 h-12 object-cover border border-amber-300/30" />' +
      '  <div class="flex-1 min-w-0">' +
      '    <p class="text-amber-300 font-semibold text-sm sm:text-base leading-snug">Install Ra-Thor</p>' +
      '    <p class="text-white/60 text-xs sm:text-sm mt-1 leading-relaxed">Home-screen icon. Offline lattice. No store.</p>' +
      '    <div class="flex flex-wrap gap-2 mt-3">' +
      '      <button type="button" id="rathor-pwa-install-btn" class="px-4 py-2 rounded-xl bg-amber-400 text-black text-sm font-semibold">Install</button>' +
      '      <button type="button" id="rathor-pwa-dismiss-btn" class="px-4 py-2 rounded-xl border border-white/20 text-white/70 text-sm">Not now</button>' +
      '    </div>' +
      '  </div>' +
      '  <button type="button" id="rathor-pwa-close-btn" aria-label="Dismiss" class="text-white/40 text-lg leading-none shrink-0 px-1">×</button>' +
      '</div>';
    document.body.appendChild(bannerEl);
    var inst = document.getElementById('rathor-pwa-install-btn');
    if (inst) inst.addEventListener('click', function () { triggerInstall(); });
    function dismiss() { markDismissed(); hideBanner(); }
    var d = document.getElementById('rathor-pwa-dismiss-btn');
    var x = document.getElementById('rathor-pwa-close-btn');
    if (d) d.addEventListener('click', dismiss);
    if (x) x.addEventListener('click', dismiss);
  }

  function showHint(html) {
    hideBanner();
    var hint = document.createElement('div');
    hint.id = 'rathor-pwa-banner';
    hint.className =
      'fixed bottom-4 left-4 right-4 sm:left-auto sm:right-6 sm:max-w-sm z-[9999] ' +
      'bg-zinc-950 border border-amber-300/40 rounded-2xl p-4 text-sm text-white/80';
    hint.innerHTML = html +
      '<button type="button" class="mt-3 text-amber-400 underline text-xs" id="rathor-ios-hint-ok">Got it</button>';
    document.body.appendChild(hint);
    bannerEl = hint;
    var ok = document.getElementById('rathor-ios-hint-ok');
    if (ok) ok.addEventListener('click', hideBanner);
    setTimeout(hideBanner, 18000);
  }

  function waitThenPrompt() {
    if (waiting) return;
    waiting = true;
    setButtons('Preparing…', true);
    setStatus('Warming the Chrome install sheet. Stay on this page.');
    registerServiceWorker();
    var started = Date.now();
    var timer = setInterval(function () {
      if (firePrompt()) {
        clearInterval(timer);
        waiting = false;
        setButtons('Install Ra-Thor', false);
        return;
      }
      if (Date.now() - started >= WAIT_MS) {
        clearInterval(timer);
        waiting = false;
        setButtons('Install Ra-Thor', false);
        setStatus('If the sheet did not open: Chrome menu ⋮ → Install app (not Add to Home screen).');
        showHint(
          '<p class="text-amber-300 font-semibold mb-1">Chrome install</p>' +
          '<p>Stay on this page a few seconds, then tap Install again — or use Chrome’s menu <strong>⋮ → Install app</strong> (not Add to Home screen).</p>'
        );
      }
    }, 350);
  }

  function triggerInstall() {
    if (isStandalone()) { setButtons('Installed', true); return; }
    if (isEmbedded()) {
      showHint(
        '<p class="text-amber-300 font-semibold mb-1">Open in Chrome</p>' +
        '<p>This in-app view cannot create a real app.</p>' +
        '<a href="' + chromeIntent() + '" class="mt-3 inline-flex px-4 py-2 rounded-xl bg-amber-400 text-black text-sm font-semibold">Open rathor.ai in Chrome</a>'
      );
      return;
    }
    if (isIos()) {
      showHint('<p class="text-amber-300 font-semibold mb-1">Add to Home Screen</p><p>Tap <strong>Share</strong> → <strong>Add to Home Screen</strong>.</p>');
      return;
    }
    if (firePrompt()) return;
    waitThenPrompt();
  }

  function wire(el) {
    if (!el || el.getAttribute('data-rt-pwa-wired') === '1') return;
    el.setAttribute('data-rt-pwa-wired', '1');
    el.addEventListener('click', function (e) { e.preventDefault(); triggerInstall(); });
  }

  window.addEventListener('beforeinstallprompt', function (e) {
    e.preventDefault();
    deferredPrompt = e;
    window.__rtPwa.ev = e;
    setButtons('Install Ra-Thor', false);
    setStatus('Ready. Tap Install for the Chrome system sheet.');
    setTimeout(function () {
      if (!wasDismissedRecently() && !isStandalone()) showBanner();
    }, 1800);
  });

  window.addEventListener('appinstalled', function () {
    deferredPrompt = null;
    if (window.__rtPwa) window.__rtPwa.ev = null;
    hideBanner();
    markDismissed();
    setButtons('Installed', true);
    setStatus('Installed on this device.');
  });

  window.rathorTriggerPWAInstall = triggerInstall;
  registerServiceWorker();
  function mount() {
    if (isStandalone()) { setButtons('Installed', true); setStatus('Running as the installed app.'); return; }
    wire(document.getElementById('rathor-hero-install'));
    wire(document.getElementById('rathor-lattice-install'));
    var extras = document.querySelectorAll('[data-rt-pwa-install]');
    for (var i = 0; i < extras.length; i++) wire(extras[i]);
    setButtons('Install Ra-Thor', false);
    if (deferredPrompt || (window.__rtPwa && window.__rtPwa.ev)) {
      setStatus('Ready. Tap Install for the Chrome system sheet.');
    }
  }
  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);
})();
