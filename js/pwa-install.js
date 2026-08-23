/* js/pwa-install.js — restore the Aug-8 lattice install that Chrome accepted
 * Workspace 14.15.6 · TOLC 8 · info@Rathor.ai
 * Capture beforeinstallprompt → warm banner → event.prompt() on tap.
 */
(function () {
  'use strict';

  var DISMISS_KEY = 'rathor-pwa-install-dismissed';
  var DISMISS_DAYS = 14;
  window.__rtPwa = window.__rtPwa || { ev: null };
  var deferredPrompt = window.__rtPwa.ev || null;
  var bannerEl = null;
  var knownInstalled = false;

  function speak(opts) {
    if (typeof window.rathorSay === 'function') {
      window.rathorSay(opts);
      return;
    }
    var note = document.getElementById('rathor-pwa-status');
    if (note && opts) {
      note.textContent = ((opts.title ? opts.title + ' — ' : '') + (opts.body || '')).trim();
    }
  }

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

  function registerServiceWorker() {
    if (!('serviceWorker' in navigator)) return;
    function go() {
      navigator.serviceWorker.register('/sw.js', { scope: '/' }).catch(function () {});
    }
    if (document.readyState === 'complete') go();
    else window.addEventListener('load', go);
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
    var note = document.getElementById('rathor-pwa-status');
    if (note && !disabled) {
      note.textContent = knownInstalled
        ? 'Already on this device. Use the home-screen icon — nothing new to install.'
        : (deferredPrompt
          ? 'Chrome is ready. One tap opens the system install sheet.'
          : 'When Chrome is ready it will offer a system sheet — or use the menu Install app.');
    }
    if (note && disabled && (label === 'Installed' || label === 'Already installed')) {
      note.textContent = isStandalone()
        ? 'You are inside the installed app. Offline-ready. No store.'
        : 'Ra-Thor is already on this device.';
    }
  }

  function firePrompt() {
    var ev = deferredPrompt || (window.__rtPwa && window.__rtPwa.ev);
    if (!ev || typeof ev.prompt !== 'function') return false;
    deferredPrompt = null;
    if (window.__rtPwa) window.__rtPwa.ev = null;
    try { ev.prompt(); } catch (e) { return false; }
    speak({
      title: 'System sheet opened',
      body: 'Chrome is asking whether to install Ra-Thor on this device. Accept or cancel there.',
      ms: 3000
    });
    if (ev.userChoice) {
      ev.userChoice.then(function (choice) {
        hideBanner();
        markDismissed();
        if (choice && choice.outcome === 'accepted') {
          knownInstalled = true;
          setButtons('Installed', true);
          speak({
            title: 'Installed',
            body: 'Chrome added Ra-Thor to this device. Look for the gold thunder icon on your home screen.',
            tone: 'ok',
            ms: 3000
          });
        } else {
          setButtons('Install Ra-Thor', false);
          speak({
            title: 'Install cancelled',
            body: 'Nothing was added. Tap Install again anytime — no store account is required.',
            tone: 'hold',
            ms: 3000
          });
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
      'bg-zinc-950/95 border border-amber-300/40 rounded-2xl shadow-2xl ' +
      'p-4 sm:p-5 backdrop-blur-md';
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
    setTimeout(hideBanner, 16000);
  }

  function triggerInstall() {
    if (isStandalone()) {
      knownInstalled = true;
      setButtons('Already installed', true);
      speak({
        title: 'Already installed',
        body: 'You are already inside the home-screen app. This is Ra-Thor on this device — no store, no extra download.',
        tone: 'ok',
        ms: 3000
      });
      return;
    }
    if (knownInstalled) {
      setButtons('Already installed', true);
      speak({
        title: 'Already on this device',
        body: 'Use the Ra-Thor home-screen icon, or Chrome ⋮ → Ra-Thor. The browser page is the same lattice.',
        tone: 'ok',
        ms: 3000
      });
      return;
    }
    if (isEmbedded()) {
      speak({
        title: 'Open in Chrome',
        body: 'This in-app view cannot install a real app. Chrome can.',
        tone: 'hold',
        ms: 3000
      });
      showHint(
        '<p class="text-amber-300 font-semibold mb-1">Open in Chrome</p>' +
        '<p>This in-app view cannot create a real app.</p>' +
        '<a href="' + chromeIntent() + '" class="mt-3 inline-flex px-4 py-2 rounded-xl bg-amber-400 text-black text-sm font-semibold">Open rathor.ai in Chrome</a>'
      );
      return;
    }
    if (isIos()) {
      speak({
        title: 'iPhone / iPad',
        body: 'Safari: Share → Add to Home Screen. That is the install path on iOS.',
        ms: 3000
      });
      showHint('<p class="text-amber-300 font-semibold mb-1">Add to Home Screen</p><p>Tap <strong>Share</strong> → <strong>Add to Home Screen</strong>.</p>');
      return;
    }
    if (firePrompt()) return;
    speak({
      title: 'Sheet not ready yet',
      body: 'Stay on this page a few seconds, then tap Install again — or Chrome menu Install app.',
      tone: 'hold',
      ms: 3000
    });
    showHint(
      '<p class="text-amber-300 font-semibold mb-1">Chrome install</p>' +
      '<p>Stay on this page a few seconds, then tap Install again — or use Chrome’s menu <strong>⋮ → Install app</strong> (not Add to Home screen).</p>'
    );
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
    setTimeout(function () {
      if (!wasDismissedRecently() && !isStandalone()) showBanner();
    }, 2500);
  });

  window.addEventListener('appinstalled', function () {
    deferredPrompt = null;
    if (window.__rtPwa) window.__rtPwa.ev = null;
    knownInstalled = true;
    hideBanner();
    markDismissed();
    setButtons('Installed', true);
    speak({
      title: 'Installed',
      body: 'Done. Chrome placed Ra-Thor on this device. Open it from the home screen next time.',
      tone: 'ok',
      ms: 3000
    });
  });

  window.rathorTriggerPWAInstall = triggerInstall;
  registerServiceWorker();

  function greetStandalone() {
    if (!isStandalone()) return;
    knownInstalled = true;
    setButtons('Already installed', true);
    try {
      if (sessionStorage.getItem('rathor-hf-standalone') === '1') return;
      sessionStorage.setItem('rathor-hf-standalone', '1');
    } catch (e) {}
    speak({
      title: 'Already installed',
      body: 'You opened the home-screen app. Offline-ready. Nothing left this device.',
      tone: 'ok',
      ms: 3000
    });
  }

  function probeRelatedApps() {
    if (!navigator.getInstalledRelatedApps) return;
    try {
      navigator.getInstalledRelatedApps().then(function (apps) {
        if (apps && apps.length) {
          knownInstalled = true;
          if (!isStandalone()) setButtons('Already installed', false);
        }
      }).catch(function () {});
    } catch (e) {}
  }

  function mount() {
    greetStandalone();
    probeRelatedApps();
    if (isStandalone()) return;
    wire(document.getElementById('rathor-hero-install'));
    wire(document.getElementById('rathor-lattice-install'));
    var extras = document.querySelectorAll('[data-rt-pwa-install]');
    for (var i = 0; i < extras.length; i++) wire(extras[i]);
    if (!knownInstalled) setButtons('Install Ra-Thor', false);
  }
  if (document.body) mount();
  else document.addEventListener('DOMContentLoaded', mount);
})();
