/* js/pwa-install.js — Ra-Thor respectful PWA install lattice
 * TOLC-8 aligned • zero tracking • dismissible • offline-ready
 * Improved Install button support
 * Also boots family-nav so chat.html joins Home/Launch/Shard/Forge/Contact/Privacy.
 */
(function () {
  'use strict';

  if (!document.querySelector('script[src*="family-nav-2026-08-22"]')) {
    var nav = document.createElement('script');
    nav.src = '/js/family-nav-2026-08-22.js';
    nav.defer = true;
    document.head.appendChild(nav);
  }

  var DISMISS_KEY = 'rathor-pwa-install-dismissed';
  var DISMISS_DAYS = 14;
  var deferredPrompt = null;
  var bannerEl = null;

  function isStandalone() {
    return (
      window.matchMedia('(display-mode: standalone)').matches ||
      window.navigator.standalone === true ||
      document.referrer.includes('android-app://')
    );
  }

  function wasDismissedRecently() {
    try {
      var raw = localStorage.getItem(DISMISS_KEY);
      if (!raw) return false;
      var ts = parseInt(raw, 10);
      if (isNaN(ts)) return false;
      return Date.now() - ts < DISMISS_DAYS * 24 * 60 * 60 * 1000;
    } catch (e) {
      return false;
    }
  }

  function markDismissed() {
    try {
      localStorage.setItem(DISMISS_KEY, String(Date.now()));
    } catch (e) {}
  }

  function registerServiceWorker() {
    if (!('serviceWorker' in navigator)) return;
    window.addEventListener('load', function () {
      navigator.serviceWorker
        .register('/sw.js', { scope: '/' })
        .then(function (reg) {
          console.log('[Ra-Thor PWA] Service worker ready', reg.scope);
        })
        .catch(function (err) {
          console.warn('[Ra-Thor PWA] SW registration failed', err);
        });
    });
  }

  function hideBanner() {
    if (!bannerEl) return;
    bannerEl.classList.add('opacity-0', 'translate-y-4');
    setTimeout(function () {
      if (bannerEl && bannerEl.parentNode) bannerEl.parentNode.removeChild(bannerEl);
      bannerEl = null;
    }, 300);
  }

  function showBanner() {
    if (bannerEl || isStandalone()) return;

    bannerEl = document.createElement('div');
    bannerEl.id = 'rathor-pwa-banner';
    bannerEl.setAttribute('role', 'dialog');
    bannerEl.setAttribute('aria-label', 'Install Ra-Thor');
    bannerEl.className =
      'fixed bottom-4 left-4 right-4 sm:left-auto sm:right-6 sm:max-w-sm z-[9999] ' +
      'bg-zinc-950/95 border border-amber-300/40 rounded-2xl shadow-2xl shadow-amber-900/20 ' +
      'p-4 sm:p-5 backdrop-blur-md transition-all duration-300 opacity-0 translate-y-4';

    bannerEl.innerHTML =
      '<div class="flex items-start gap-3">' +
      '  <img src="/icons/ra-thor-icon-192.png" alt="" width="48" height="48" ' +
      '       class="rounded-xl shrink-0 w-12 h-12 object-cover border border-amber-300/30" />' +
      '  <div class="flex-1 min-w-0">' +
      '    <p class="text-amber-300 font-semibold text-sm sm:text-base leading-snug">Install Ra-Thor</p>' +
      '    <p class="text-white/60 text-xs sm:text-sm mt-1 leading-relaxed">' +
      '      Add a home-screen icon for quick, offline access. No app store. Fully under your control.' +
      '    </p>' +
      '    <div class="flex flex-wrap gap-2 mt-3">' +
      '      <button type="button" id="rathor-pwa-install-btn" ' +
      '        class="px-4 py-2 rounded-xl bg-amber-400 hover:bg-amber-300 text-black text-sm font-semibold transition-colors">' +
      '        Install' +
      '      </button>' +
      '      <button type="button" id="rathor-pwa-dismiss-btn" ' +
      '        class="px-4 py-2 rounded-xl border border-white/20 text-white/70 hover:text-white text-sm transition-colors">' +
      '        Not now' +
      '      </button>' +
      '    </div>' +
      '  </div>' +
      '  <button type="button" id="rathor-pwa-close-btn" aria-label="Dismiss" ' +
      '    class="text-white/40 hover:text-white/80 text-lg leading-none shrink-0 px-1">×</button>' +
      '</div>';

    document.body.appendChild(bannerEl);
    requestAnimationFrame(function () {
      if (bannerEl) bannerEl.classList.remove('opacity-0', 'translate-y-4');
    });

    document.getElementById('rathor-pwa-install-btn').addEventListener('click', function () {
      if (!deferredPrompt) {
        hideBanner();
        markDismissed();
        showIosHint();
        return;
      }
      deferredPrompt.prompt();
      deferredPrompt.userChoice.then(function (choice) {
        deferredPrompt = null;
        hideBanner();
        markDismissed();
        if (choice && choice.outcome === 'accepted') {
          console.log('[Ra-Thor PWA] User accepted install');
        }
      });
    });

    function dismiss() {
      markDismissed();
      hideBanner();
    }
    document.getElementById('rathor-pwa-dismiss-btn').addEventListener('click', dismiss);
    document.getElementById('rathor-pwa-close-btn').addEventListener('click', dismiss);
  }

  function showIosHint() {
    var isIos = /iphone|ipad|ipod/i.test(navigator.userAgent);
    if (!isIos) {
      alert('To install Ra-Thor: use your browser menu → "Install app" or "Add to Home Screen".');
      return;
    }
    var hint = document.createElement('div');
    hint.className =
      'fixed bottom-4 left-4 right-4 sm:left-auto sm:right-6 sm:max-w-sm z-[9999] ' +
      'bg-zinc-950 border border-amber-300/40 rounded-2xl p-4 text-sm text-white/80';
    hint.innerHTML =
      '<p class="text-amber-300 font-semibold mb-1">Add to Home Screen</p>' +
      '<p>On iPhone/iPad: tap <strong>Share</strong> → <strong>Add to Home Screen</strong>. ' +
      'Ra-Thor opens like an app, offline-ready.</p>' +
      '<button type="button" class="mt-3 text-amber-400 underline text-xs" id="rathor-ios-hint-ok">Got it</button>';
    document.body.appendChild(hint);
    document.getElementById('rathor-ios-hint-ok').addEventListener('click', function () {
      if (hint.parentNode) hint.parentNode.removeChild(hint);
    });
    setTimeout(function () {
      if (hint.parentNode) hint.parentNode.removeChild(hint);
    }, 12000);
  }

  window.addEventListener('beforeinstallprompt', function (e) {
    e.preventDefault();
    deferredPrompt = e;
    setTimeout(function () {
      if (!wasDismissedRecently() && !isStandalone()) showBanner();
    }, 4500);
  });

  window.addEventListener('appinstalled', function () {
    deferredPrompt = null;
    hideBanner();
    markDismissed();
    console.log('[Ra-Thor PWA] Installed');
  });

  function maybeIosSoftOffer() {
    var isIos = /iphone|ipad|ipod/i.test(navigator.userAgent);
    if (!isIos || isStandalone() || wasDismissedRecently()) return;
    setTimeout(function () {
      if (!isStandalone() && !wasDismissedRecently()) showBanner();
    }, 8000);
  }

  registerServiceWorker();
  maybeIosSoftOffer();

  window.rathorTriggerPWAInstall = function () {
    if (isStandalone()) {
      alert('Ra-Thor is already installed and running as an app.');
      return;
    }
    if (deferredPrompt) {
      deferredPrompt.prompt();
      deferredPrompt.userChoice.then(function () {
        deferredPrompt = null;
        hideBanner();
        markDismissed();
      });
      return;
    }
    localStorage.removeItem(DISMISS_KEY);
    showBanner();
    setTimeout(function () {
      if (!deferredPrompt && !isStandalone()) {
        showIosHint();
      }
    }, 900);
  };
})();
