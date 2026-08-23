/* js/pwa-install.js — Ra-Thor warm, dismissible PWA install lattice
 * TOLC-8 aligned • zero tracking • offline-ready • family-nav boot
 * Restored 2026-08-22/23: register SW immediately, inject manifest tags,
 * soft home-screen offer on every live surface (not Chat-only).
 */
(function () {
  'use strict';

  var DISMISS_KEY = 'rathor-pwa-install-dismissed';
  var DISMISS_DAYS = 14;
  var deferredPrompt = null;
  var bannerEl = null;

  function ensureHeadTags() {
    var head = document.head;
    if (!head) return;
    if (!document.querySelector('link[rel="manifest"]')) {
      var man = document.createElement('link');
      man.rel = 'manifest';
      man.href = '/manifest.json';
      head.appendChild(man);
    }
    if (!document.querySelector('link[rel="apple-touch-icon"]')) {
      var apple = document.createElement('link');
      apple.rel = 'apple-touch-icon';
      apple.href = '/icons/ra-thor-icon-192.png';
      head.appendChild(apple);
    }
    if (!document.querySelector('meta[name="theme-color"]')) {
      var theme = document.createElement('meta');
      theme.name = 'theme-color';
      theme.content = '#fcd34d';
      head.appendChild(theme);
    }
    if (!document.querySelector('meta[name="apple-mobile-web-app-capable"]')) {
      var cap = document.createElement('meta');
      cap.name = 'apple-mobile-web-app-capable';
      cap.content = 'yes';
      head.appendChild(cap);
    }
    if (!document.querySelector('meta[name="mobile-web-app-capable"]')) {
      var mcap = document.createElement('meta');
      mcap.name = 'mobile-web-app-capable';
      mcap.content = 'yes';
      head.appendChild(mcap);
    }
    if (!document.querySelector('meta[name="apple-mobile-web-app-title"]')) {
      var title = document.createElement('meta');
      title.name = 'apple-mobile-web-app-title';
      title.content = 'Ra-Thor';
      head.appendChild(title);
    }
  }

  function bootFamilyNav() {
    if (document.querySelector('script[src*="family-nav-2026-08-22"]')) return;
    var nav = document.createElement('script');
    nav.src = '/js/family-nav-2026-08-22.js';
    nav.defer = true;
    (document.head || document.documentElement).appendChild(nav);
  }

  function isStandalone() {
    return (
      window.matchMedia('(display-mode: standalone)').matches ||
      window.navigator.standalone === true ||
      (document.referrer && document.referrer.indexOf('android-app://') !== -1)
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
    var start = function () {
      navigator.serviceWorker
        .register('/sw.js', { scope: '/' })
        .then(function (reg) {
          console.log('[Ra-Thor PWA] Service worker ready', reg.scope);
        })
        .catch(function (err) {
          console.warn('[Ra-Thor PWA] SW registration failed', err);
        });
    };
    if (document.readyState === 'complete') start();
    else window.addEventListener('load', start);
  }

  function hideBanner() {
    if (!bannerEl) return;
    bannerEl.classList.add('opacity-0', 'translate-y-4');
    setTimeout(function () {
      if (bannerEl && bannerEl.parentNode) bannerEl.parentNode.removeChild(bannerEl);
      bannerEl = null;
    }, 280);
  }

  function showIosHint() {
    var isIos = /iphone|ipad|ipod/i.test(navigator.userAgent);
    var hint = document.createElement('div');
    hint.id = 'rathor-pwa-hint';
    hint.setAttribute('role', 'status');
    hint.className =
      'fixed bottom-4 left-4 right-4 sm:left-auto sm:right-6 sm:max-w-sm z-[9999] ' +
      'bg-zinc-950 border border-amber-300/40 rounded-2xl p-4 text-sm text-white/80 shadow-2xl';
    hint.innerHTML = isIos
      ? '<p class="text-amber-300 font-semibold mb-1">Add to Home Screen</p>' +
        '<p>On iPhone/iPad: tap <strong>Share</strong> → <strong>Add to Home Screen</strong>. ' +
        'Ra-Thor opens like an app, offline-ready. No store. Yours to keep.</p>' +
        '<button type="button" class="mt-3 text-amber-400 underline text-xs" id="rathor-ios-hint-ok">Got it</button>'
      : '<p class="text-amber-300 font-semibold mb-1">Install from your browser</p>' +
        '<p>Use the browser menu → <strong>Install app</strong> or <strong>Add to Home Screen</strong>. ' +
        'Same lattice, on your device, with no account we control.</p>' +
        '<button type="button" class="mt-3 text-amber-400 underline text-xs" id="rathor-ios-hint-ok">Got it</button>';
    document.body.appendChild(hint);
    var ok = document.getElementById('rathor-ios-hint-ok');
    if (ok) {
      ok.addEventListener('click', function () {
        if (hint.parentNode) hint.parentNode.removeChild(hint);
      });
    }
    setTimeout(function () {
      if (hint.parentNode) hint.parentNode.removeChild(hint);
    }, 14000);
  }

  function showBanner() {
    if (bannerEl || isStandalone() || !document.body) return;

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
      '    <p class="text-amber-300 font-semibold text-sm sm:text-base leading-snug">Keep Ra-Thor on this device</p>' +
      '    <p class="text-white/60 text-xs sm:text-sm mt-1 leading-relaxed">' +
      '      A warm home-screen icon. Offline-ready. No store, no account, fully under your control.' +
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

    var installBtn = document.getElementById('rathor-pwa-install-btn');
    if (installBtn) {
      installBtn.addEventListener('click', function () {
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
    }

    function dismiss() {
      markDismissed();
      hideBanner();
    }
    var d1 = document.getElementById('rathor-pwa-dismiss-btn');
    var d2 = document.getElementById('rathor-pwa-close-btn');
    if (d1) d1.addEventListener('click', dismiss);
    if (d2) d2.addEventListener('click', dismiss);
  }

  function scheduleSoftOffer() {
    if (isStandalone() || wasDismissedRecently()) return;
    setTimeout(function () {
      if (!wasDismissedRecently() && !isStandalone()) showBanner();
    }, 4800);
  }

  window.addEventListener('beforeinstallprompt', function (e) {
    e.preventDefault();
    deferredPrompt = e;
    if (!wasDismissedRecently() && !isStandalone()) {
      setTimeout(function () {
        if (!wasDismissedRecently() && !isStandalone()) showBanner();
      }, 1200);
    }
  });

  window.addEventListener('appinstalled', function () {
    deferredPrompt = null;
    hideBanner();
    markDismissed();
    console.log('[Ra-Thor PWA] Installed');
  });

  window.rathorTriggerPWAInstall = function () {
    if (isStandalone()) {
      showIosHint();
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
    try {
      localStorage.removeItem(DISMISS_KEY);
    } catch (e) {}
    showBanner();
  };

  ensureHeadTags();
  bootFamilyNav();
  registerServiceWorker();
  if (document.body) scheduleSoftOffer();
  else document.addEventListener('DOMContentLoaded', scheduleSoftOffer);
})();
