/* js/pwa-install.js — native Chromium install, one gesture
 * Workspace 14.15.6 · TOLC 8 · zero tracking
 * Captures beforeinstallprompt without a second homepage detour.
 * iOS still needs Share → Add to Home Screen (Apple has no prompt API).
 */
(function () {
  'use strict';

  var DISMISS_KEY = 'rathor-pwa-install-dismissed';
  var DISMISS_DAYS = 14;
  var deferredPrompt = null;
  var swReady = false;

  function ensureHeadTags() {
    var head = document.head;
    if (!head) return;
    if (!document.querySelector('link[rel="manifest"]')) {
      var man = document.createElement('link');
      man.rel = 'manifest';
      man.href = '/manifest.json';
      head.appendChild(man);
    }
    var jpegIcon = document.querySelector('link[rel="icon"][href*=".jpg"], link[rel="icon"][type="image/jpeg"]');
    if (jpegIcon) {
      jpegIcon.setAttribute('href', '/icons/ra-thor-icon-192.png');
      jpegIcon.setAttribute('type', 'image/png');
    }
    if (!document.querySelector('link[rel="icon"]')) {
      var fav = document.createElement('link');
      fav.rel = 'icon';
      fav.type = 'image/png';
      fav.href = '/icons/ra-thor-icon-192.png';
      head.appendChild(fav);
    }
    if (!document.querySelector('link[rel="apple-touch-icon"]')) {
      var apple = document.createElement('link');
      apple.rel = 'apple-touch-icon';
      apple.sizes = '192x192';
      apple.href = '/icons/ra-thor-icon-192.png';
      head.appendChild(apple);
    }
    if (!document.querySelector('meta[name="theme-color"]')) {
      var theme = document.createElement('meta');
      theme.name = 'theme-color';
      theme.content = '#fcd34d';
      head.appendChild(theme);
    }
    ['apple-mobile-web-app-capable', 'mobile-web-app-capable'].forEach(function (name) {
      if (!document.querySelector('meta[name="' + name + '"]')) {
        var m = document.createElement('meta');
        m.name = name;
        m.content = 'yes';
        head.appendChild(m);
      }
    });
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
      (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches) ||
      window.navigator.standalone === true ||
      (document.referrer && document.referrer.indexOf('android-app://') !== -1)
    );
  }

  function isIos() {
    return /iphone|ipad|ipod/i.test(navigator.userAgent || '');
  }

  function markDismissed() {
    try { localStorage.setItem(DISMISS_KEY, String(Date.now())); } catch (e) {}
  }

  function registerServiceWorker() {
    if (!('serviceWorker' in navigator)) return Promise.resolve(null);
    if (swReady) return navigator.serviceWorker.ready.catch(function () { return null; });
    return navigator.serviceWorker
      .register('/sw.js', { scope: '/' })
      .then(function (reg) {
        swReady = true;
        return navigator.serviceWorker.ready.then(function () { return reg; });
      })
      .catch(function (err) {
        console.warn('[Ra-Thor PWA] SW registration failed', err);
        return null;
      });
  }

  function setControlState(readyNative, installed) {
    var nodes = document.querySelectorAll(
      '#rathor-hero-install, #rathor-lattice-install, #rathor-pwa-install-btn, [data-rt-pwa-install]'
    );
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (installed) {
        el.textContent = 'Installed';
        el.setAttribute('aria-label', 'Ra-Thor is installed on this device');
        el.disabled = true;
        continue;
      }
      el.disabled = false;
      el.setAttribute('aria-label', 'Install Ra-Thor on this device');
      if (el.id === 'rathor-pwa-install-btn') {
        el.textContent = 'Install Ra-Thor';
      } else {
        el.innerHTML = '<i class="fa-solid fa-download"></i> Install Ra-Thor';
      }
    }
    var note = document.getElementById('rathor-pwa-status');
    if (note) {
      note.textContent = installed
        ? 'Running as an installed app on this device.'
        : 'One tap installs the real app from rathor.ai. Offline-ready. No store.';
    }
  }

  function showIosHint() {
    if (document.getElementById('rathor-pwa-hint') || isStandalone()) return;
    var hint = document.createElement('div');
    hint.id = 'rathor-pwa-hint';
    hint.setAttribute('role', 'status');
    hint.className =
      'fixed bottom-4 left-4 right-4 sm:left-auto sm:right-6 sm:max-w-sm z-[9999] ' +
      'bg-zinc-950 border border-sky-300/40 rounded-2xl p-4 text-sm text-white/80 shadow-2xl';
    hint.innerHTML =
      '<p class="text-sky-200 font-semibold mb-1">Add to Home Screen</p>' +
      '<p>In Safari tap <strong>Share</strong> → <strong>Add to Home Screen</strong>. Apple does not allow a one-tap install prompt.</p>' +
      '<button type="button" class="mt-3 text-sky-300 underline text-xs" id="rathor-ios-hint-ok">Got it</button>';
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

  function fireNativeInstall() {
    if (!deferredPrompt) return Promise.resolve(false);
    var ev = deferredPrompt;
    deferredPrompt = null;
    try { ev.prompt(); } catch (e) { return Promise.resolve(false); }
    return ev.userChoice
      .then(function (choice) {
        markDismissed();
        var accepted = !!(choice && choice.outcome === 'accepted');
        setControlState(false, accepted || isStandalone());
        return accepted;
      })
      .catch(function () {
        setControlState(false, isStandalone());
        return false;
      });
  }

  function triggerInstall() {
    if (isStandalone()) {
      setControlState(false, true);
      return;
    }
    if (isIos()) {
      showIosHint();
      return;
    }
    if (deferredPrompt) {
      fireNativeInstall();
      return;
    }
    registerServiceWorker().then(function () {
      var waited = 0;
      var tick = setInterval(function () {
        waited += 150;
        if (deferredPrompt) {
          clearInterval(tick);
          fireNativeInstall();
        } else if (waited >= 2200) {
          clearInterval(tick);
          if (isIos()) showIosHint();
        }
      }, 150);
    });
  }

  function wireButton(el) {
    if (!el || el.getAttribute('data-rt-pwa-wired') === '1') return;
    el.setAttribute('data-rt-pwa-wired', '1');
    el.addEventListener('click', function (e) {
      e.preventDefault();
      triggerInstall();
    });
  }

  function mountSiteCta() {
    if (isStandalone()) {
      setControlState(false, true);
      return;
    }
    var existing = document.getElementById('rathor-hero-install');
    if (existing) wireButton(existing);
    wireButton(document.getElementById('rathor-lattice-install'));
    var extras = document.querySelectorAll('[data-rt-pwa-install]');
    for (var i = 0; i < extras.length; i++) wireButton(extras[i]);
    setControlState(!!deferredPrompt, false);
  }

  window.addEventListener('beforeinstallprompt', function (e) {
    e.preventDefault();
    deferredPrompt = e;
    setControlState(true, false);
  });

  window.addEventListener('appinstalled', function () {
    deferredPrompt = null;
    markDismissed();
    setControlState(false, true);
  });

  window.rathorTriggerPWAInstall = triggerInstall;

  ensureHeadTags();
  bootFamilyNav();
  registerServiceWorker();
  if (document.body) mountSiteCta();
  else document.addEventListener('DOMContentLoaded', mountSiteCta);
})();
