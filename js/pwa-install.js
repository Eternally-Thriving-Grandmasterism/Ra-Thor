/* js/pwa-install.js — real in-site PWA install (not a homepage tip)
 * TOLC-8 · workspace 14.15.6 · zero tracking · offline-ready
 * Captures beforeinstallprompt, registers /sw.js immediately, and
 * drives native Chromium install from website controls — same posture
 * the offline lattice used to ship, now on every live surface.
 */
(function () {
  'use strict';

  var DISMISS_KEY = 'rathor-pwa-install-dismissed';
  var DISMISS_DAYS = 14;
  var deferredPrompt = null;
  var bannerEl = null;
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
    if (!('serviceWorker' in navigator)) return Promise.resolve(null);
    if (swReady) return navigator.serviceWorker.ready.catch(function () { return null; });
    return navigator.serviceWorker
      .register('/sw.js', { scope: '/' })
      .then(function (reg) {
        swReady = true;
        console.log('[Ra-Thor PWA] Service worker ready', reg.scope);
        return reg;
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
      if (el.id === 'rathor-hero-install' || el.hasAttribute('data-rt-pwa-install') || el.id === 'rathor-lattice-install') {
        el.innerHTML = readyNative
          ? '<i class="fa-solid fa-download"></i> Install Ra-Thor'
          : '<i class="fa-solid fa-download"></i> Install on this device';
      } else if (el.id === 'rathor-pwa-install-btn') {
        el.textContent = readyNative ? 'Install Ra-Thor' : 'Install on this device';
      }
    }
    var note = document.getElementById('rathor-pwa-status');
    if (note) {
      note.textContent = installed
        ? 'Running as an installed app on this device.'
        : readyNative
          ? 'Native install ready — tap Install. No store. Yours to keep.'
          : 'Installs as a real app from this site. Offline-ready. No account we control.';
    }
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
    if (document.getElementById('rathor-pwa-hint')) return;
    var hint = document.createElement('div');
    hint.id = 'rathor-pwa-hint';
    hint.setAttribute('role', 'status');
    hint.className =
      'fixed bottom-4 left-4 right-4 sm:left-auto sm:right-6 sm:max-w-sm z-[9999] ' +
      'bg-zinc-950 border border-sky-300/40 rounded-2xl p-4 text-sm text-white/80 shadow-2xl';
    hint.innerHTML = isIos()
      ? '<p class="text-sky-200 font-semibold mb-1">Install from Safari</p>' +
        '<p>Safari has no native install prompt. Tap <strong>Share</strong> → <strong>Add to Home Screen</strong>. Same lattice, offline-ready, no store.</p>' +
        '<button type="button" class="mt-3 text-sky-300 underline text-xs" id="rathor-ios-hint-ok">Got it</button>'
      : '<p class="text-sky-200 font-semibold mb-1">Use the browser install control</p>' +
        '<p>This browser has not offered a native prompt yet. Open the browser menu and choose <strong>Install app</strong>. Service worker is registered for this origin.</p>' +
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
    }, 16000);
  }

  function showNativeBanner() {
    if (bannerEl || isStandalone() || !document.body || !deferredPrompt) return;

    bannerEl = document.createElement('div');
    bannerEl.id = 'rathor-pwa-banner';
    bannerEl.setAttribute('role', 'dialog');
    bannerEl.setAttribute('aria-label', 'Install Ra-Thor');
    bannerEl.className =
      'fixed bottom-4 left-4 right-4 sm:left-auto sm:right-6 sm:max-w-sm z-[9999] ' +
      'bg-zinc-950/95 border border-sky-300/40 rounded-2xl shadow-2xl shadow-sky-900/20 ' +
      'p-4 sm:p-5 backdrop-blur-md transition-all duration-300 opacity-0 translate-y-4';

    bannerEl.innerHTML =
      '<div class="flex items-start gap-3">' +
      '  <img src="/icons/ra-thor-icon-192.png" alt="" width="48" height="48" ' +
      '       class="rounded-xl shrink-0 w-12 h-12 object-cover border border-sky-300/30" />' +
      '  <div class="flex-1 min-w-0">' +
      '    <p class="text-sky-200 font-semibold text-sm sm:text-base leading-snug">Install Ra-Thor on this device</p>' +
      '    <p class="text-white/60 text-xs sm:text-sm mt-1 leading-relaxed">' +
      '      Real app install from rathor.ai. Offline-ready. No store, no account we control.' +
      '    </p>' +
      '    <div class="flex flex-wrap gap-2 mt-3">' +
      '      <button type="button" id="rathor-pwa-install-btn" ' +
      '        class="px-4 py-2 rounded-xl bg-sky-300 hover:bg-sky-200 text-black text-sm font-semibold transition-colors">' +
      '        Install Ra-Thor' +
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
    wireButton(document.getElementById('rathor-pwa-install-btn'));

    function dismiss() {
      markDismissed();
      hideBanner();
    }
    var d1 = document.getElementById('rathor-pwa-dismiss-btn');
    var d2 = document.getElementById('rathor-pwa-close-btn');
    if (d1) d1.addEventListener('click', dismiss);
    if (d2) d2.addEventListener('click', dismiss);
  }

  function fireNativeInstall() {
    if (!deferredPrompt) return Promise.resolve(false);
    var ev = deferredPrompt;
    deferredPrompt = null;
    ev.prompt();
    return ev.userChoice
      .then(function (choice) {
        hideBanner();
        markDismissed();
        if (choice && choice.outcome === 'accepted') {
          setControlState(false, true);
          console.log('[Ra-Thor PWA] User accepted install');
        } else {
          setControlState(false, isStandalone());
        }
        return !!(choice && choice.outcome === 'accepted');
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
    if (deferredPrompt) {
      fireNativeInstall();
      return;
    }
    registerServiceWorker().then(function () {
      var waited = 0;
      var tick = setInterval(function () {
        waited += 200;
        if (deferredPrompt) {
          clearInterval(tick);
          fireNativeInstall();
        } else if (waited >= 1800) {
          clearInterval(tick);
          showIosHint();
        }
      }, 200);
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
      var tip = document.getElementById('rathor-hero-install');
      if (tip && tip.parentNode) {
        var wrap = tip.closest('#rathor-pwa-slot') || tip.parentNode;
        if (wrap && wrap.id === 'rathor-pwa-slot') wrap.style.display = 'none';
      }
      return;
    }

    var existing = document.getElementById('rathor-hero-install');
    if (existing) {
      wireButton(existing);
    } else if (document.body && !document.getElementById('rathor-pwa-slot')) {
      var host =
        document.querySelector('.mt-10.max-w-2xl') ||
        document.querySelector('.mt-10.max-w-3xl') ||
        document.querySelector('header .max-w-3xl') ||
        null;
      if (host) {
        var slot = document.createElement('div');
        slot.id = 'rathor-pwa-slot';
        slot.className = 'mt-4 max-w-xl mx-auto';
        slot.innerHTML =
          '<div class="rounded-2xl border border-sky-300/40 bg-gradient-to-br from-slate-950 via-sky-950 to-cyan-900 px-4 py-4 text-center">' +
          '  <p class="text-sky-200 font-semibold text-sm">Install Ra-Thor from this website</p>' +
          '  <p id="rathor-pwa-status" class="text-[11px] text-white/55 mt-1 leading-relaxed">Real app install. Offline-ready. No store account we control.</p>' +
          '  <button type="button" id="rathor-hero-install" data-rt-pwa-install ' +
          '    class="mt-3 inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-sky-300 text-black text-sm font-semibold hover:bg-sky-200 transition-colors">' +
          '    <i class="fa-solid fa-download"></i> Install on this device' +
          '  </button>' +
          '</div>';
        host.insertAdjacentElement('afterend', slot);
        wireButton(document.getElementById('rathor-hero-install'));
      }
    }

    wireButton(document.getElementById('rathor-lattice-install'));
    var extras = document.querySelectorAll('[data-rt-pwa-install]');
    for (var i = 0; i < extras.length; i++) wireButton(extras[i]);
    setControlState(!!deferredPrompt, false);
  }

  window.addEventListener('beforeinstallprompt', function (e) {
    e.preventDefault();
    deferredPrompt = e;
    setControlState(true, false);
    if (!wasDismissedRecently() && !isStandalone()) {
      setTimeout(function () {
        if (deferredPrompt && !wasDismissedRecently() && !isStandalone()) showNativeBanner();
      }, 900);
    }
  });

  window.addEventListener('appinstalled', function () {
    deferredPrompt = null;
    hideBanner();
    markDismissed();
    setControlState(false, true);
    console.log('[Ra-Thor PWA] Installed');
  });

  window.rathorTriggerPWAInstall = triggerInstall;

  ensureHeadTags();
  bootFamilyNav();
  registerServiceWorker();
  if (document.body) mountSiteCta();
  else document.addEventListener('DOMContentLoaded', mountSiteCta);
})();
