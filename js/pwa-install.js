/* js/pwa-install.js — native install, early-event safe
 * Workspace 14.15.6 · TOLC 8 · info@Rathor.ai
 */
(function () {
  'use strict';

  window.__rtPwa = window.__rtPwa || { ev: null };
  var deferredPrompt = window.__rtPwa.ev || null;
  var swReady = false;

  function ua() { return navigator.userAgent || ''; }
  function isStandalone() {
    return (
      (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches) ||
      window.navigator.standalone === true ||
      (document.referrer && document.referrer.indexOf('android-app://') !== -1)
    );
  }
  function isIos() { return /iphone|ipad|ipod/i.test(ua()); }
  function isAndroid() { return /android/i.test(ua()); }
  function isEmbedded() {
    var s = ua();
    if (/\bwv\b/.test(s) || /WebView/i.test(s)) return true;
    if (/Grok\/|TwitterAndroid|FBAN|FBAV|Instagram|Line\/|WhatsApp/i.test(s)) return true;
    try {
      if (window.self !== window.top) return true;
    } catch (e) { return true; }
    return false;
  }

  function chromeIntent() {
    return 'intent://rathor.ai/#Intent;scheme=https;package=com.android.chrome;S.browser_fallback_url=https%3A%2F%2Frathor.ai%2F;end';
  }

  function ensureHeadTags() {
    var head = document.head;
    if (!head) return;
    if (!document.querySelector('link[rel="manifest"]')) {
      var man = document.createElement('link');
      man.rel = 'manifest';
      man.href = '/manifest.json';
      head.appendChild(man);
    }
    var jpeg = document.querySelector('link[rel="icon"][href*=".jpg"], link[rel="icon"][type="image/jpeg"]');
    if (jpeg) {
      jpeg.setAttribute('href', '/icons/ra-thor-icon-192.png');
      jpeg.setAttribute('type', 'image/png');
    }
    ['apple-mobile-web-app-capable', 'mobile-web-app-capable'].forEach(function (name) {
      if (!document.querySelector('meta[name="' + name + '"]')) {
        var m = document.createElement('meta');
        m.name = name;
        m.content = 'yes';
        head.appendChild(m);
      }
    });
  }

  function registerServiceWorker() {
    if (!('serviceWorker' in navigator)) return Promise.resolve(null);
    if (swReady) return navigator.serviceWorker.ready.catch(function () { return null; });
    return navigator.serviceWorker
      .register('/sw.js', { scope: '/', updateViaCache: 'none' })
      .then(function (reg) {
        swReady = true;
        if (reg && reg.update) try { reg.update(); } catch (e) {}
        return navigator.serviceWorker.ready.then(function () { return reg; });
      })
      .catch(function (err) {
        console.warn('[Ra-Thor PWA] SW failed', err);
        return null;
      });
  }

  function setControlState(readyNative, installed) {
    var nodes = document.querySelectorAll(
      '#rathor-hero-install, #rathor-lattice-install, #rathor-pwa-install-btn, [data-rt-pwa-install]'
    );
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (el.tagName === 'INSTALL') continue;
      if (installed) {
        el.textContent = 'Installed';
        el.disabled = true;
        continue;
      }
      el.disabled = false;
      if (el.id === 'rathor-pwa-install-btn') el.textContent = 'Install Ra-Thor';
      else el.innerHTML = '<i class="fa-solid fa-download"></i> Install Ra-Thor';
    }
    var note = document.getElementById('rathor-pwa-status');
    if (note) {
      note.textContent = installed
        ? 'Running as an installed app on this device.'
        : readyNative
          ? 'Native install ready. One tap — Chrome system sheet.'
          : isEmbedded()
            ? 'Open in Chrome to install as a real app (in-app browsers cannot).'
            : 'Install from this website. Offline-ready. No store.';
    }
  }

  function closeHint() {
    var hint = document.getElementById('rathor-pwa-hint');
    if (hint && hint.parentNode) hint.parentNode.removeChild(hint);
  }

  function showHint(html) {
    closeHint();
    if (!document.body) return;
    var hint = document.createElement('div');
    hint.id = 'rathor-pwa-hint';
    hint.setAttribute('role', 'dialog');
    hint.className =
      'fixed bottom-4 left-4 right-4 sm:left-auto sm:right-6 sm:max-w-sm z-[9999] ' +
      'bg-zinc-950 border border-sky-300/40 rounded-2xl p-4 text-sm text-white/80 shadow-2xl';
    hint.innerHTML = html +
      '<button type="button" class="mt-3 text-sky-300 underline text-xs" id="rathor-ios-hint-ok">Close</button>';
    document.body.appendChild(hint);
    var ok = document.getElementById('rathor-ios-hint-ok');
    if (ok) ok.addEventListener('click', closeHint);
    setTimeout(closeHint, 20000);
  }

  function fireNativeInstall() {
    var ev = deferredPrompt || (window.__rtPwa && window.__rtPwa.ev);
    if (!ev || typeof ev.prompt !== 'function') return Promise.resolve(false);
    deferredPrompt = null;
    if (window.__rtPwa) window.__rtPwa.ev = null;
    try { ev.prompt(); } catch (e) { return Promise.resolve(false); }
    return ev.userChoice.then(function (choice) {
      var accepted = !!(choice && choice.outcome === 'accepted');
      setControlState(false, accepted || isStandalone());
      return accepted;
    }).catch(function () {
      setControlState(false, isStandalone());
      return false;
    });
  }

  function waitForPrompt(ms) {
    return new Promise(function (resolve) {
      if (deferredPrompt || (window.__rtPwa && window.__rtPwa.ev)) {
        deferredPrompt = deferredPrompt || window.__rtPwa.ev;
        resolve(true);
        return;
      }
      var waited = 0;
      var tick = setInterval(function () {
        waited += 100;
        if (window.__rtPwa && window.__rtPwa.ev) deferredPrompt = window.__rtPwa.ev;
        if (deferredPrompt) {
          clearInterval(tick);
          resolve(true);
        } else if (waited >= ms) {
          clearInterval(tick);
          resolve(false);
        }
      }, 100);
    });
  }

  function triggerInstall() {
    if (isStandalone()) {
      setControlState(false, true);
      showHint('<p class="text-sky-200 font-semibold mb-1">Already installed</p><p>Ra-Thor is running as an app on this device.</p>');
      return;
    }
    if (isIos()) {
      showHint('<p class="text-sky-200 font-semibold mb-1">Add from Safari</p><p>Apple has no install API. Tap <strong>Share</strong> → <strong>Add to Home Screen</strong>.</p>');
      return;
    }
    if (isEmbedded()) {
      showHint(
        '<p class="text-sky-200 font-semibold mb-1">Open in Chrome</p>' +
        '<p>This in-app browser cannot install a real app. Chrome can.</p>' +
        '<a href="' + chromeIntent() + '" class="mt-3 inline-flex px-4 py-2 rounded-xl bg-sky-300 text-black text-sm font-semibold">Open rathor.ai in Chrome</a>'
      );
      try { window.location.href = chromeIntent(); } catch (e) {}
      return;
    }

    registerServiceWorker().then(function () {
      return waitForPrompt(3500);
    }).then(function (ready) {
      if (ready || deferredPrompt || (window.__rtPwa && window.__rtPwa.ev)) {
        return fireNativeInstall();
      }
      var inst = document.querySelector('install');
      if (inst && typeof inst.click === 'function') {
        try { inst.click(); return true; } catch (e) {}
      }
      if (isAndroid()) {
        showHint(
          '<p class="text-sky-200 font-semibold mb-1">Use Chrome’s Install app</p>' +
          '<p>Tap the <strong>⋮</strong> menu → <strong>Install app</strong> (or the install icon in the address bar). If you already added a shortcut with a Chrome badge, delete that icon first, then install again.</p>' +
          '<a href="' + chromeIntent() + '" class="mt-3 inline-flex px-4 py-2 rounded-xl bg-sky-300 text-black text-sm font-semibold">Reopen in Chrome</a>'
        );
      } else {
        showHint(
          '<p class="text-sky-200 font-semibold mb-1">Install from the address bar</p>' +
          '<p>Look for the install icon on the right of the URL bar, or the browser menu item <strong>Install Ra-Thor</strong>.</p>'
        );
      }
      return false;
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
    var slot = document.getElementById('rathor-pwa-slot');
    if (slot && !slot.querySelector('install')) {
      var native = document.createElement('install');
      native.setAttribute('style', 'display:block;margin:0.75rem auto 0;');
      slot.appendChild(native);
    }
    wireButton(document.getElementById('rathor-hero-install'));
    wireButton(document.getElementById('rathor-lattice-install'));
    var extras = document.querySelectorAll('[data-rt-pwa-install]');
    for (var i = 0; i < extras.length; i++) wireButton(extras[i]);
    setControlState(!!(deferredPrompt || (window.__rtPwa && window.__rtPwa.ev)), false);
  }

  window.addEventListener('beforeinstallprompt', function (e) {
    e.preventDefault();
    deferredPrompt = e;
    window.__rtPwa.ev = e;
    setControlState(true, false);
  });

  window.addEventListener('appinstalled', function () {
    deferredPrompt = null;
    if (window.__rtPwa) window.__rtPwa.ev = null;
    closeHint();
    setControlState(false, true);
  });

  window.rathorTriggerPWAInstall = triggerInstall;

  ensureHeadTags();
  registerServiceWorker();
  if (document.body) mountSiteCta();
  else document.addEventListener('DOMContentLoaded', mountSiteCta);
})();
