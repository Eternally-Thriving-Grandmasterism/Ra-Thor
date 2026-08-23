/* js/pwa-install.js — adaptive install intelligence
 * Workspace 14.15.6 · TOLC 8 · info@Rathor.ai
 * Chrome/Edge/Brave/Samsung: native sheet.
 * iOS / macOS Safari / Firefox / in-app: best real alternative.
 */
(function () {
  'use strict';

  window.__rtPwa = window.__rtPwa || { ev: null };
  var deferredPrompt = window.__rtPwa.ev || null;
  var swReady = false;
  var dismissed = false;

  function ua() { return navigator.userAgent || ''; }

  function isStandalone() {
    return (
      (window.matchMedia && window.matchMedia('(display-mode: standalone)').matches) ||
      window.navigator.standalone === true ||
      (document.referrer && document.referrer.indexOf('android-app://') !== -1)
    );
  }

  function env() {
    var s = ua();
    var android = /android/i.test(s);
    var ios = /iphone|ipad|ipod/i.test(s);
    var mac = /macintosh|mac os x/i.test(s);
    var chrome = /Chrome\//.test(s) && !/Edg\//.test(s) && !/OPR\//.test(s) && !/SamsungBrowser/.test(s);
    var edge = /Edg\//.test(s);
    var brave = !!(navigator.brave);
    var opera = /OPR\//.test(s);
    var samsung = /SamsungBrowser/.test(s);
    var firefox = /Firefox\//.test(s);
    var safari = /Safari\//.test(s) && !/Chrome\//.test(s) && !/Chromium\//.test(s);
    var embedded = false;
    if (/\bwv\b/.test(s) || /WebView/i.test(s)) embedded = true;
    if (/Grok\/|TwitterAndroid|FBAN|FBAV|Instagram|Line\/|WhatsApp/i.test(s)) embedded = true;
    try { if (window.self !== window.top) embedded = true; } catch (e) { embedded = true; }

    if (isStandalone()) return { id: 'standalone', family: 'app' };
    if (embedded) return { id: ios ? 'webview-ios' : 'webview-android', family: 'webview', ios: ios, android: android };
    if (ios) return { id: safari ? 'ios-safari' : 'ios-other', family: 'ios', chrome: chrome, edge: edge, firefox: firefox };
    if (samsung) return { id: 'samsung', family: 'chromium' };
    if (firefox && android) return { id: 'fx-android', family: 'firefox' };
    if (firefox) return { id: 'fx-desktop', family: 'firefox' };
    if (safari && mac) return { id: 'safari-mac', family: 'safari' };
    if (chrome || edge || brave || opera) return { id: android ? 'chromium-android' : 'chromium-desktop', family: 'chromium', android: android, edge: edge };
    if (android) return { id: 'android-other', family: 'chromium' };
    return { id: 'unknown', family: 'unknown' };
  }

  function chromeIntent() {
    return 'intent://rathor.ai/#Intent;scheme=https;package=com.android.chrome;S.browser_fallback_url=https%3A%2F%2Frathor.ai%2F;end';
  }

  function copy() {
    var e = env();
    if (e.id === 'standalone') {
      return { title: 'Installed', body: 'Ra-Thor is already running as an app on this device.', cta: 'Installed' };
    }
    if (e.family === 'webview') {
      return {
        title: e.ios ? 'Open in Safari' : 'Open in Chrome',
        body: 'In-app browsers cannot install a real app. Open rathor.ai in ' + (e.ios ? 'Safari' : 'Chrome') + ' and tap Install there.',
        cta: e.ios ? 'Open in Safari' : 'Open in Chrome'
      };
    }
    if (e.family === 'ios') {
      return {
        title: 'Add to Home Screen',
        body: 'Apple has no one-tap install API. Tap Share (the square with the ↑) → Add to Home Screen. Same gold bolt, offline lattice, no store.',
        cta: 'How to add'
      };
    }
    if (e.id === 'safari-mac') {
      return {
        title: 'Add to Dock',
        body: 'In Safari: File → Add to Dock… (Sonoma and later). Ra-Thor opens as its own app window.',
        cta: 'How to add'
      };
    }
    if (e.id === 'fx-desktop') {
      return {
        title: 'Firefox on desktop',
        body: 'Firefox desktop has no built-in PWA install. Bookmark rathor.ai, or open this same page in Chrome / Edge and tap Install for a real app window.',
        cta: 'Open in Chrome'
      };
    }
    if (e.id === 'fx-android') {
      return {
        title: 'Install from Firefox',
        body: 'Firefox menu (⋮) → Install. Or open in Chrome for the system app sheet.',
        cta: 'How to install'
      };
    }
    if (e.id === 'samsung') {
      return {
        title: 'Install Ra-Thor',
        body: deferredPrompt ? 'One tap opens Samsung’s install sheet.' : 'Look for the install icon in the address bar, or the menu item Install page as app.',
        cta: 'Install Ra-Thor'
      };
    }
    return {
      title: 'Install Ra-Thor',
      body: deferredPrompt
        ? 'One tap opens the browser’s system install sheet. Offline-ready. No store.'
        : (e.android
          ? 'When Chrome is ready it will show a system sheet. You can also use ⋮ → Install app.'
          : 'Use the install icon in the address bar, or tap below when the system sheet is ready.'),
      cta: 'Install Ra-Thor'
    };
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
    if (!document.querySelector('meta[name="apple-mobile-web-app-title"]')) {
      var t = document.createElement('meta');
      t.name = 'apple-mobile-web-app-title';
      t.content = 'Ra-Thor';
      head.appendChild(t);
    }
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
      .catch(function () { return null; });
  }

  function setControlState() {
    var installed = isStandalone();
    var c = copy();
    var nodes = document.querySelectorAll('#rathor-hero-install, #rathor-lattice-install, #rathor-pwa-install-btn, [data-rt-pwa-install]');
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (el.tagName === 'INSTALL') continue;
      if (installed) {
        el.textContent = 'Installed';
        el.disabled = true;
        continue;
      }
      el.disabled = false;
      if (el.id === 'rathor-pwa-install-btn') el.textContent = c.cta;
      else el.innerHTML = '<i class="fa-solid fa-download"></i> ' + c.cta;
    }
    var note = document.getElementById('rathor-pwa-status');
    if (note) note.textContent = c.body;
    var title = document.querySelector('#rathor-pwa-slot p.text-sky-200, #rathor-pwa-slot .rt-pwa-title');
    if (title) title.textContent = c.title;
  }

  function closeSheet() {
    var el = document.getElementById('rathor-pwa-hint');
    if (el && el.parentNode) el.parentNode.removeChild(el);
  }

  function showSheet(inner) {
    closeSheet();
    if (!document.body) return;
    var wrap = document.createElement('div');
    wrap.id = 'rathor-pwa-hint';
    wrap.setAttribute('role', 'dialog');
    wrap.setAttribute('aria-label', 'Install Ra-Thor');
    wrap.className =
      'fixed bottom-4 left-4 right-4 sm:left-auto sm:right-6 sm:max-w-sm z-[9999] ' +
      'bg-zinc-950 border border-sky-300/40 rounded-2xl p-4 text-sm text-white/80 shadow-2xl';
    wrap.innerHTML = inner +
      '<button type="button" class="mt-3 text-sky-300 underline text-xs" id="rathor-pwa-sheet-close">Close</button>';
    document.body.appendChild(wrap);
    var x = document.getElementById('rathor-pwa-sheet-close');
    if (x) x.addEventListener('click', closeSheet);
    setTimeout(closeSheet, 24000);
  }

  function iosCoach() {
    showSheet(
      '<p class="text-sky-200 font-semibold mb-2">Add Ra-Thor to Home Screen</p>' +
      '<ol class="list-decimal pl-5 space-y-1.5 text-white/75 text-[13px] leading-relaxed">' +
      '<li>Tap the <strong>Share</strong> button (square with ↑).</li>' +
      '<li>Scroll and tap <strong>Add to Home Screen</strong>.</li>' +
      '<li>Tap <strong>Add</strong>. The gold bolt opens as its own app.</li>' +
      '</ol>'
    );
  }

  function fireNativeInstall() {
    var ev = deferredPrompt || (window.__rtPwa && window.__rtPwa.ev);
    if (!ev || typeof ev.prompt !== 'function') return Promise.resolve(false);
    deferredPrompt = null;
    if (window.__rtPwa) window.__rtPwa.ev = null;
    try { ev.prompt(); } catch (e) { return Promise.resolve(false); }
    return ev.userChoice.then(function (choice) {
      var accepted = !!(choice && choice.outcome === 'accepted');
      if (accepted) dismissed = true;
      setControlState();
      return accepted;
    }).catch(function () {
      setControlState();
      return false;
    });
  }

  function tryNavigatorInstall() {
    if (typeof navigator.install !== 'function') return Promise.resolve(false);
    try {
      return Promise.resolve(navigator.install()).then(function () { return true; }).catch(function () { return false; });
    } catch (e) {
      return Promise.resolve(false);
    }
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
        waited += 120;
        if (window.__rtPwa && window.__rtPwa.ev) deferredPrompt = window.__rtPwa.ev;
        if (deferredPrompt) {
          clearInterval(tick);
          resolve(true);
        } else if (waited >= ms) {
          clearInterval(tick);
          resolve(false);
        }
      }, 120);
    });
  }

  function triggerInstall() {
    var e = env();
    if (e.id === 'standalone') {
      setControlState();
      showSheet('<p class="text-sky-200 font-semibold mb-1">Already installed</p><p>You are in the app window.</p>');
      return;
    }
    if (e.family === 'ios') { iosCoach(); return; }
    if (e.id === 'safari-mac') {
      showSheet('<p class="text-sky-200 font-semibold mb-1">Add to Dock</p><p>Safari menu <strong>File → Add to Dock…</strong> then confirm. Works on macOS Sonoma and later.</p>');
      return;
    }
    if (e.family === 'webview') {
      if (e.ios) {
        showSheet('<p class="text-sky-200 font-semibold mb-1">Open in Safari</p><p>Copy rathor.ai and open it in Safari, then Share → Add to Home Screen.</p>');
        return;
      }
      showSheet(
        '<p class="text-sky-200 font-semibold mb-1">Open in Chrome</p>' +
        '<p>This in-app view cannot create a real app. Chrome can.</p>' +
        '<a href="' + chromeIntent() + '" class="mt-3 inline-flex px-4 py-2 rounded-xl bg-sky-300 text-black text-sm font-semibold">Open rathor.ai in Chrome</a>'
      );
      return;
    }
    if (e.id === 'fx-desktop') {
      showSheet(
        '<p class="text-sky-200 font-semibold mb-1">Firefox desktop</p>' +
        '<p>Firefox does not install PWAs. Bookmark this page, or open <strong>rathor.ai</strong> in Chrome or Edge and use Install for a standalone app.</p>'
      );
      return;
    }
    if (e.id === 'fx-android') {
      showSheet(
        '<p class="text-sky-200 font-semibold mb-1">Firefox on Android</p>' +
        '<p>Menu <strong>⋮ → Install</strong>. For the cleanest system app, open the same URL in Chrome.</p>' +
        '<a href="' + chromeIntent() + '" class="mt-3 inline-flex px-4 py-2 rounded-xl bg-sky-300 text-black text-sm font-semibold">Open in Chrome</a>'
      );
      return;
    }

    registerServiceWorker()
      .then(function () { return tryNavigatorInstall(); })
      .then(function (ok) {
        if (ok) return true;
        return waitForPrompt(2800).then(function (ready) {
          if (ready || deferredPrompt || (window.__rtPwa && window.__rtPwa.ev)) return fireNativeInstall();
          return false;
        });
      })
      .then(function (ok) {
        if (ok) return;
        var e2 = env();
        if (e2.id === 'samsung') {
          showSheet('<p class="text-sky-200 font-semibold mb-1">Samsung Internet</p><p>Tap the <strong>install / +</strong> control in the address bar, or the menu item <strong>Install page as app</strong>.</p>');
          return;
        }
        if (e2.android || e2.id === 'chromium-android') {
          showSheet('<p class="text-sky-200 font-semibold mb-1">Chrome install</p><p>Tap <strong>⋮ → Install app</strong> (not “Add to Home screen”). If an old shortcut with a Chrome badge exists, delete it first.</p>');
          return;
        }
        showSheet('<p class="text-sky-200 font-semibold mb-1">Install from the address bar</p><p>Look for the install icon on the right of the URL bar, or the browser menu item <strong>Install Ra-Thor</strong>.</p>');
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
      setControlState();
      var slot = document.getElementById('rathor-pwa-slot');
      if (slot) slot.style.display = 'none';
      return;
    }
    var slot2 = document.getElementById('rathor-pwa-slot');
    if (slot2) {
      var title = slot2.querySelector('p.font-semibold, p.text-sky-200');
      if (title) title.classList.add('rt-pwa-title');
    }
    wireButton(document.getElementById('rathor-hero-install'));
    wireButton(document.getElementById('rathor-lattice-install'));
    var extras = document.querySelectorAll('[data-rt-pwa-install]');
    for (var i = 0; i < extras.length; i++) wireButton(extras[i]);
    setControlState();
  }

  window.addEventListener('beforeinstallprompt', function (e) {
    e.preventDefault();
    deferredPrompt = e;
    window.__rtPwa.ev = e;
    setControlState();
    if (!dismissed && !isStandalone() && env().family === 'chromium') {
      setTimeout(function () {
        if (!dismissed && !isStandalone() && deferredPrompt && !document.getElementById('rathor-pwa-hint')) {
          var c = copy();
          showSheet(
            '<p class="text-sky-200 font-semibold mb-1">' + c.title + '</p>' +
            '<p class="text-white/70 text-[13px] leading-relaxed">' + c.body + '</p>' +
            '<button type="button" id="rathor-pwa-install-btn" class="mt-3 px-4 py-2 rounded-xl bg-sky-300 text-black text-sm font-semibold">Install Ra-Thor</button>'
          );
          wireButton(document.getElementById('rathor-pwa-install-btn'));
        }
      }, 10000);
    }
  });

  window.addEventListener('appinstalled', function () {
    deferredPrompt = null;
    if (window.__rtPwa) window.__rtPwa.ev = null;
    dismissed = true;
    closeSheet();
    setControlState();
  });

  window.rathorTriggerPWAInstall = triggerInstall;
  ensureHeadTags();
  registerServiceWorker();
  if (document.body) mountSiteCta();
  else document.addEventListener('DOMContentLoaded', mountSiteCta);
})();
