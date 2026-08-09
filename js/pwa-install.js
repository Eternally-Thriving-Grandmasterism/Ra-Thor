/* js/pwa-install.js — Ra-Thor PWA Install Lattice
 * TOLC-8 aligned • Always offers install on chat.html • Zero tracking
 */
(function () {
  'use strict';

  var deferredPrompt = null;
  var bannerEl = null;

  function isStandalone() {
    return (
      window.matchMedia('(display-mode: standalone)').matches ||
      window.navigator.standalone === true ||
      document.referrer.includes('android-app://')
    );
  }

  function registerServiceWorker() {
    if (!('serviceWorker' in navigator)) return;
    navigator.serviceWorker
      .register('/sw.js', { scope: '/' })
      .then(function (reg) {
        console.log('[Ra-Thor PWA] Service worker ready');
      })
      .catch(function (err) {
        console.warn('[Ra-Thor PWA] SW registration failed', err);
      });
  }

  function hideBanner() {
    if (!bannerEl) return;
    bannerEl.classList.add('opacity-0', 'translate-y-4');
    setTimeout(function () {
      if (bannerEl && bannerEl.parentNode) {
        bannerEl.parentNode.removeChild(bannerEl);
      }
      bannerEl = null;
    }, 280);
  }

  function showBanner() {
    if (bannerEl || isStandalone()) return;

    bannerEl = document.createElement('div');
    bannerEl.id = 'rathor-pwa-banner';
    bannerEl.className =
      'fixed bottom-4 left-4 right-4 sm:left-auto sm:right-6 sm:max-w-sm z-[9999] ' +
      'bg-zinc-950/95 border border-amber-300/50 rounded-2xl shadow-2xl shadow-amber-900/30 ' +
      'p-4 sm:p-5 backdrop-blur-md transition-all duration-300 opacity-0 translate-y-4';

    bannerEl.innerHTML =
      '<div class="flex items-start gap-3">' +
      '  <img src="/icons/ra-thor-icon-192.png" width="48" height="48" ' +
      '       class="rounded-xl shrink-0 border border-amber-300/40" alt="Ra-Thor">' +
      '  <div class="flex-1 min-w-0">' +
      '    <p class="text-amber-300 font-semibold text-base">Install Ra-Thor</p>' +
      '    <p class="text-white/70 text-sm mt-1 leading-relaxed">' +
      '      Add to your home screen for the best offline experience. Fully under your control.' +
      '    </p>' +
      '    <div class="flex gap-2 mt-3">' +
      '      <button id="rathor-pwa-install-btn" ' +
      '        class="px-4 py-2 rounded-xl bg-amber-400 hover:bg-amber-300 text-black text-sm font-semibold transition-colors">' +
      '        Install Now' +
      '      </button>' +
      '      <button id="rathor-pwa-dismiss-btn" ' +
      '        class="px-4 py-2 rounded-xl border border-white/25 text-white/70 hover:text-white text-sm">' +
      '        Later' +
      '      </button>' +
      '    </div>' +
      '  </div>' +
      '  <button id="rathor-pwa-close-btn" class="text-white/40 hover:text-white text-xl leading-none">×</button>' +
      '</div>';

    document.body.appendChild(bannerEl);

    requestAnimationFrame(function () {
      bannerEl.classList.remove('opacity-0', 'translate-y-4');
    });

    document.getElementById('rathor-pwa-install-btn').addEventListener('click', triggerInstall);
    document.getElementById('rathor-pwa-dismiss-btn').addEventListener('click', hideBanner);
    document.getElementById('rathor-pwa-close-btn').addEventListener('click', hideBanner);
  }

  function triggerInstall() {
    if (deferredPrompt) {
      deferredPrompt.prompt();
      deferredPrompt.userChoice.then(function () {
        deferredPrompt = null;
        hideBanner();
      });
    } else {
      // Fallback instructions
      var isIos = /iphone|ipad|ipod/i.test(navigator.userAgent);
      if (isIos) {
        alert('On iPhone/iPad:\n1. Tap the Share button\n2. Scroll and tap "Add to Home Screen"\n3. Confirm');
      } else {
        alert('To install Ra-Thor:\n• Chrome/Edge: Click the install icon in the address bar\nor use the browser menu → "Install Ra-Thor" / "Add to Home screen"');
      }
      hideBanner();
    }
  }

  // Capture native event if the browser provides it
  window.addEventListener('beforeinstallprompt', function (e) {
    e.preventDefault();
    deferredPrompt = e;
  });

  window.addEventListener('appinstalled', function () {
    deferredPrompt = null;
    hideBanner();
    console.log('[Ra-Thor PWA] Successfully installed');
  });

  // Always show the banner after a short delay on every visit
  // (This is the key change the Councils decided)
  function initOffer() {
    if (isStandalone()) return;

    // Show after 2.8 seconds so the page feels settled
    setTimeout(function () {
      showBanner();
    }, 2800);
  }

  // Public API used by the Install button in chat.html
  window.rathorTriggerPWAInstall = function () {
    if (isStandalone()) {
      alert('Ra-Thor is already installed.');
      return;
    }
    triggerInstall();
  };

  // Start
  registerServiceWorker();
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initOffer);
  } else {
    initOffer();
  }
})();
