/* sw.js — Ra-Thor vanilla offline worker
 * Workspace 14.15.6 · LOCK 20260915b
 * Pass-through documents. Fetch handler present for Chrome installability.
 * Contact: info@Rathor.ai
 */
var LOCK = '20260915b';
var CACHE = 'rathor-core-' + LOCK;
var PRECACHE = [
  '/', '/index.html', '/chat.html', '/contact.html', '/privacy.html',
  '/offline.html', '/thanks.html', '/go-x.html', '/Launch-Ra-Thor.html',
  '/micro-moment.html', '/sovereign-shard.html', '/web-forge.html', '/science-watches.html',
  '/constellation-week.html', '/employ.html', '/manifest.json',
  '/mercy-motion-vision-engine.js', '/fuzzy-mercy-logic.js', '/one-organism-launch.js',
  '/js/pwa-boot.js', '/js/pwa-install.js', '/js/lang-offer.js',
  '/js/rathor-feedback.js', '/js/rathor-theme.js', '/js/rathor-unify.js',
  '/css/rathor-theme.css', '/css/rathor-theme-rest-a.css', '/css/rathor-theme-rest-b.css',
  '/js/family-nav-2026-08-22.js', '/js/site-lock-2026-08-22.js',
  '/js/science-map-lock.js', '/js/watch-footer-lock.js', '/js/sovereign-shard.js', '/js/chat.js',
  '/js/contact-i18n.js', '/js/week-window.js', '/js/week-window.json',
  '/js/google-translate-optin.js',
  '/i18n/en.js', '/i18n/ar.js', '/i18n/es.js', '/i18n/fr.js', '/i18n/nl.js',
  '/i18n/de.js', '/i18n/zh.js', '/i18n/ja.js', '/i18n/pt.js', '/i18n/ru.js',
  '/i18n/hi.js', '/i18n/it.js', '/i18n/ko.js', '/i18n/uk.js', '/i18n/pl.js',
  '/i18n/tr.js', '/i18n/vi.js', '/i18n/id.js', '/i18n/sv.js', '/i18n/th.js',
  '/i18n/el.js', '/i18n/fa.js', '/i18n/he.js',
  '/icons/ra-thor-icon-192.png', '/icons/ra-thor-icon-512.png'
];

self.addEventListener('install', function (event) {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE).then(function (cache) {
      return Promise.all(PRECACHE.map(function (url) {
        return cache.add(url).catch(function () { return null; });
      }));
    })
  );
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (key) {
        if (key.indexOf(LOCK) === -1 && key.indexOf('rathor-models') === -1 && key.indexOf('rathor-queue') === -1) {
          return caches.delete(key);
        }
      }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener('fetch', function (event) {
  var req = event.request;
  if (req.method !== 'GET') return;
  var url;
  try { url = new URL(req.url); } catch (e) { return; }
  if (url.origin !== self.location.origin) return;

  if (req.mode === 'navigate' || req.destination === 'document') {
    event.respondWith(
      fetch(req).then(function (res) {
        return res;
      }).catch(function () {
        return caches.match(req).then(function (hit) {
          return hit || caches.match('/index.html') || caches.match('/offline.html');
        });
      })
    );
    return;
  }

  if (req.destination === 'script' || req.destination === 'style') {
    event.respondWith(
      fetch(req).then(function (res) {
        if (res && res.ok) {
          var copy = res.clone();
          caches.open(CACHE).then(function (cache) { cache.put(req, copy); });
        }
        return res;
      }).catch(function () {
        return caches.match(req);
      })
    );
    return;
  }

  event.respondWith(
    caches.match(req).then(function (hit) {
      if (hit) return hit;
      return fetch(req).then(function (res) {
        if (res && res.ok && req.destination === 'image') {
          var copy = res.clone();
          caches.open(CACHE).then(function (cache) { cache.put(req, copy); });
        }
        return res;
      });
    })
  );
});

console.log('[Ra-Thor SW] lattice lock ' + LOCK + ' • workspace 14.15.6');
