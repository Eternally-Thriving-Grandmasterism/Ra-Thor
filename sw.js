/* sw.js — Ra-Thor vanilla offline worker
 * Workspace 14.15.6 · LOCK 2026-08-23-native2
 * No CDN. No Workbox. Root scope. Contact: info@Rathor.ai
 */
var LOCK = '2026-08-23-native2';
var CACHE = 'rathor-core-' + LOCK;

var PRECACHE = [
  '/',
  '/index.html',
  '/chat.html',
  '/contact.html',
  '/privacy.html',
  '/offline.html',
  '/thanks.html',
  '/go-x.html',
  '/Launch-Ra-Thor.html',
  '/sovereign-shard.html',
  '/web-forge.html',
  '/manifest.json',
  '/js/pwa-install.js',
  '/js/family-nav-2026-08-22.js',
  '/js/site-lock-2026-08-22.js',
  '/js/science-map-lock.js',
  '/js/sovereign-shard.js',
  '/js/chat.js',
  '/js/contact-i18n.js',
  '/icons/ra-thor-icon-192.png',
  '/icons/ra-thor-icon-512.png'
];

self.addEventListener('install', function (event) {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE).then(function (cache) {
      return Promise.all(
        PRECACHE.map(function (url) {
          return cache.add(url).catch(function () { return null; });
        })
      );
    })
  );
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(
        keys.map(function (key) {
          if (key.indexOf(LOCK) === -1 && key.indexOf('rathor-models') === -1 && key.indexOf('rathor-queue') === -1) {
            return caches.delete(key);
          }
        })
      );
    }).then(function () {
      return self.clients.claim();
    })
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
        var copy = res.clone();
        caches.open(CACHE).then(function (cache) { cache.put(req, copy); });
        return res;
      }).catch(function () {
        return caches.match(req).then(function (hit) {
          return hit || caches.match('/index.html') || caches.match('/offline.html');
        });
      })
    );
    return;
  }

  event.respondWith(
    caches.match(req).then(function (hit) {
      if (hit) return hit;
      return fetch(req).then(function (res) {
        if (res && res.ok && (req.destination === 'image' || req.destination === 'script' || req.destination === 'style' || url.pathname.indexOf('/icons/') === 0)) {
          var copy = res.clone();
          caches.open(CACHE).then(function (cache) { cache.put(req, copy); });
        }
        return res;
      });
    })
  );
});

console.log('[Ra-Thor SW] vanilla native lock ' + LOCK + ' • workspace 14.15.6');
