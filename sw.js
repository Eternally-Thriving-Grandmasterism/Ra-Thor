// sw.js — Ra-Thor mercy-gated offline worker
// Workspace 14.15.6 · family walk lock 2026-08-23-pwa
// Contact: info@Rathor.ai

importScripts('https://storage.googleapis.com/workbox-cdn/releases/7.1.0/workbox-sw.js');

workbox.setConfig({ debug: false });

var LOCK = '2026-08-23-pwa';

workbox.precaching.precacheAndRoute(self.__WB_MANIFEST || [
  { url: '/index.html', revision: LOCK },
  { url: '/chat.html', revision: LOCK },
  { url: '/contact.html', revision: LOCK },
  { url: '/privacy.html', revision: LOCK },
  { url: '/offline.html', revision: LOCK },
  { url: '/thanks.html', revision: LOCK },
  { url: '/go-x.html', revision: LOCK },
  { url: '/Launch-Ra-Thor.html', revision: LOCK },
  { url: '/sovereign-shard.html', revision: LOCK },
  { url: '/web-forge.html', revision: LOCK },
  { url: '/manifest.json', revision: LOCK },
  { url: '/js/family-nav-2026-08-22.js', revision: LOCK },
  { url: '/js/site-lock-2026-08-22.js', revision: LOCK },
  { url: '/js/pwa-install.js', revision: LOCK },
  { url: '/js/sovereign-shard.js', revision: LOCK },
  { url: '/js/chat.js', revision: LOCK },
  { url: '/js/contact-i18n.js', revision: LOCK },
  { url: '/icons/ra-thor-icon-192.png', revision: LOCK },
  { url: '/icons/ra-thor-icon-512.png', revision: LOCK }
]);

workbox.routing.setCatchHandler(function (args) {
  if (args.event && args.event.request && args.event.request.destination === 'document') {
    return caches.match('/offline.html');
  }
  return Response.error();
});

workbox.routing.registerRoute(
  function (ctx) { return ctx.request.destination === 'document'; },
  new workbox.strategies.NetworkFirst({
    cacheName: 'rathor-pages-' + LOCK,
    networkTimeoutSeconds: 3,
    plugins: [
      new workbox.expiration.ExpirationPlugin({
        maxEntries: 20,
        maxAgeSeconds: 24 * 60 * 60
      })
    ]
  })
);

workbox.routing.registerRoute(
  /\.(?:png|jpg|jpeg|svg|gif|ico|woff2?|ttf|css)$/,
  new workbox.strategies.CacheFirst({
    cacheName: 'rathor-static-' + LOCK,
    plugins: [
      new workbox.expiration.ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 60 * 24 * 60 * 60
      })
    ]
  })
);

workbox.routing.registerRoute(
  /\/js\/.*\.js$/,
  new workbox.strategies.NetworkFirst({
    cacheName: 'rathor-js-' + LOCK,
    networkTimeoutSeconds: 3,
    plugins: [
      new workbox.expiration.ExpirationPlugin({
        maxEntries: 40,
        maxAgeSeconds: 7 * 24 * 60 * 60
      })
    ]
  })
);

workbox.routing.registerRoute(
  function (ctx) { return ctx.url.href.indexOf('@xenova/transformers') !== -1; },
  new workbox.strategies.CacheFirst({
    cacheName: 'rathor-models',
    plugins: [
      new workbox.expiration.ExpirationPlugin({
        maxEntries: 50,
        maxAgeSeconds: 30 * 24 * 60 * 60
      })
    ]
  })
);

var QUEUE_NAME = 'rathor-offline-queue';

self.addEventListener('fetch', function (event) {
  if (event.request.url.indexOf('/api/') !== -1 || event.request.url.indexOf('/sync/') !== -1) {
    event.respondWith(
      fetch(event.request).catch(function () {
        return caches.open('rathor-queue').then(function (cache) {
          cache.put(event.request.url, event.request.clone());
          return self.registration.sync.register(QUEUE_NAME);
        }).then(function () {
          return new Response('Queued for reconnection', { status: 202 });
        });
      })
    );
  }
});

self.addEventListener('sync', function (event) {
  if (event.tag === QUEUE_NAME) {
    event.waitUntil(
      caches.open('rathor-queue').then(function (cache) {
        return cache.keys().then(function (requests) {
          return Promise.all(requests.map(function (req) {
            return fetch(req).then(function (resp) {
              if (resp.ok) return cache.delete(req.url);
            }).catch(function () {});
          }));
        });
      })
    );
  }
});

self.addEventListener('install', function (event) {
  self.skipWaiting();
});

self.addEventListener('activate', function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.map(function (key) {
        if (key.indexOf(LOCK) === -1 && key.indexOf('rathor-models') === -1 && key.indexOf('rathor-queue') === -1 && key.indexOf('workbox-precache') === -1) {
          return caches.delete(key);
        }
      }));
    }).then(function () {
      return self.clients.claim();
    })
  );
});

console.log('[Ra-Thor SW] family lock ' + LOCK + ' • workspace 14.15.6');
