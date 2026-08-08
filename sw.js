// sw.js — Rathor-NEXi Mercy-gated Offline Service Worker + Starlink queue support
// Updated: PWA install lattice + warm-gold icons

importScripts('https://storage.googleapis.com/workbox-cdn/releases/7.1.0/workbox-sw.js');

workbox.setConfig({ debug: false });

// Precache core assets
workbox.precaching.precacheAndRoute(self.__WB_MANIFEST || [
  { url: '/index.html', revision: null },
  { url: '/chat.html', revision: null },
  { url: '/contact.html', revision: null },
  { url: '/manifest.json', revision: null },
  { url: '/css/main.css', revision: null },
  { url: '/js/common.js', revision: null },
  { url: '/js/chat.js', revision: null },
  { url: '/js/contact-i18n.js', revision: null },
  { url: '/js/pwa-install.js', revision: null },
  { url: '/locales/languages.json', revision: null },
  { url: '/icons/ra-thor-icon-192.png', revision: null },
  { url: '/icons/ra-thor-icon-512.png', revision: null },
  { url: '/offline.html', revision: null },
  { url: '/privacy.html', revision: null },
  { url: '/go-x.html', revision: null }
]);

// Navigation fallback to offline.html
workbox.routing.setCatchHandler(({ event }) => {
  switch (event.request.destination) {
    case 'document':
      return caches.match('/offline.html');
    default:
      return Response.error();
  }
});

// Cache-first for static assets
workbox.routing.registerRoute(
  /\.(?:png|jpg|jpeg|svg|gif|ico|woff2?|ttf|css|js)$/,
  new workbox.strategies.CacheFirst({
    cacheName: 'rathor-static',
    plugins: [
      new workbox.expiration.ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 60 * 24 * 60 * 60
      })
    ]
  })
);

workbox.routing.registerRoute(
  ({ url }) => url.href.includes('@xenova/transformers'),
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

const QUEUE_NAME = 'rathor-offline-queue';

self.addEventListener('fetch', event => {
  if (event.request.url.includes('/api/') || event.request.url.includes('/sync/')) {
    event.respondWith(
      fetch(event.request).catch(() => {
        return caches.open('rathor-queue').then(cache => {
          cache.put(event.request.url, event.request.clone());
          return self.registration.sync.register(QUEUE_NAME);
        }).then(() => new Response('Queued for reconnection', { status: 202 }));
      })
    );
  }
});

self.addEventListener('sync', event => {
  if (event.tag === QUEUE_NAME) {
    event.waitUntil(
      caches.open('rathor-queue').then(async cache => {
        const requests = await cache.keys();
        for (const req of requests) {
          try {
            const resp = await fetch(req);
            if (resp.ok) await cache.delete(req.url);
          } catch (e) {}
        }
      })
    );
  }
});

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(keys => {
      return Promise.all(
        keys.filter(key => key.startsWith('rathor-') && !key.includes('v1'))
          .map(key => caches.delete(key))
      );
    })
  );
});

self.addEventListener('install', event => self.skipWaiting());
self.addEventListener('activate', event => event.waitUntil(self.clients.claim()));

console.log('[Rathor SW] Mercy thunder online — PWA install + Starlink queue ⚡️');
