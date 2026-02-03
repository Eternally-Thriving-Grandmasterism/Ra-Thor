/**
 * Rathor-NEXi Service Worker – Workbox v9 Optimized
 * Precaching, runtime strategies, background sync, cleanup
 * MIT License – Autonomicity Games Inc. 2026
 */

importScripts('https://storage.googleapis.com/workbox-cdn/releases/7.1.0/workbox-sw.js');

if (workbox) {
  console.log('Workbox loaded successfully');

  workbox.setConfig({ debug: false }); // set true for dev

  const { precaching, routing, strategies, expiration, backgroundSync } = workbox;

  // ────────────────────────────────────────────────────────────────
  // Precaching – core app shell
  precaching.precacheAndRoute([
    { url: '/', revision: null },
    { url: '/index.html', revision: null },
    { url: '/privacy.html', revision: null },
    { url: '/thanks.html', revision: null },
    { url: '/manifest.json', revision: null },
    { url: '/icons/thunder-favicon-192.jpg', revision: null },
    { url: '/icons/thunder-favicon-512.jpg', revision: null },
    { url: '/metta-rewriting-engine.js', revision: null },
    { url: '/atomese-knowledge-bridge.js', revision: null },
    { url: '/hyperon-reasoning-layer.js', revision: null }
  ]);

  // ────────────────────────────────────────────────────────────────
  // Runtime caching strategies

  // 1. Google fonts (if you ever add) – CacheFirst
  routing.registerRoute(
    ({ url }) => url.origin === 'https://fonts.googleapis.com' || url.origin === 'https://fonts.gstatic.com',
    new strategies.CacheFirst({
      cacheName: 'google-fonts',
      plugins: [
        new expiration.ExpirationPlugin({
          maxEntries: 30,
          maxAgeSeconds: 60 * 60 * 24 * 365 // 1 year
        })
      ]
    })
  );

  // 2. Images – CacheFirst with expiration
  routing.registerRoute(
    ({ request }) => request.destination === 'image',
    new strategies.CacheFirst({
      cacheName: 'images',
      plugins: [
        new expiration.ExpirationPlugin({
          maxEntries: 60,
          maxAgeSeconds: 30 * 24 * 60 * 60 // 30 days
        })
      ]
    })
  );

  // 3. JS/CSS from our domain – StaleWhileRevalidate
  routing.registerRoute(
    ({ request }) => request.destination === 'script' || request.destination === 'style',
    new strategies.StaleWhileRevalidate({
      cacheName: 'static-resources'
    })
  );

  // 4. API / Worker calls – NetworkFirst (fallback to cache if offline)
  routing.registerRoute(
    ({ url }) => url.origin === self.location.origin && url.pathname.includes('grok-proxy'),
    new strategies.NetworkFirst({
      cacheName: 'api-responses',
      plugins: [
        new expiration.ExpirationPlugin({
          maxEntries: 50,
          maxAgeSeconds: 24 * 60 * 60 // 24 hours
        })
      ]
    })
  );

  // ────────────────────────────────────────────────────────────────
  // Background Sync – retry failed POSTs to Grok Worker
  const bgSyncPlugin = new backgroundSync.BackgroundSyncPlugin('rathor-chat-queue', {
    maxRetentionTime: 24 * 60 // Retry for max 24 hours
  });

  routing.registerRoute(
    ({ url }) => url.href.includes('rathor-grok-proxy.ceo-c42.workers.dev'),
    new strategies.NetworkOnly({
      plugins: [bgSyncPlugin]
    }),
    'POST'
  );

} else {
  console.error('Workbox failed to load');
}

// ────────────────────────────────────────────────────────────────
// Fallback for offline navigation – serve index.html
self.addEventListener('fetch', event => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() => caches.match('/index.html'))
    );
  }
});
