importScripts('https://storage.googleapis.com/workbox-cdn/releases/6.5.4/workbox-sw.js');

workbox.setConfig({ debug: false }); // Set true during dev

// Precache manifest (auto-generated or manual)
workbox.precaching.precacheAndRoute(self.__WB_MANIFEST || []);

// StaleWhileRevalidate for pages/navigation (fast + fresh)
workbox.routing.registerRoute(
  ({ request }) => request.mode === 'navigate',
  async ({ event }) => {
    try {
      return await workbox.strategies.staleWhileRevalidate({
        cacheName: 'rathor-pages',
        plugins: [
          new workbox.cacheableResponse.CacheableResponsePlugin({ statuses: [0, 200] })
        ]
      }).handle({ event });
    } catch (err) {
      // Mercy fallback: serve offline.html
      return caches.match('/offline.html') || new Response(
        '<h1>Offline Mercy Thunder</h1><p>Lattice resting. Reconnect to awaken.</p>',
        { headers: { 'Content-Type': 'text/html' } }
      );
    }
  }
);

// CacheFirst for static assets (images, icons, thunder assets)
workbox.routing.registerRoute(
  /\.(?:png|jpg|jpeg|svg|gif|ico|woff2?|ttf)$/,
  new workbox.strategies.CacheFirst({
    cacheName: 'rathor-static',
    plugins: [
      new workbox.expiration.ExpirationPlugin({
        maxEntries: 100,
        maxAgeSeconds: 60 * 24 * 60 * 60, // 60 days
      })
    ]
  })
);

// NetworkFirst for any potential dynamic (future API)
workbox.routing.registerRoute(
  /\/api\//,
  new workbox.strategies.NetworkFirst({
    cacheName: 'rathor-dynamic',
    networkTimeoutSeconds: 3
  })
);

// Install → skip waiting
self.addEventListener('install', event => {
  self.skipWaiting();
});

// Activate → claim clients
self.addEventListener('activate', event => {
  event.waitUntil(self.clients.claim());
});
