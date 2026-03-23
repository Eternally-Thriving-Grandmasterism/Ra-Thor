const CACHE_NAME = 'ra-thor-eternal-v2026';
const urlsToCache = [
  '/',
  '/ra-thor-standalone-demo.html',
  '/core/ra-thor-sovereign-orchestrator.js',
  '/crates/ra-thor-kernel/pkg/ra_thor_kernel_bg.wasm',
  '/webllm-mercy-integration.js',
  '/transformer-offline.js',
  '/models/phi-2-q4f16_1-mlc/phi-2-q4f16_1-00001-of-00002.bin', // shard example—add real URLs or local paths
  '/models/phi-2-q4f16_1-mlc/phi-2-q4f16_1-00002-of-00002.bin',
  '/icons/icon-192.png',
  '/icons/icon-512.png'
];

self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(urlsToCache))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request).then(networkRes => {
        if (!networkRes) return networkRes;
        return caches.open(CACHE_NAME).then(cache => {
          cache.put(event.request, networkRes.clone());
          return networkRes;
        });
      }).catch(() => caches.match(event.request)))
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(self.clients.claim());
});
