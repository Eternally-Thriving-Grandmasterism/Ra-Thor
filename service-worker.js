// service-worker.js
// Sovereign Eternal Cache — makes Ra-Thor fully offline PWA (installable, works forever in airplane mode)

const CACHE_NAME = 'ra-thor-sovereign-v2026';
const urlsToCache = [
  '/',
  '/ra-thor-standalone-demo.html',
  '/core/ra-thor-sovereign-orchestrator.js',
  '/crates/ra-thor-kernel/pkg/ra_thor_kernel_bg.wasm',
  '/core/rbe-economy-simulator-with-convergence.js',
  // Add all other core files here — the kernel will self-update
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
      .then(response => response || fetch(event.request))
  );
});

self.addEventListener('activate', event => {
  event.waitUntil(self.clients.claim());
});
