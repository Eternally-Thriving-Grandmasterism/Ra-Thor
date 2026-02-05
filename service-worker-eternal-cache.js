// service-worker-eternal-cache.js – sovereign dynamic precache for models/assets v1
// Hooks into sw.js or standalone; precache WebLLM/Transformers post-download
// MIT License – Autonomicity Games Inc. 2026

// Assume imported in sw.js or as module; use self.addEventListener('install', ...) pattern

const CACHE_NAME = 'rathor-eternal-models-v1';
const PRECACHE_ASSETS = [
  '/', // shell
  '/index.html',
  '/main.js',
  '/rathor-chat-ui-streaming.js',
  // ... other statics from manifest
];

// Dynamic model URLs (post-download, add via message or fetch event)
let dynamicModelUrls = []; // e.g., WebLLM model blobs from CacheStorage keys or known URLs

async function addToEternalCache(urls) {
  const cache = await caches.open(CACHE_NAME);
  try {
    await cache.addAll(urls);
    console.log("[EternalCache] Precaced models eternally:", urls);
  } catch (err) {
    console.error("[EternalCache] Precace failed:", err);
  }
}

// Listen for messages from main thread (post-download trigger)
self.addEventListener('message', (event) => {
  if (event.data.type === 'PRECACHE_MODELS') {
    dynamicModelUrls = event.data.urls || [];
    addToEternalCache(dynamicModelUrls);
  }
});

// Install: precache static shell
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE_ASSETS))
  );
  self.skipWaiting();
});

// Activate: claim clients
self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

// Fetch: serve from cache first, fallback network + dynamic add
self.addEventListener('fetch', (event) => {
  if (event.request.url.includes('mlc.ai') || event.request.url.includes('huggingface.co') || event.request.url.endsWith('.bin') || event.request.url.endsWith('.gguf')) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          if (!response || response.status !== 200 || response.type !== 'basic') return response;
          const responseToCache = response.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(event.request, responseToCache));
          return response;
        });
      })
    );
  } else {
    event.respondWith(
      caches.match(event.request).then((response) => response || fetch(event.request))
    );
  }
});

// Example: from main.js after model load
// navigator.serviceWorker.controller.postMessage({ type: 'PRECACHE_MODELS', urls: ['https://.../phi-3-model.bin'] });
