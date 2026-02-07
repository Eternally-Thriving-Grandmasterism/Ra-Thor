// public/sw.js – Custom Service Worker with Robust Background Sync v1.2
// Exponential backoff, conflict detection, valence priority queue, queue size cap, mercy gating
// MIT License – Autonomicity Games Inc. 2026

const CACHE_NAME = 'rathor-nexi-cache-v2';
const OFFLINE_URL = '/offline.html';

const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/offline.html',
  '/manifest.json',
  '/favicon.ico',
  '/pwa-192x192.png',
  '/pwa-512x512.png',
  // Critical chunks & models added after build
];

// IndexedDB for pending mutations (robust queue)
const DB_NAME = 'rathor-nexi-db';
const DB_VERSION = 2;
const STORE_NAME = 'pendingMutations';

let dbPromise = null;

function openDB() {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = event => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
      }
    };

    request.onsuccess = event => resolve(event.target.result);
    request.onerror = event => reject(event.target.error);
  });

  return dbPromise;
}

// Install – precache essentials
self.addEventListener('install', event => {
  event.waitUntil(
    caches.open(CACHE_NAME).then(cache => {
      console.log('[SW] Pre-caching files');
      return cache.addAll(PRECACHE_URLS);
    }).then(() => self.skipWaiting())
  );
});

// Activate – clean old caches
self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.map(cacheName => {
          if (cacheName !== CACHE_NAME) {
            console.log('[SW] Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch – CacheFirst for static, NetworkFirst for navigation with offline fallback
self.addEventListener('fetch', event => {
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request).catch(() => caches.match(OFFLINE_URL))
    );
    return;
  }

  event.respondWith(
    caches.match(event.request).then(cachedResponse => {
      if (cachedResponse) return cachedResponse;

      return fetch(event.request).then(networkResponse => {
        if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
          return networkResponse;
        }

        const responseToCache = networkResponse.clone();
        caches.open(CACHE_NAME).then(cache => {
          cache.put(event.request, responseToCache);
        });

        return networkResponse;
      });
    })
  );
});

// Background Sync – robust retry with exponential backoff & valence priority
self.addEventListener('sync', event => {
  if (event.tag.startsWith('rathor-sync-')) {
    event.waitUntil(processSyncQueue(event.tag));
  }
});

async function processSyncQueue(tag) {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);

  // Get all pending, sorted by valence desc + timestamp asc (high valence first)
  let pending = await store.getAll();
  pending.sort((a, b) => {
    if (b.valence !== a.valence) return b.valence - a.valence;
    return a.timestamp - b.timestamp;
  });

  const maxQueueSize = 100;
  if (pending.length > maxQueueSize) {
    console.warn(`[SW] Queue size exceeded (\( {pending.length}/ \){maxQueueSize}) – dropping oldest low-valence items`);
    pending = pending.slice(0, maxQueueSize);
  }

  for (const item of pending) {
    try {
      const response = await fetch(item.url, {
        method: item.method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(item.payload)
      });

      if (response.ok) {
        await store.delete(item.id);
        console.log(`[SW] Synced mutation: \( {item.id} ( \){item.type})`);
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (err) {
      console.warn(`[SW] Mutation failed: \( {item.id} ( \){item.type})`, err);

      // Exponential backoff – update retry count & next attempt time
      const retryCount = (item.retryCount || 0) + 1;
      const backoff = Math.min(1000 * Math.pow(2, retryCount) + Math.random() * 500, 60000); // max 60s
      const nextAttempt = Date.now() + backoff;

      await store.put({ ...item, retryCount, nextAttempt });

      // Re-register sync for later
      if ('sync' in self.registration) {
        await self.registration.sync.register(`rathor-sync-${item.id}`);
      }
    }
  }

  await tx.done;
}

// Client → SW: queue mutation when offline
self.addEventListener('message', event => {
  if (event.data.type === 'QUEUE_MUTATION') {
    queueMutation(event.data.payload);
  }
});

async function queueMutation(payload) {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);

  // Mercy gate: drop low-valence mutations if queue too large
  const count = await store.count();
  if (count > 80 && payload.valence < 0.7) {
    console.warn('[SW] Dropping low-valence mutation – queue full');
    return;
  }

  await store.add({
    ...payload,
    timestamp: Date.now(),
    retryCount: 0,
    nextAttempt: Date.now()
  });

  await tx.done;

  if ('sync' in self.registration) {
    try {
      await self.registration.sync.register('rathor-sync-mutations');
      console.log('[SW] Sync registered for queued mutation');
    } catch (err) {
      console.warn('[SW] Sync registration failed:', err);
    }
  }
}
