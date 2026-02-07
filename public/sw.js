// public/sw.js – Custom Service Worker with Robust Background Sync v1.4
// Exponential backoff, conflict resolution, valence priority, queue cap, haptic feedback on completion
// MIT License – Autonomicity Games Inc. 2026

const CACHE_NAME = 'rathor-nexi-cache-v3';
const OFFLINE_URL = '/offline.html';

const PRECACHE_URLS = [
  '/',
  '/index.html',
  '/offline.html',
  '/manifest.json',
  '/favicon.ico',
  '/pwa-192x192.png',
  '/pwa-512x512.png',
];

// IndexedDB for pending mutations
const DB_NAME = 'rathor-nexi-db';
const DB_VERSION = 3;
const STORE_NAME = 'pendingMutations';

let dbPromise = null;

function openDB() {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = event => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        const store = db.createObjectStore(STORE_NAME, { keyPath: 'id', autoIncrement: true });
        store.createIndex('type', 'type', { unique: false });
        store.createIndex('correlationId', 'correlationId', { unique: true });
        store.createIndex('nextAttempt', 'nextAttempt', { unique: false });
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

// Background Sync – robust retry with conflict resolution & haptic feedback
self.addEventListener('sync', event => {
  if (event.tag.startsWith('rathor-sync-')) {
    event.waitUntil(processSyncQueue(event.tag));
  }
});

async function processSyncQueue(tag) {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, 'readwrite');
  const store = tx.objectStore(STORE_NAME);

  let pending = await store.getAll();
  pending.sort((a, b) => {
    if (b.valence !== a.valence) return b.valence - a.valence;
    return a.timestamp - b.timestamp;
  });

  const maxQueueSize = 100;
  if (pending.length > maxQueueSize) {
    console.warn(`[SW] Queue size exceeded (\( {pending.length}/ \){maxQueueSize}) – dropping oldest low-valence items`);
    const toDelete = pending.slice(maxQueueSize);
    for (const item of toDelete) {
      await store.delete(item.id);
    }
    pending = pending.slice(0, maxQueueSize);
  }

  let syncedCount = 0;
  let conflictCount = 0;
  let dropCount = 0;

  for (const item of pending) {
    if (item.nextAttempt && item.nextAttempt > Date.now()) continue;

    try {
      const newer = await store.index('correlationId').getAllKeys(item.correlationId);
      if (newer.length > 1 && newer.some(id => id > item.id)) {
        console.log(`[SW] Conflict detected – skipping older mutation ${item.id}`);
        await store.delete(item.id);
        conflictCount++;
        continue;
      }

      const response = await fetch(item.url, {
        method: item.method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(item.payload)
      });

      if (response.ok) {
        await store.delete(item.id);
        syncedCount++;
        console.log(`[SW] Synced mutation: \( {item.id} ( \){item.type})`);
      } else {
        throw new Error(`HTTP ${response.status}`);
      }
    } catch (err) {
      console.warn(`[SW] Mutation failed: \( {item.id} ( \){item.type})`, err);

      const retryCount = (item.retryCount || 0) + 1;
      const backoff = Math.min(1000 * Math.pow(2, retryCount) + Math.random() * 1000, 120000);
      const nextAttempt = Date.now() + backoff;

      await store.put({ ...item, retryCount, nextAttempt });

      if ('sync' in self.registration) {
        await self.registration.sync.register(`rathor-sync-${item.id}`);
      }
    }
  }

  await tx.done;

  // Haptic feedback on sync completion – postMessage to clients
  if (syncedCount > 0 || conflictCount > 0 || dropCount > 0) {
    const clients = await self.clients.matchAll({ type: 'window', includeUncontrolled: true });
    for (const client of clients) {
      client.postMessage({
        type: 'SYNC_COMPLETED',
        payload: {
          synced: syncedCount,
          conflicts: conflictCount,
          dropped: dropCount,
          total: pending.length,
          valence: currentValence.get() // assume client has access or send from SW context
        }
      });
    }
  }
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

  const existing = await store.index('correlationId').getAllKeys(payload.correlationId);
  if (existing.length > 0) {
    const latest = existing.reduce((a, b) => Math.max(a, b));
    if (payload.timestamp > latest.timestamp) {
      await store.delete(latest.id);
    } else {
      console.log('[SW] Duplicate mutation dropped – older version');
      return;
    }
  }

  const count = await store.count();
  if (count > 80 && payload.valence < 0.6) {
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
