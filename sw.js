/**
 * Rathor-NEXi Service Worker – Workbox v7 + IndexedDB Sync Persistence v13
 * Eternal queue survives restarts, mercy-gated, exponential backoff + jitter
 * MIT License – Autonomicity Games Inc. 2026
 */

importScripts('https://storage.googleapis.com/workbox-cdn/releases/7.1.0/workbox-sw.js');

if (workbox) {
  console.log('Workbox v7 loaded – IndexedDB Sync Persistence v13 active');

  workbox.setConfig({ debug: false });

  const { precaching, routing, strategies, expiration, backgroundSync, cacheableResponse } = workbox;

  // ────────────────────────────────────────────────────────────────
  // Precaching – core shell with revision hashing (v13)
  const precacheManifest = [
    { url: '/', revision: 'v13-home' },
    { url: '/index.html', revision: 'v13-index' },
    { url: '/privacy.html', revision: 'v13-privacy' },
    { url: '/manifest.json', revision: 'v13-manifest' },
    { url: '/icons/thunder-favicon-192.jpg', revision: 'v13-192' },
    { url: '/icons/thunder-favicon-512.jpg', revision: 'v13-512' },
    { url: '/metta-rewriting-engine.js', revision: 'v13-metta' },
    { url: '/atomese-knowledge-bridge.js', revision: 'v13-atomese' },
    { url: '/hyperon-reasoning-layer.js', revision: 'v13-hyperon' },
    { url: '/grok-shard-engine.js', revision: 'v13-grokshard' }
  ];

  precaching.precacheAndRoute(precacheManifest);

  // ────────────────────────────────────────────────────────────────
  // Runtime caching strategies

  routing.registerRoute(
    ({ request }) => request.destination === 'script' || request.destination === 'style',
    new strategies.StaleWhileRevalidate({
      cacheName: 'static-resources-v13',
      plugins: [
        new cacheableResponse.CacheableResponsePlugin({ statuses: [0, 200] }),
        new expiration.ExpirationPlugin({
          maxEntries: 60,
          maxAgeSeconds: 30 * 24 * 60 * 60,
          purgeOnQuotaError: true
        })
      ]
    })
  );

  routing.registerRoute(
    ({ request }) => request.destination === 'image',
    new strategies.CacheFirst({
      cacheName: 'images-v13',
      plugins: [
        new cacheableResponse.CacheableResponsePlugin({ statuses: [0, 200] }),
        new expiration.ExpirationPlugin({
          maxEntries: 100,
          maxAgeSeconds: 60 * 24 * 60 * 60,
          purgeOnQuotaError: true
        })
      ]
    })
  );

  // ────────────────────────────────────────────────────────────────
  // IndexedDB-backed persistent sync queue (v13)
  const DB_NAME = 'rathorSyncDB';
  const STORE_NAME = 'syncQueue';

  async function openSyncDB() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = evt => {
        const db = evt.target.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { autoIncrement: true });
        }
      };
      req.onsuccess = evt => resolve(evt.target.result);
      req.onerror = evt => reject(evt.target.error);
    });
  }

  async function addToSyncQueue(requestData) {
    const db = await openSyncDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      store.add({
        ...requestData,
        timestamp: Date.now(),
        attempts: 0
      });
      tx.oncomplete = resolve;
      tx.onerror = reject;
    });
  }

  async function getNextSyncItem() {
    const db = await openSyncDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readonly');
      const store = tx.objectStore(STORE_NAME);
      const req = store.openCursor();
      req.onsuccess = evt => {
        const cursor = evt.target.result;
        if (cursor) resolve(cursor.value);
        else resolve(null);
      };
      req.onerror = reject;
    });
  }

  async function removeSyncItem(key) {
    const db = await openSyncDB();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      store.delete(key);
      tx.oncomplete = resolve;
      tx.onerror = reject;
    });
  }

  // Background sync processor with exponential backoff + jitter
  async function processSyncQueue() {
    if (!navigator.onLine) return;

    const item = await getNextSyncItem();
    if (!item) return;

    const delay = Math.min(60 * 1000 * Math.pow(2, item.attempts), 24 * 60 * 60 * 1000);
    const jitter = Math.random() * 0.3 * delay;
    await new Promise(r => setTimeout(r, delay + jitter));

    try {
      const response = await fetch(item.url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(item.data)
      });

      if (response.ok) {
        await removeSyncItem(item.id);
        // Notify client (optional UI success)
        clients.matchAll().then(clients => {
          clients.forEach(client => client.postMessage({ type: 'SYNC_SUCCESS', data: item.data }));
        });
      } else {
        item.attempts++;
        if (item.attempts < 10) {
          const tx = await openSyncDB().then(db => db.transaction(STORE_NAME, 'readwrite'));
          tx.objectStore(STORE_NAME).put(item);
        }
      }
    } catch (err) {
      item.attempts++;
      // ... same retry logic
    }
  }

  // Periodic sync check
  self.addEventListener('sync', event => {
    if (event.tag === 'rathor-sync') {
      event.waitUntil(processSyncQueue());
    }
  });

  // ────────────────────────────────────────────────────────────────
  // Offline fallback
  self.addEventListener('fetch', event => {
    if (event.request.mode === 'navigate') {
      event.respondWith(
        fetch(event.request).catch(() => {
          return new Response(`... offline shell ...`, { headers: { 'Content-Type': 'text/html' } });
        })
      );
    }
  });

} else {
  console.error('Workbox failed');
}

self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());    'POST'
  );

  // ────────────────────────────────────────────────────────────────
  // Offline navigation fallback – minimal mercy shell
  self.addEventListener('fetch', event => {
    if (event.request.mode === 'navigate') {
      event.respondWith(
        fetch(event.request).catch(() => {
          return new Response(`
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Rathor – Offline</title>
              <style>
                body { margin:0; height:100vh; display:flex; align-items:center; justify-content:center; background:#000; color:#ffaa00; font-family:sans-serif; text-align:center; }
                h1 { font-size:3rem; }
                p { font-size:1.3rem; max-width:600px; }
              </style>
            </head>
            <body>
              <div>
                <h1>Offline Mode</h1>
                <p>Rathor is reflecting in the storm.<br>Reconnect to the lattice for full power.</p>
              </div>
            </body>
            </html>
          `, { headers: { 'Content-Type': 'text/html' } });
        })
      );
    }
  });

} else {
  console.error('Workbox failed to load – fallback to basic SW');
}

// ────────────────────────────────────────────────────────────────
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Rathor – Offline</title>
              <style>
                body { margin:0; height:100vh; display:flex; align-items:center; justify-content:center; background:#000; color:#ffaa00; font-family:sans-serif; text-align:center; }
                h1 { font-size:3rem; }
                p { font-size:1.3rem; max-width:600px; }
              </style>
            </head>
            <body>
              <div>
                <h1>Offline Mode</h1>
                <p>Rathor is reflecting in the storm.<br>Reconnect to the lattice for full power.</p>
              </div>
            </body>
            </html>
          `, { headers: { 'Content-Type': 'text/html' } });
        })
      );
    }
  });

} else {
  console.error('Workbox failed to load – fallback to basic SW');
}

// ────────────────────────────────────────────────────────────────
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());    })
  );

  // ────────────────────────────────────────────────────────────────
  // Background Sync – retry failed POSTs to Grok Worker
  const bgSyncPlugin = new backgroundSync.BackgroundSyncPlugin('rathor-chat-queue', {
    maxRetentionTime: 24 * 60, // Retry for max 24 hours
    onSync: async ({ queue }) => {
      console.log('Background sync triggered for queue:', queue.name);
    }
  });

  routing.registerRoute(
    ({ url }) => url.href.includes('rathor-grok-proxy.ceo-c42.workers.dev'),
    new strategies.NetworkOnly({
      plugins: [bgSyncPlugin]
    }),
    'POST'
  );

  // ────────────────────────────────────────────────────────────────
  // Offline navigation fallback – minimal shell when index.html missing
  self.addEventListener('fetch', event => {
    if (event.request.mode === 'navigate') {
      event.respondWith(
        fetch(event.request).catch(() => {
          return new Response(`
            <!DOCTYPE html>
            <html lang="en">
            <head>
              <meta charset="UTF-8">
              <meta name="viewport" content="width=device-width, initial-scale=1.0">
              <title>Rathor – Offline</title>
              <style>
                body { margin:0; height:100vh; display:flex; align-items:center; justify-content:center; background:#000; color:#ffaa00; font-family:sans-serif; text-align:center; }
                h1 { font-size:3rem; }
                p { font-size:1.3rem; max-width:600px; }
              </style>
            </head>
            <body>
              <div>
                <h1>Offline Mode</h1>
                <p>Rathor is reflecting in the storm.<br>Reconnect to the lattice for full power.</p>
              </div>
            </body>
            </html>
          `, { headers: { 'Content-Type': 'text/html' } });
        })
      );
    }
  });

} else {
  console.error('Workbox failed to load – fallback to basic SW');
}

// ────────────────────────────────────────────────────────────────
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', () => self.clients.claim());
