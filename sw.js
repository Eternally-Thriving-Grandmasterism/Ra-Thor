const CACHE_VERSION = 'rathor-cache-v7';
const CACHE_NAME = `rathor-static-${CACHE_VERSION}`;

const urlsToCache = [
  '/',
  '/index.html',
  '/privacy.html',
  '/thanks.html',
  '/manifest.json',
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

self.addEventListener('activate', event => {
  event.waitUntil(
    caches.keys().then(cacheNames => {
      return Promise.all(
        cacheNames.filter(name => name.startsWith('rathor-static-') && name !== CACHE_NAME)
          .map(name => caches.delete(name))
      );
    }).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', event => {
  event.respondWith(
    caches.match(event.request)
      .then(response => response || fetch(event.request))
  );
});

// Regular Background Sync (queued messages)
self.addEventListener('sync', event => {
  if (event.tag === 'sync-chat-messages') {
    event.waitUntil(syncQueuedMessagesWithRetry());
  }
});

// Periodic Background Sync – runs every ~12-24h (browser decides)
self.addEventListener('periodicsync', event => {
  if (event.tag === 'rathor-periodic-sync') {
    event.waitUntil(performPeriodicSync());
  }
});

async function performPeriodicSync() {
  // Example tasks: refresh cached assets, prune old history, check for updates
  const cache = await caches.open(CACHE_NAME);
  await cache.addAll(urlsToCache);  // refresh static files

  // Optional: notify client of sync
  const clients = await self.clients.matchAll();
  clients.forEach(client => client.postMessage({ type: 'periodic-sync-complete' }));

  console.log('Periodic sync completed');
}

// IndexedDB helpers (from previous)
function openDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open('rathorChatDB', 4);
    request.onupgradeneeded = event => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains('messages')) {
        db.createObjectStore('messages', { keyPath: 'id', autoIncrement: true });
      }
      if (!db.objectStoreNames.contains('queuedMessages')) {
        const store = db.createObjectStore('queuedMessages', { keyPath: 'id', autoIncrement: true });
        store.createIndex('retryCount', 'retryCount');
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function syncQueuedMessagesWithRetry() {
  const db = await openDB();
  const tx = db.transaction('queuedMessages', 'readwrite');
  const store = tx.objectStore('queuedMessages');
  const messages = await store.getAll();

  if (messages.length === 0) return;

  let backoff = 1000;
  for (const msg of messages) {
    const retryCount = msg.retryCount || 0;
    if (retryCount >= 10) continue;

    try {
      const response = await fetch(msg.url || 'https://YOUR_WORKER_URL', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(msg.payload)
      });

      if (response.ok) {
        await store.delete(msg.id);
      } else {
        msg.retryCount = retryCount + 1;
        await store.put(msg);
        await new Promise(r => setTimeout(r, backoff));
        backoff = Math.min(backoff * 2, 1800000);
      }
    } catch (err) {
      break;
    }
  }
}
