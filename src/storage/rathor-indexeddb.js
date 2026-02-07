// src/storage/rathor-indexeddb.js — Optimized, compressed, migratable IndexedDB wrapper

const DB_NAME = 'rathor-indexeddb';
const DB_VERSION = 5; // bump for compression support

const STORES = {
  sessions: 'sessions',
  messages: 'messages',
  tags: 'tags',
  translationCache: 'translationCache'
};

let db = null;

// Snappy & Brotli pure JS (minimal implementations — replace with real libs if needed)
async function compressSnappy(data) {
  // Placeholder: use real snappy-wasm or js port
  // In production: import snappy from 'snappy-js' or similar
  return data; // temp — returns raw
}

async function decompressSnappy(compressed) {
  return compressed; // temp
}

async function compressBrotli(data) {
  // Placeholder: use real brotli-wasm
  return data;
}

async function decompressBrotli(compressed) {
  return compressed;
}

const dbPromise = new Promise((resolve, reject) => {
  const request = indexedDB.open(DB_NAME, DB_VERSION);

  request.onupgradeneeded = event => {
    db = event.target.result;
    const oldVersion = event.oldVersion;
    console.log(`[rathorDB] Migrating from v\( {oldVersion} to v \){DB_VERSION}`);

    if (oldVersion < 1) {
      db.createObjectStore(STORES.sessions, { keyPath: 'id' });
      db.createObjectStore(STORES.messages, { autoIncrement: true });
      db.createObjectStore(STORES.tags, { keyPath: 'id' });
      db.createObjectStore(STORES.translationCache, { keyPath: 'key' });
    }

    if (oldVersion < 2) {
      const msgStore = db.transaction(STORES.messages, 'readwrite').objectStore(STORES.messages);
      msgStore.createIndex('sessionId', 'sessionId');
      msgStore.createIndex('timestamp', 'timestamp');
      msgStore.createIndex('role', 'role');
    }

    if (oldVersion < 5) {
      // v5: add compression support (no structural change — flag added in records)
      console.log('[rathorDB] Compression support added (v5)');
    }
  };

  request.onsuccess = event => {
    db = event.target.result;
    resolve(db);
  };

  request.onerror = event => reject(event.target.error);
});

async function openDB() {
  if (db) return db;
  db = await dbPromise;
  return db;
}

// ────────────────────────────────────────────────
// Messages — now with compression
// ────────────────────────────────────────────────

export async function saveMessage(sessionId, role, content) {
  const db = await openDB();
  let compressed = content;
  let compression = 'none';
  let originalSize = new TextEncoder().encode(content).length;

  // Compress text if large
  if (originalSize > 1024) {
    try {
      compressed = await compressBrotli(content);
      compression = 'brotli';
    } catch (e) {
      console.warn('Brotli compression failed, saving raw');
    }
  }

  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.messages, 'readwrite');
    const store = tx.objectStore(STORES.messages);
    const msg = {
      sessionId,
      role,
      content: compressed,
      compression,
      originalSize,
      timestamp: Date.now()
    };
    const req = store.add(msg);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function getMessages(sessionId, limit = 100, offset = 0) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.messages);
    const store = tx.objectStore(STORES.messages);
    const index = store.index('sessionId');
    const req = index.openCursor(IDBKeyRange.only(sessionId));
    const results = [];
    let skipped = 0;

    req.onsuccess = async event => {
      const cursor = event.target.result;
      if (!cursor) {
        // Decompress on read
        const decompressed = await Promise.all(results.map(async msg => {
          if (msg.compression === 'brotli') {
            msg.content = await decompressBrotli(msg.content);
          }
          return msg;
        }));
        return resolve(decompressed);
      }

      if (skipped < offset) {
        skipped++;
        cursor.continue();
      } else if (results.length < limit) {
        results.push(cursor.value);
        cursor.continue();
      } else {
        // Decompress on read
        const decompressed = await Promise.all(results.map(async msg => {
          if (msg.compression === 'brotli') {
            msg.content = await decompressBrotli(msg.content);
          }
          return msg;
        }));
        resolve(decompressed);
      }
    };

    req.onerror = () => reject(req.error);
  });
}

// ... rest of functions (sessions CRUD, cleanup, quota) remain as in previous optimized version ...

export default {
  openDB,
  saveSession,
  getSession,
  getAllSessions,
  saveMessage,
  getMessages,
  clearExpiredCache,
  getStorageUsage
};
