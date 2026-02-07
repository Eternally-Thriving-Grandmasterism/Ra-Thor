// src/storage/rathor-indexeddb.js — Optimized Mercy-gated IndexedDB wrapper

const DB_NAME = 'rathorDB';
const DB_VERSION = 3; // bump when adding indexes or stores
const STORES = {
  sessions: 'sessions',
  messages: 'messages',
  tags: 'tags',
  translationCache: 'translationCache'
};

let db = null;

const dbPromise = new Promise((resolve, reject) => {
  const request = indexedDB.open(DB_NAME, DB_VERSION);

  request.onupgradeneeded = event => {
    const db = event.target.result;
    const oldVersion = event.oldVersion;

    if (oldVersion < 1) {
      // Initial schema
      const sessionStore = db.createObjectStore(STORES.sessions, { keyPath: 'id' });
      sessionStore.createIndex('name', 'name');

      db.createObjectStore(STORES.messages, { autoIncrement: true });
      db.createObjectStore(STORES.tags, { keyPath: 'id' });
      db.createObjectStore(STORES.translationCache, { keyPath: 'key' });
    }

    if (oldVersion < 2) {
      // Add indexes for performance
      const messageStore = db.transaction(STORES.messages, 'readwrite').objectStore(STORES.messages);
      messageStore.createIndex('sessionId', 'sessionId');
      messageStore.createIndex('timestamp', 'timestamp');
      messageStore.createIndex('role', 'role');
    }

    if (oldVersion < 3) {
      // Add compression flag & quota tracking
      // Future migration logic here
    }
  };

  request.onsuccess = event => {
    db = event.target.result;
    db.onerror = err => console.error('IndexedDB error:', err);
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
// Sessions CRUD
// ────────────────────────────────────────────────

export async function saveSession(session) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.sessions, 'readwrite');
    const store = tx.objectStore(STORES.sessions);
    const req = store.put(session);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function getSession(id) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.sessions);
    const store = tx.objectStore(STORES.sessions);
    const req = store.get(id);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

export async function getAllSessions() {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.sessions);
    const store = tx.objectStore(STORES.sessions);
    const req = store.getAll();
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

// ────────────────────────────────────────────────
// Messages CRUD (batched + paginated)
// ────────────────────────────────────────────────

export async function saveMessages(sessionId, messages) {
  const db = await openDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(STORES.messages, 'readwrite');
    const store = tx.objectStore(STORES.messages);
    let count = 0;
    messages.forEach(msg => {
      msg.sessionId = sessionId;
      msg.timestamp = Date.now();
      const req = store.add(msg);
      req.onsuccess = () => { count++; if (count === messages.length) resolve(); };
      req.onerror = () => reject(req.error);
    });
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

    req.onsuccess = event => {
      const cursor = event.target.result;
      if (!cursor) return resolve(results);

      if (skipped < offset) {
        skipped++;
        cursor.continue();
      } else if (results.length < limit) {
        results.push(cursor.value);
        cursor.continue();
      } else {
        resolve(results);
      }
    };

    req.onerror = () => reject(req.error);
  });
}

// ────────────────────────────────────────────────
// Utility: Clear old data (quota management)
// ────────────────────────────────────────────────

export async function clearExpiredCache(days = 30) {
  const db = await openDB();
  const cutoff = Date.now() - days * 24 * 60 * 60 * 1000;
  return new Promise((resolve, reject) => {
    const tx = db.transaction([STORES.messages, STORES.sessions], 'readwrite');
    const msgStore = tx.objectStore(STORES.messages);
    const msgIndex = msgStore.index('timestamp');
    const req = msgIndex.openCursor(IDBKeyRange.upperBound(cutoff));
    req.onsuccess = e => {
      const cursor = e.target.result;
      if (cursor) {
        cursor.delete();
        cursor.continue();
      }
    };
    tx.oncomplete = resolve;
    tx.onerror = reject;
  });
}
