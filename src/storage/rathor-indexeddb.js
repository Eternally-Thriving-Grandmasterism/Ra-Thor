const DB_NAME = 'RathorNEXiDB';
const DB_VERSION = 1;
const STORES = {
  CHAT_HISTORY: 'chat-history',
  MERCY_LOGS: 'mercy-logs',
  EVOLUTION_STATES: 'evolution-states',
  USER_PREFERENCES: 'user-preferences'
};

class RathorIndexedDB {
  constructor() {
    this.db = null;
  }

  async open() {
    if (this.db) return this.db;

    return new Promise((resolve, reject) => {
      const request = indexedDB.open(DB_NAME, DB_VERSION);

      request.onerror = (event) => {
        console.error('[Rathor IndexedDB] Database open failed:', event.target.error?.name || event.target.error);
        reject(event.target.error);
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        this.db.onerror = (e) => console.error('[Rathor DB] Global DB error:', e.target.error);
        console.log('[Rathor IndexedDB] Opened v' + DB_VERSION);
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        console.log('[Rathor IndexedDB] Schema upgrade to v' + DB_VERSION);

        const createStoreIfMissing = (name, options = {}, indexes = []) => {
          if (!db.objectStoreNames.contains(name)) {
            const store = db.createObjectStore(name, options);
            indexes.forEach(([key, unique]) => store.createIndex(key, key, { unique }));
          }
        };

        createStoreIfMissing(STORES.CHAT_HISTORY, { keyPath: 'id', autoIncrement: true }, ['timestamp', 'role']);
        createStoreIfMissing(STORES.MERCY_LOGS, { keyPath: 'id', autoIncrement: true }, ['timestamp', 'valence']);
        createStoreIfMissing(STORES.EVOLUTION_STATES, { keyPath: 'bloomId' });
        createStoreIfMissing(STORES.USER_PREFERENCES, { keyPath: 'key' });
      };
    });
  }

  // Core transaction helper — mercy-wrapped for safety
  async _transaction(storeNames, mode = 'readonly', callback) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeNames, mode);
      let result;

      tx.oncomplete = () => resolve(result);
      tx.onerror = (e) => {
        console.error('[Rathor TX] Transaction error:', e.target.error?.name || e.target.error);
        reject(e.target.error);
      };
      tx.onabort = () => {
        console.warn('[Rathor TX] Transaction aborted');
        reject(new Error('Transaction aborted'));
      };

      try {
        result = callback(tx);
      } catch (err) {
        tx.abort();
        reject(err);
      }
    });
  }

  async add(storeName, data) {
    return this._transaction(storeName, 'readwrite', (tx) => {
      const store = tx.objectStore(storeName);
      const enhanced = { ...data, timestamp: Date.now() };
      const req = store.add(enhanced);
      return new Promise((res, rej) => {
        req.onsuccess = () => res(req.result);
        req.onerror = () => rej(req.error);
      });
    });
  }

  async put(storeName, data) {
    return this._transaction(storeName, 'readwrite', (tx) => {
      const store = tx.objectStore(storeName);
      const req = store.put(data);
      return new Promise((res, rej) => {
        req.onsuccess = () => res(req.result);
        req.onerror = () => rej(req.error);
      });
    });
  }

  async get(storeName, key) {
    return this._transaction(storeName, 'readonly', (tx) => {
      const store = tx.objectStore(storeName);
      const req = store.get(key);
      return new Promise((res, rej) => {
        req.onsuccess = () => res(req.result);
        req.onerror = () => rej(req.error);
      });
    });
  }

  async getAll(storeName, indexName = null, range = null, direction = 'next') {
    return this._transaction(storeName, 'readonly', (tx) => {
      const store = tx.objectStore(storeName);
      let req;
      if (indexName) {
        const index = store.index(indexName);
        req = index.openCursor(range, direction);
      } else {
        req = store.openCursor(null, direction);
      }

      const results = [];
      return new Promise((resolve, reject) => {
        req.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor) {
            results.push(cursor.value);
            cursor.continue();
          } else {
            resolve(results);
          }
        };
        req.onerror = () => reject(req.error);
      });
    });
  }

  async getLatestChat(limit = 50) {
    const all = await this.getAll(STORES.CHAT_HISTORY, 'timestamp', null, 'prev');
    return all.slice(0, limit).reverse(); // newest first
  }

  async batchAdd(storeName, items) {
    return this._transaction(storeName, 'readwrite', (tx) => {
      const store = tx.objectStore(storeName);
      items.forEach(item => {
        store.add({ ...item, timestamp: Date.now() });
      });
      return Promise.resolve(true); // success on tx complete
    });
  }

  async clear(storeName) {
    return this._transaction(storeName, 'readwrite', (tx) => {
      const store = tx.objectStore(storeName);
      store.clear();
      return Promise.resolve();
    });
  }

  // Mercy valence gate + quota fallback hint
  async addWithValence(storeName, data) {
    if (storeName === STORES.MERCY_LOGS && (data.valence ?? 0) < 0.999) {
      throw new Error('Mercy valence threshold violation — write blocked');
    }
    try {
      return await this.add(storeName, data);
    } catch (err) {
      if (err.name === 'QuotaExceededError') {
        console.warn('[Rathor DB] Quota exceeded — consider cleanup or localStorage fallback');
      }
      throw err;
    }
  }
}

export const rathorDB = new RathorIndexedDB();
