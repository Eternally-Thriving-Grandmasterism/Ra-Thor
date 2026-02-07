const DB_NAME = 'RathorNEXiDB';
const DB_VERSION = 2; // Bump if adding new fields/indexes

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
        console.error('[Rathor IndexedDB] Open failed:', event.target.error);
        reject(event.target.error);
      };

      request.onsuccess = (event) => {
        this.db = event.target.result;
        console.log('[Rathor IndexedDB] Opened v' + this.db.version);
        resolve(this.db);
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        const oldVersion = event.oldVersion || 0;
        console.log(`[Rathor IndexedDB] Upgrading from v\( {oldVersion} to v \){DB_VERSION}`);

        const createOrUpdateStore = (name, options = {}, indexes = []) => {
          let store;
          if (db.objectStoreNames.contains(name)) {
            store = event.target.transaction.objectStore(name);
          } else {
            store = db.createObjectStore(name, options);
          }
          indexes.forEach(([keyPath, unique = false]) => {
            if (!store.indexNames.contains(keyPath)) {
              store.createIndex(keyPath, keyPath, { unique });
            }
          });
          return store;
        };

        if (oldVersion < 1) {
          createOrUpdateStore(STORES.CHAT_HISTORY, { keyPath: 'id', autoIncrement: true }, [
            ['timestamp', false],
            ['role', false]
          ]);
          // ... other stores as before
        }

        if (oldVersion < 2) {
          const chatStore = event.target.transaction.objectStore(STORES.CHAT_HISTORY);
          if (!chatStore.indexNames.contains('sessionId')) {
            chatStore.createIndex('sessionId', 'sessionId', { unique: false });
          }
        }
      };

      request.onblocked = () => {
        console.warn('[Rathor IndexedDB] Upgrade blocked — close other tabs');
      };
    });
  }

  async _transaction(storeNames, mode = 'readonly', callback) {
    const db = await this.open();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeNames, mode);
      tx.oncomplete = () => resolve();
      tx.onerror = (e) => reject(e.target.error);
      tx.onabort = () => reject(new Error('Transaction aborted'));
      callback(tx);
    });
  }

  // ────────────────────────────────────────────────
  // Chat Persistence Methods
  // ────────────────────────────────────────────────

  async saveMessage(message) {
    // message = { role: 'user'|'rathor', content: string, timestamp?: number, valence?: number, sessionId?: string }
    const enhanced = {
      ...message,
      timestamp: message.timestamp || Date.now(),
      sessionId: message.sessionId || 'default' // future multi-session support
    };

    return this._transaction(STORES.CHAT_HISTORY, 'readwrite', (tx) => {
      const store = tx.objectStore(STORES.CHAT_HISTORY);
      const req = store.add(enhanced);
      return new Promise((res, rej) => {
        req.onsuccess = () => res(req.result);
        req.onerror = () => rej(req.error);
      });
    });
  }

  async loadHistory(limit = 100, sessionId = 'default') {
    return this._transaction(STORES.CHAT_HISTORY, 'readonly', (tx) => {
      const store = tx.objectStore(STORES.CHAT_HISTORY);
      const index = store.index('timestamp');
      const range = IDBKeyRange.lowerBound(0);
      const req = index.openCursor(range, 'prev'); // newest first

      const messages = [];
      return new Promise((resolve, reject) => {
        let count = 0;
        req.onsuccess = (event) => {
          const cursor = event.target.result;
          if (cursor && count < limit) {
            if (cursor.value.sessionId === sessionId) {
              messages.push(cursor.value);
              count++;
            }
            cursor.continue();
          } else {
            resolve(messages.reverse()); // oldest first for rendering
          }
        };
        req.onerror = () => reject(req.error);
      });
    });
  }

  async clearChatHistory(sessionId = 'default') {
    return this._transaction(STORES.CHAT_HISTORY, 'readwrite', (tx) => {
      const store = tx.objectStore(STORES.CHAT_HISTORY);
      const index = store.index('sessionId');
      const req = index.openCursor(IDBKeyRange.only(sessionId));
      req.onsuccess = (event) => {
        const cursor = event.target.result;
        if (cursor) {
          cursor.delete();
          cursor.continue();
        }
      };
    });
  }
}

export const rathorDB = new RathorIndexedDB();          //       value.lastName = names.join(' ');
          //       delete value.name;
          //       cursor.update(value);
          //     }
          //     cursor.continue();
          //   }
          // };
        }

        // Future versions: Add more if (oldVersion < X) blocks
      };

      request.onblocked = () => {
        console.warn('[Rathor IndexedDB] Upgrade blocked — close other tabs');
        alert('Please close other tabs/windows using Rathor to complete lattice upgrade.');
      };
    });
  }

  // ... rest of class (add, put, get, getAll, batchAdd, clear, addWithValence) as in previous full version ...
  // (Omit repeating them here for brevity — keep them identical in your overwrite)

  async migrateExampleData() {
    // Optional: Manual migration trigger if needed outside upgrade
    // But prefer doing in onupgradeneeded
  }
}

export const rathorDB = new RathorIndexedDB();
