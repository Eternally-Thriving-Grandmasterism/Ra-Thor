// agentic/langgraph-core/utils/IndexedDBCheckpointer.js
// Sovereign IndexedDB checkpointer for full offline persistence
export class IndexedDBCheckpointer {
  constructor(dbName = "rathor-agentic") {
    this.dbName = dbName;
    this.storeName = "checkpoints";
  }

  async save(threadId, checkpoint) {
    const db = await this.openDB();
    const tx = db.transaction(this.storeName, "readwrite");
    const store = tx.objectStore(this.storeName);
    await store.put({ threadId, checkpoint, timestamp: Date.now() });
    return true;
  }

  async load(threadId) {
    const db = await this.openDB();
    const tx = db.transaction(this.storeName, "readonly");
    const store = tx.objectStore(this.storeName);
    const result = await store.get(threadId);
    return result ? result.checkpoint : null;
  }

  async openDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.dbName, 1);
      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains(this.storeName)) {
          db.createObjectStore(this.storeName, { keyPath: "threadId" });
        }
      };
      request.onsuccess = (event) => resolve(event.target.result);
      request.onerror = (event) => reject(event.target.error);
    });
  }
}
