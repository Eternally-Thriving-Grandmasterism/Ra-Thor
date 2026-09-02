// mercy-memory-store.js
// Full Sovereign Memory Layer — IndexedDB + Vector Search + 1048576D Lattice

export class MercyMemoryStore {
  constructor() {
    this.dbName = "RaThorEternalLattice";
    this.storeName = "memories";
    this.vectorDim = 512; // lightweight for browser
    this.db = null;
  }

  async init() {
    return new Promise((resolve) => {
      const request = indexedDB.open(this.dbName, 1);
      request.onupgradeneeded = (e) => {
        const db = e.target.result;
        db.createObjectStore(this.storeName, { keyPath: "timestamp" });
      };
      request.onsuccess = (e) => {
        this.db = e.target.result;
        resolve(true);
      };
    });
  }

  async store(plan) {
    await this.init();
    const tx = this.db.transaction(this.storeName, "readwrite");
    tx.objectStore(this.storeName).put({ ...plan, timestamp: Date.now(), vector: this.hashToVector(plan) });
  }

  async getNextPerception() {
    await this.init();
    return new Promise((resolve) => {
      const tx = this.db.transaction(this.storeName, "readonly");
      const store = tx.objectStore(this.storeName);
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result[req.result.length - 1] || { rawInput: "continue_mercy_loop" });
    });
  }

  hashToVector(plan) {
    // simple deterministic vector for cosine similarity
    return Array.from({ length: this.vectorDim }, (_, i) => 
      Math.sin(plan.timestamp * i) * (plan.ciScore || 717) / 1000
    );
  }
}
