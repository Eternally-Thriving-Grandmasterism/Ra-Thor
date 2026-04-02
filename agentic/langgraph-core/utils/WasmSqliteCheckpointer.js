// agentic/langgraph-core/utils/WasmSqliteCheckpointer.js
// version: 17.231.0-advanced-sqljs-features
// Sovereign WASM SQLite checkpointer with prepared statements, transactions,
// backup/restore, PRAGMA optimizations, and Mercy Gate enforcement.
// Fully enshrines previous version while adding all advanced sql.js capabilities.

import { enforceMercyGates, calculateLumenasCI } from '../../core/mercy-gates.js';

export class WasmSqliteCheckpointer {
  constructor() {
    this.db = null;
    this.threadId = 'rathor-main-thread';
    this.stmtSave = null;   // prepared statement for save
    this.stmtLoad = null;   // prepared statement for load
  }

  async initialize() {
    if (!this.db) {
      const SQL = await import('https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/sql-wasm.min.js');
      const config = { 
        locateFile: file => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/${file}`,
        useBigInt: true 
      };
      this.SQL = await SQL.default(config);
      this.db = new this.SQL.Database();

      // Advanced PRAGMA optimizations for performance + durability
      this.db.run('PRAGMA journal_mode=WAL;');
      this.db.run('PRAGMA synchronous=NORMAL;');
      this.db.run('PRAGMA cache_size=-20000;'); // \~80 MB cache

      // Create table
      this.db.run(`
        CREATE TABLE IF NOT EXISTS checkpointer (
          thread_id TEXT PRIMARY KEY,
          checkpoint BLOB,
          timestamp INTEGER
        )
      `);

      // Prepare statements once (performance win)
      this.stmtSave = this.db.prepare(`
        INSERT OR REPLACE INTO checkpointer (thread_id, checkpoint, timestamp)
        VALUES (?, ?, ?)
      `);
      this.stmtLoad = this.db.prepare('SELECT checkpoint FROM checkpointer WHERE thread_id = ?');
    }
    return this;
  }

  async save(state, threadId = this.threadId) {
    const lumenas = calculateLumenasCI(state);
    if (lumenas < 0.999) {
      console.warn('Mercy Gate blocked WASM SQLite save');
      return false;
    }

    await this.initialize();

    // Transaction for atomicity
    this.db.run('BEGIN TRANSACTION;');
    try {
      const blob = new Uint8Array(JSON.stringify(state));
      this.stmtSave.run([threadId, blob, Date.now()]);
      this.db.run('COMMIT;');

      await this._persistToIndexedDB(threadId, blob); // backup layer
      return true;
    } catch (e) {
      this.db.run('ROLLBACK;');
      console.error('WASM SQLite save failed:', e);
      return false;
    }
  }

  async load(threadId = this.threadId) {
    await this.initialize();
    const result = this.stmtLoad.get([threadId]);
    if (result && result.length) {
      const blob = result[0];
      return JSON.parse(new TextDecoder().decode(blob));
    }
    return null;
  }

  // Advanced backup/restore (full DB snapshot)
  async backup() {
    await this.initialize();
    return this.db.export(); // Uint8Array of entire DB
  }

  async restore(buffer) {
    await this.initialize();
    this.db = new this.SQL.Database(buffer);
    // Re-prepare statements after restore
    this.stmtSave = this.db.prepare('INSERT OR REPLACE INTO checkpointer (thread_id, checkpoint, timestamp) VALUES (?, ?, ?)');
    this.stmtLoad = this.db.prepare('SELECT checkpoint FROM checkpointer WHERE thread_id = ?');
  }

  async _persistToIndexedDB(threadId, blob) {
    const dbName = 'rathor-wasm-backup';
    const request = indexedDB.open(dbName, 1);
    return new Promise((resolve) => {
      request.onupgradeneeded = (e) => e.target.result.createObjectStore('backups');
      request.onsuccess = (e) => {
        const tx = e.target.result.transaction('backups', 'readwrite');
        tx.objectStore('backups').put(blob, threadId);
        resolve();
      };
    });
  }

  // Cleanup (prevents memory leaks)
  close() {
    if (this.stmtSave) this.stmtSave.free();
    if (this.stmtLoad) this.stmtLoad.free();
    if (this.db) this.db.close();
  }
}

// Singleton
export const wasmSqliteCheckpointer = new WasmSqliteCheckpointer();
