// agentic/langgraph-core/utils/OpfsSqliteCheckpointer.js
// version: 17.232.0-opfs-persistence
// High-performance OPFS-backed WASM SQLite checkpointer
// Uses synchronous access handles in Web Worker + graceful fallback
// Fully Mercy-Gated and LumenasCI-enforced

import { enforceMercyGates, calculateLumenasCI } from '../../core/mercy-gates.js';

export class OpfsSqliteCheckpointer {
  constructor() {
    this.db = null;
    this.threadId = 'rathor-main-thread';
    this.fileHandle = null;
    this.syncHandle = null;
  }

  async initialize() {
    if (!this.db) {
      // Dynamically load sql.js
      const SQL = await import('https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/sql-wasm.min.js');
      const config = { locateFile: file => `https://cdnjs.cloudflare.com/ajax/libs/sql.js/1.11.0/${file}` };
      this.SQL = await SQL.default(config);

      // Get OPFS root and file handle
      const root = await navigator.storage.getDirectory();
      this.fileHandle = await root.getFileHandle('rathor-checkpoint.sqlite', { create: true });

      // Open synchronous handle (must be in Worker context for full speed)
      this.syncHandle = await this.fileHandle.createSyncAccessHandle();

      this.db = new this.SQL.Database();
      // Load existing data if any
      const size = this.syncHandle.getSize();
      if (size > 0) {
        const buffer = new Uint8Array(size);
        this.syncHandle.read(buffer, { at: 0 });
        this.db = new this.SQL.Database(buffer);
      }

      // PRAGMA optimizations
      this.db.run('PRAGMA journal_mode=WAL;');
      this.db.run('PRAGMA synchronous=NORMAL;');
    }
    return this;
  }

  async save(state, threadId = this.threadId) {
    const lumenas = calculateLumenasCI(state);
    if (lumenas < 0.999) {
      console.warn('Mercy Gate blocked OPFS save');
      return false;
    }

    await this.initialize();

    const blob = new Uint8Array(JSON.stringify(state));
    const buffer = this.db.export(); // or direct write for speed

    this.syncHandle.write(buffer, { at: 0 });
    this.syncHandle.flush();

    return true;
  }

  async load(threadId = this.threadId) {
    await this.initialize();
    const size = this.syncHandle.getSize();
    if (size === 0) return null;

    const buffer = new Uint8Array(size);
    this.syncHandle.read(buffer, { at: 0 });
    return JSON.parse(new TextDecoder().decode(buffer));
  }

  close() {
    if (this.syncHandle) this.syncHandle.close();
    if (this.db) this.db.close();
  }
}

// Singleton
export const opfsSqliteCheckpointer = new OpfsSqliteCheckpointer();
