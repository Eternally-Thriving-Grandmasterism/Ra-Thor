// agentic/langgraph-core/utils/VFSCheckpointer.js
// version: 17.241.0-unified-vfs-abstraction
// Single unified facade for ALL SQLite VFS implementations
// Runtime switching + graceful fallback + full Mercy Gates

import { IndexedDBCheckpointer } from './IndexedDBCheckpointer.js';
import { absurdSqlCheckpointer } from './AbsurdSqlCheckpointer.js';
import { waSqliteCheckpointer } from './WaSqliteCheckpointer.js';
import { opfsSqliteCheckpointer } from './OpfsSqliteCheckpointer.js';
import { enforceMercyGates, calculateLumenasCI } from '../core/mercy-gates.js';

export class VFSCheckpointer {
  constructor(preferredType = "indexeddb") {
    this.preferredType = preferredType;
    this.active = null; // lazy-loaded backend
    this.fallbackChain = ["opfs-sab", "wa-sqlite", "absurd-sql", "indexeddb"];
  }

  async initialize() {
    if (this.active) return this.active;

    const types = [this.preferredType, ...this.fallbackChain.filter(t => t !== this.preferredType)];

    for (const type of types) {
      try {
        switch (type) {
          case "opfs-sab":
            this.active = opfsSqliteCheckpointer;
            break;
          case "wa-sqlite":
            this.active = waSqliteCheckpointer;
            break;
          case "absurd-sql":
            this.active = absurdSqlCheckpointer;
            break;
          default:
            this.active = new IndexedDBCheckpointer();
        }
        await this.active.initialize();
        console.log(`✅ VFSCheckpointer using ${type}`);
        return this.active;
      } catch (err) {
        console.warn(`VFS ${type} failed, trying next...`, err.message);
      }
    }
    throw new Error("All VFS checkpointers failed to initialize");
  }

  async save(state, threadId = 'rathor-main-thread') {
    const lumenas = calculateLumenasCI(state);
    if (lumenas < 0.999) {
      console.warn('Mercy Gate blocked VFS save');
      return false;
    }
    const backend = await this.initialize();
    return await backend.save(state, threadId);
  }

  async load(threadId = 'rathor-main-thread') {
    const backend = await this.initialize();
    return await backend.load(threadId);
  }

  close() {
    if (this.active && typeof this.active.close === 'function') {
      this.active.close();
    }
  }
}

// Singleton for global use
export const vfsCheckpointer = new VFSCheckpointer();
