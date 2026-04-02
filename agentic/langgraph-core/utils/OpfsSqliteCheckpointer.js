// agentic/langgraph-core/utils/OpfsSqliteCheckpointer.js
// version: 17.232.0-opfs-web-worker-integrated
// Now uses dedicated Web Worker for true off-main-thread OPFS + SQLite
// Graceful fallback to IndexedDB if Worker fails
// Fully enshrines previous version

import { enforceMercyGates, calculateLumenasCI } from '../../core/mercy-gates.js';
import { IndexedDBCheckpointer } from './IndexedDBCheckpointer.js';

export class OpfsSqliteCheckpointer {
  constructor() {
    this.worker = null;
    this.fallback = new IndexedDBCheckpointer();
    this.threadId = 'rathor-main-thread';
  }

  async initialize() {
    if (!this.worker) {
      try {
        this.worker = new Worker(new URL('./opfs-worker.js', import.meta.url), { type: 'module' });
        
        // Wait for worker ready
        await new Promise((resolve, reject) => {
          this.worker.onmessage = (e) => {
            if (e.data.success && e.data.action === 'initialized') resolve();
            else if (!e.data.success) reject(new Error(e.data.error || 'Worker init failed'));
          };
          this.worker.postMessage({ action: 'initialize' });
        });
      } catch (err) {
        console.warn('OPFS Web Worker failed — falling back to IndexedDB:', err.message);
        this.worker = null;
        return this.fallback;
      }
    }
    return this;
  }

  async save(state, threadId = this.threadId) {
    const lumenas = calculateLumenasCI(state);
    if (lumenas < 0.999) {
      console.warn('Mercy Gate blocked OPFS save');
      return false;
    }

    const checkpointer = await this.initialize();
    if (checkpointer === this.fallback) {
      return await this.fallback.save(state, threadId);
    }

    return new Promise((resolve) => {
      this.worker.onmessage = (e) => resolve(e.data.success);
      this.worker.postMessage({ action: 'save', state });
    });
  }

  async load(threadId = this.threadId) {
    const checkpointer = await this.initialize();
    if (checkpointer === this.fallback) {
      return await this.fallback.load(threadId);
    }

    return new Promise((resolve) => {
      this.worker.onmessage = (e) => resolve(e.data.success ? e.data.data : null);
      this.worker.postMessage({ action: 'load' });
    });
  }

  close() {
    if (this.worker) this.worker.terminate();
    if (this.fallback) this.fallback.close?.();
  }
}

// Singleton
export const opfsSqliteCheckpointer = new OpfsSqliteCheckpointer();
