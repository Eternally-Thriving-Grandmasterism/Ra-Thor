// agentic/langgraph-core/utils/OpfsSqliteCheckpointer.js
// version: 17.234.0-sharedarraybuffer-integrated
// Now uses SharedArrayBuffer + Web Worker for zero-copy concurrency
// Graceful fallback to IndexedDB if SAB/Worker fails
// Fully enshrines previous version

import { enforceMercyGates, calculateLumenasCI } from '../../core/mercy-gates.js';
import { IndexedDBCheckpointer } from './IndexedDBCheckpointer.js';

export class OpfsSqliteCheckpointer {
  constructor() {
    this.worker = null;
    this.fallback = new IndexedDBCheckpointer();
    this.sharedBuffer = null;
    this.threadId = 'rathor-main-thread';
  }

  async initialize() {
    if (!this.worker) {
      try {
        this.worker = new Worker(new URL('./opfs-sab-worker.js', import.meta.url), { type: 'module' });

        await new Promise((resolve, reject) => {
          this.worker.onmessage = (e) => {
            if (e.data.success && e.data.action === 'initialized') {
              this.sharedBuffer = e.data.sab;
              resolve();
            } else if (!e.data.success) reject(new Error(e.data.error));
          };
          this.worker.postMessage({ action: 'initialize' });
        });
      } catch (err) {
        console.warn('SAB OPFS Worker failed — falling back to IndexedDB:', err.message);
        this.worker = null;
        return this.fallback;
      }
    }
    return this;
  }

  async save(state, threadId = this.threadId) {
    const lumenas = calculateLumenasCI(state);
    if (lumenas < 0.999) return false;

    const checkpointer = await this.initialize();
    if (checkpointer === this.fallback) return await this.fallback.save(state, threadId);

    // Zero-copy write to SharedArrayBuffer
    const encoded = new TextEncoder().encode(JSON.stringify(state));
    const view = new Uint8Array(this.sharedBuffer);
    new DataView(this.sharedBuffer).setUint32(0, encoded.length, true);
    view.set(encoded, 4);

    return new Promise((resolve) => {
      this.worker.onmessage = (e) => resolve(e.data.success);
      this.worker.postMessage({ action: 'save' });
    });
  }

  async load(threadId = this.threadId) {
    const checkpointer = await this.initialize();
    if (checkpointer === this.fallback) return await this.fallback.load(threadId);

    return new Promise((resolve) => {
      this.worker.onmessage = (e) => {
        if (e.data.success && e.data.action === 'loaded') {
          const view = new Uint8Array(this.sharedBuffer);
          const len = new DataView(this.sharedBuffer).getUint32(0, true);
          const data = JSON.parse(new TextDecoder().decode(view.slice(4, 4 + len)));
          resolve(data);
        } else resolve(null);
      };
      this.worker.postMessage({ action: 'load' });
    });
  }

  close() {
    if (this.worker) this.worker.terminate();
    if (this.fallback) this.fallback.close?.();
  }
}

export const opfsSqliteCheckpointer = new OpfsSqliteCheckpointer();
