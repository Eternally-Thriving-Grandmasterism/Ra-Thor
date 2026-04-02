// agentic/langgraph-core/persistence/WorkerPool.js
// version: 17.243.0-worker-pool
// Production Worker Pool for concurrent VFS checkpointing
// Manages multiple OPFS/SAB workers with load balancing, task queue, and SAB coordination
// Integrates directly with VFSCheckpointer and LangGraphPersistenceLayer

import { opfsSqliteCheckpointer } from '../utils/OpfsSqliteCheckpointer.js';

export class WorkerPool {
  constructor(maxWorkers = navigator.hardwareConcurrency || 8) {
    this.maxWorkers = Math.max(2, Math.min(maxWorkers, 16)); // safe limits
    this.workers = [];
    this.taskQueue = [];
    this.currentWorkerIndex = 0;
    this.initialized = false;
  }

  async initialize() {
    if (this.initialized) return;

    for (let i = 0; i < this.maxWorkers; i++) {
      try {
        const worker = new Worker(new URL('../utils/opfs-sab-worker.js', import.meta.url), { type: 'module' });
        await new Promise((resolve, reject) => {
          worker.onmessage = (e) => {
            if (e.data.success && e.data.action === 'initialized') resolve();
            else reject(new Error(e.data.error || 'Worker init failed'));
          };
          worker.postMessage({ action: 'initialize' });
        });
        this.workers.push(worker);
      } catch (err) {
        console.warn(`Worker ${i} failed to start`, err);
      }
    }

    this.initialized = true;
    console.log(`✅ WorkerPool initialized with ${this.workers.length} workers`);
  }

  async executeTask(task) {
    await this.initialize();

    if (this.workers.length === 0) {
      // Single-threaded fallback
      return await opfsSqliteCheckpointer[task.method](...task.args);
    }

    // Round-robin load balancing
    const worker = this.workers[this.currentWorkerIndex % this.workers.length];
    this.currentWorkerIndex++;

    return new Promise((resolve) => {
      worker.onmessage = (e) => {
        if (e.data.success) resolve(e.data);
        else resolve({ success: false, error: e.data.error });
      };
      worker.postMessage(task);
    });
  }

  async save(state, threadId) {
    return this.executeTask({ method: 'save', args: [state, threadId] });
  }

  async load(threadId) {
    return this.executeTask({ method: 'load', args: [threadId] });
  }

  terminate() {
    this.workers.forEach(w => w.terminate());
    this.workers = [];
    this.initialized = false;
  }
}

// Singleton for global use
export const workerPool = new WorkerPool();
