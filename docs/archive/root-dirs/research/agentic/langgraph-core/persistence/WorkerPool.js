// agentic/langgraph-core/persistence/WorkerPool.js
// version: 17.246.0-worker-pool-memory-optimized
// Production Worker Pool with aggressive memory optimization
// Dynamic worker scaling, reusable SAB, pressure monitoring, aggressive cleanup

import { opfsSqliteCheckpointer } from '../utils/OpfsSqliteCheckpointer.js';
import { enforceMercyGates, calculateLumenasCI } from '../../core/mercy-gates.js';

export class WorkerPool {
  constructor(maxWorkers = null) {
    // Dynamic limit based on device memory (in GB)
    const deviceMemoryGB = navigator.deviceMemory || 4;
    this.maxWorkers = maxWorkers ?? Math.max(2, Math.min(Math.floor(deviceMemoryGB * 2), 12));
    this.workers = [];
    this.taskQueue = [];
    this.currentWorkerIndex = 0;
    this.initialized = false;
    this.retryCount = new Map();
    this.isShuttingDown = false;
    this.sharedBuffer = null; // Reusable SAB
    this.lastMemoryCheck = Date.now();
  }

  async initialize() {
    if (this.initialized) return;

    // Create reusable SharedArrayBuffer once
    if (!this.sharedBuffer) {
      this.sharedBuffer = new SharedArrayBuffer(8 * 1024 * 1024); // 8 MB reusable buffer
    }

    for (let i = 0; i < this.maxWorkers; i++) {
      try {
        const worker = new Worker(new URL('../utils/opfs-sab-worker.js', import.meta.url), { type: 'module' });
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error('Worker init timeout')), 3000);
          worker.onmessage = (e) => {
            clearTimeout(timeout);
            if (e.data.success && e.data.action === 'initialized') resolve();
            else reject(new Error(e.data.error || 'Worker init failed'));
          };
          worker.postMessage({ action: 'initialize' });
        });
        worker.onerror = (err) => this.handleWorkerCrash(worker, i, err);
        this.workers.push(worker);
      } catch (err) {
        console.warn(`Worker ${i} failed to start`, err);
      }
    }

    this.initialized = true;
    console.log(`✅ WorkerPool initialized with ${this.workers.length} workers (deviceMemory: ${navigator.deviceMemory || 'unknown'} GB)`);
  }

  async checkMemoryPressure() {
    if (Date.now() - this.lastMemoryCheck < 5000) return; // check every 5s
    this.lastMemoryCheck = Date.now();

    if ('memory' in performance) {
      const used = performance.memory.usedJSHeapSize / (1024 * 1024);
      if (used > 600) { // high pressure
        console.warn(`Memory pressure detected (${used.toFixed(0)} MB) — reducing workers`);
        if (this.workers.length > 2) {
          const deadWorker = this.workers.pop();
          deadWorker.terminate();
        }
      }
    }
  }

  handleWorkerCrash(worker, index, err) {
    console.error(`Worker ${index} crashed:`, err);
    if (this.isShuttingDown) return;

    this.workers = this.workers.filter(w => w !== worker);
    worker.terminate();

    // Restart only if we still have capacity
    if (this.workers.length < this.maxWorkers) {
      setTimeout(() => this.initialize(), 500);
    }
  }

  async executeTask(task) {
    await this.initialize();
    await this.checkMemoryPressure();

    if (this.workers.length === 0) {
      console.warn('All workers failed — falling back to single-threaded OPFS');
      return await opfsSqliteCheckpointer[task.method](...task.args);
    }

    const worker = this.workers[this.currentWorkerIndex % this.workers.length];
    this.currentWorkerIndex = (this.currentWorkerIndex + 1) % this.workers.length;

    return new Promise((resolve) => {
      const timeout = setTimeout(() => resolve({ success: false, error: 'Worker timeout' }), 8000);

      worker.onmessage = (e) => {
        clearTimeout(timeout);
        if (e.data.success) {
          resolve(e.data);
        } else {
          const retries = (this.retryCount.get(task.id) || 0) + 1;
          this.retryCount.set(task.id, retries);
          if (retries <= 3) {
            setTimeout(() => resolve(this.executeTask(task)), 150 * Math.pow(2, retries));
          } else {
            resolve({ success: false, error: e.data.error || 'Max retries exceeded' });
          }
        }
      };

      worker.postMessage(task);
    });
  }

  async save(state, threadId) {
    const lumenas = calculateLumenasCI(state);
    if (lumenas < 0.999) return false;
    return this.executeTask({ method: 'save', args: [state, threadId], id: `save-${Date.now()}` });
  }

  async load(threadId) {
    return this.executeTask({ method: 'load', args: [threadId], id: `load-${Date.now()}` });
  }

  terminate() {
    this.isShuttingDown = true;
    this.workers.forEach(w => w.terminate());
    this.workers = [];
    this.taskQueue = [];
    this.initialized = false;
    this.retryCount.clear();
    if (this.sharedBuffer) this.sharedBuffer = null; // release SAB
  }
}

// Singleton for global use
export const workerPool = new WorkerPool();
