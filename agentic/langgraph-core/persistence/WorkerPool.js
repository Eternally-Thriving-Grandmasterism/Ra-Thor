// agentic/langgraph-core/persistence/WorkerPool.js
// version: 17.244.0-worker-pool-robust-error-handling
// Production Worker Pool with full error handling, crash recovery,
// exponential backoff, graceful degradation, and Mercy Gates integration

import { opfsSqliteCheckpointer } from '../utils/OpfsSqliteCheckpointer.js';
import { enforceMercyGates, calculateLumenasCI } from '../../core/mercy-gates.js';

export class WorkerPool {
  constructor(maxWorkers = navigator.hardwareConcurrency || 8) {
    this.maxWorkers = Math.max(2, Math.min(maxWorkers, 16));
    this.workers = [];
    this.taskQueue = [];
    this.currentWorkerIndex = 0;
    this.initialized = false;
    this.retryCount = new Map(); // taskId → retry count
    this.isShuttingDown = false;
  }

  async initialize() {
    if (this.initialized) return;

    for (let i = 0; i < this.maxWorkers; i++) {
      try {
        const worker = new Worker(new URL('../utils/opfs-sab-worker.js', import.meta.url), { type: 'module' });
        
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error('Worker init timeout')), 5000);
          worker.onmessage = (e) => {
            clearTimeout(timeout);
            if (e.data.success && e.data.action === 'initialized') resolve();
            else reject(new Error(e.data.error || 'Worker init failed'));
          };
          worker.postMessage({ action: 'initialize' });
        });

        // Error handler for crashed workers
        worker.onerror = (err) => this.handleWorkerCrash(worker, i, err);
        this.workers.push(worker);
      } catch (err) {
        console.warn(`Worker ${i} failed to start`, err);
      }
    }

    this.initialized = true;
    console.log(`✅ WorkerPool initialized with ${this.workers.length} workers`);
  }

  handleWorkerCrash(worker, index, err) {
    console.error(`Worker ${index} crashed:`, err);
    if (this.isShuttingDown) return;

    // Remove dead worker
    this.workers = this.workers.filter(w => w !== worker);
    worker.terminate();

    // Restart a new worker
    setTimeout(async () => {
      try {
        const newWorker = new Worker(new URL('../utils/opfs-sab-worker.js', import.meta.url), { type: 'module' });
        await new Promise((resolve) => {
          newWorker.onmessage = (e) => {
            if (e.data.success && e.data.action === 'initialized') resolve();
          };
          newWorker.postMessage({ action: 'initialize' });
        });
        newWorker.onerror = (e) => this.handleWorkerCrash(newWorker, index, e);
        this.workers.push(newWorker);
        console.log(`🔄 Restarted crashed worker at index ${index}`);
      } catch (restartErr) {
        console.error('Worker restart failed', restartErr);
      }
    }, 300);
  }

  async executeTask(task) {
    await this.initialize();
    if (this.workers.length === 0) {
      // Full fallback to single-threaded
      console.warn('All workers failed — falling back to single-threaded OPFS');
      return await opfsSqliteCheckpointer[task.method](...task.args);
    }

    const worker = this.workers[this.currentWorkerIndex % this.workers.length];
    this.currentWorkerIndex++;

    return new Promise((resolve) => {
      const timeout = setTimeout(() => {
        resolve({ success: false, error: 'Worker timeout' });
      }, 10000);

      worker.onmessage = (e) => {
        clearTimeout(timeout);
        if (e.data.success) {
          resolve(e.data);
        } else {
          // Retry logic with backoff
          const retries = (this.retryCount.get(task.id) || 0) + 1;
          this.retryCount.set(task.id, retries);
          if (retries <= 3) {
            setTimeout(() => resolve(this.executeTask(task)), 100 * Math.pow(2, retries));
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
    if (lumenas < 0.999) {
      console.warn('Mercy Gate blocked WorkerPool save');
      return false;
    }
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
  }
}

// Singleton for global use
export const workerPool = new WorkerPool();
