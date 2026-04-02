// agentic/langgraph-core/persistence/WorkerPoolBenchmark.js
// version: 17.248.0-workerpool-benchmark-per-thread-memory-tracking
// Enhanced benchmark framework with detailed per-thread memory tracking
// Tracks memory delta per thread, max/avg per-thread delta, plus aggregate

import { workerPool } from './WorkerPool.js';
import { vfsCheckpointer } from '../utils/VFSCheckpointer.js';

export class WorkerPoolBenchmark {
  constructor() {
    this.results = [];
  }

  async run(config = {}) {
    const { vfsType = "opfs-sab", threads = navigator.hardwareConcurrency || 8, iterations = 200, stateSizeKB = 50 } = config;

    vfsCheckpointer.preferredType = vfsType;
    await workerPool.initialize();

    const startTime = performance.now();
    const beforeMemory = this.getMemoryUsage();

    const promises = [];
    for (let i = 0; i < threads; i++) {
      promises.push(this._runThread(i, iterations, stateSizeKB));
    }

    const threadResults = await Promise.all(promises);

    const afterMemory = this.getMemoryUsage();
    const aggregateDeltaMB = ((afterMemory - beforeMemory) || 0).toFixed(1);

    const totalDuration = performance.now() - startTime;
    const avgSave = threadResults.reduce((sum, r) => sum + r.avgSave, 0) / threads;
    const p95 = this._calculateP95(threadResults.flatMap(r => r.times));
    const throughput = (threads * iterations * 2) / (totalDuration / 1000);

    // Per-thread memory tracking
    const perThreadDeltas = threadResults.map(r => r.memoryDeltaMB);
    const maxDeltaPerThread = Math.max(...perThreadDeltas);
    const avgDeltaPerThread = perThreadDeltas.reduce((sum, d) => sum + d, 0) / perThreadDeltas.length;

    const result = {
      vfsType,
      threads,
      iterations,
      avgSave: parseFloat(avgSave.toFixed(2)),
      p95: parseFloat(p95.toFixed(2)),
      throughput: parseFloat(throughput.toFixed(0)),
      totalDuration: parseFloat(totalDuration.toFixed(0)),
      aggregateMemoryDeltaMB: parseFloat(aggregateDeltaMB),
      perThreadDeltas: perThreadDeltas.map(d => parseFloat(d.toFixed(1))),
      maxDeltaPerThread: parseFloat(maxDeltaPerThread.toFixed(1)),
      avgDeltaPerThread: parseFloat(avgDeltaPerThread.toFixed(1)),
      timestamp: new Date().toISOString()
    };

    this.results.push(result);
    return result;
  }

  async _runThread(threadId, iterations, stateSizeKB) {
    const beforeThreadMemory = this.getMemoryUsage();
    const times = [];

    const testState = {
      userInput: "Benchmark test payload",
      language: "en",
      lumenasCI: 0.9995,
      sessionHistory: Array.from({ length: Math.floor(stateSizeKB / 2) }, (_, i) => `entry-${i}`)
    };

    for (let i = 0; i < iterations; i++) {
      const start = performance.now();
      await workerPool.save(testState, `benchmark-thread-\( {threadId}- \){i}`);
      await workerPool.load(`benchmark-thread-\( {threadId}- \){i}`);
      times.push(performance.now() - start);
    }

    const afterThreadMemory = this.getMemoryUsage();
    const memoryDeltaMB = ((afterThreadMemory - beforeThreadMemory) || 0);

    const avgSave = times.reduce((a, b) => a + b, 0) / times.length;
    return { avgSave, times, memoryDeltaMB };
  }

  getMemoryUsage() {
    if ('memory' in performance) {
      return performance.memory.usedJSHeapSize / (1024 * 1024);
    }
    return 0;
  }

  _calculateP95(numbers) {
    const sorted = [...numbers].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length * 0.95)];
  }

  getLatestResults() {
    return this.results;
  }

  exportCSV() {
    if (this.results.length === 0) return '';
    const headers = Object.keys(this.results[0]).join(',');
    const rows = this.results.map(r => Object.values(r).join(',')).join('\n');
    return `\( {headers}\n \){rows}`;
  }
}

// Singleton for global use in prototype page
export const workerPoolBenchmark = new WorkerPoolBenchmark();
