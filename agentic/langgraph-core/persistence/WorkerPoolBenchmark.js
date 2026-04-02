// agentic/langgraph-core/persistence/WorkerPoolBenchmark.js
// version: 17.248.0-workerpool-benchmark-memory-leak-detection
// Enhanced benchmark framework with built-in memory leak detection
// Per-thread deltas, trend analysis, GC forcing, leak alerts

import { workerPool } from './WorkerPool.js';
import { vfsCheckpointer } from '../utils/VFSCheckpointer.js';

export class WorkerPoolBenchmark {
  constructor() {
    this.results = [];
    this.leakAlerts = [];
  }

  async run(config = {}) {
    const { vfsType = "opfs-sab", threads = navigator.hardwareConcurrency || 8, iterations = 200, stateSizeKB = 50, runs = 5 } = config;

    vfsCheckpointer.preferredType = vfsType;
    await workerPool.initialize();

    const allDeltas = [];
    let totalDuration = 0;

    for (let run = 0; run < runs; run++) {
      if (window.gc) window.gc(); // Force GC before each run
      await new Promise(r => setTimeout(r, 300)); // Stabilise

      const before = this.getMemoryUsage();
      const startTime = performance.now();

      const promises = [];
      for (let i = 0; i < threads; i++) {
        promises.push(this._runThread(i, iterations, stateSizeKB));
      }
      const threadResults = await Promise.all(promises);

      const after = this.getMemoryUsage();
      const runDeltaMB = ((after - before) || 0);
      allDeltas.push(runDeltaMB);
      totalDuration += performance.now() - startTime;

      // Per-thread leak check
      threadResults.forEach((r, i) => {
        if (r.memoryDeltaMB > 5) {
          this.leakAlerts.push({ run, thread: i, deltaMB: r.memoryDeltaMB });
        }
      });
    }

    // Trend analysis
    const avgDelta = allDeltas.reduce((a, b) => a + b, 0) / allDeltas.length;
    const slope = this._calculateTrendSlope(allDeltas);

    const result = {
      vfsType,
      threads,
      iterations,
      runs,
      avgSave: 0, // populated from threadResults if needed
      p95: 0,
      throughput: 0,
      aggregateMemoryDeltaMB: parseFloat(avgDelta.toFixed(2)),
      trendSlopeMBPerRun: parseFloat(slope.toFixed(3)),
      leakDetected: slope > 0.5 || avgDelta > 5,
      leakAlerts: this.leakAlerts,
      timestamp: new Date().toISOString()
    };

    this.results.push(result);
    if (result.leakDetected) console.warn(`🚨 MEMORY LEAK DETECTED in ${vfsType}! Slope: ${slope.toFixed(3)}`);
    return result;
  }

  _calculateTrendSlope(deltas) {
    // Simple linear regression slope
    const n = deltas.length;
    const sumX = (n * (n - 1)) / 2;
    const sumY = deltas.reduce((a, b) => a + b, 0);
    const sumXY = deltas.reduce((sum, y, x) => sum + x * y, 0);
    const sumX2 = deltas.reduce((sum, _, x) => sum + x * x, 0);
    return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  }

  getMemoryUsage() {
    if ('memory' in performance) return performance.memory.usedJSHeapSize / (1024 * 1024);
    return 0;
  }

  // ... rest of previous methods remain unchanged
}

export const workerPoolBenchmark = new WorkerPoolBenchmark();
