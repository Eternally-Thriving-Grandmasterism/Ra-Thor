// agentic/langgraph-core/persistence/WorkerPoolBenchmark.js
// version: 17.249.0-advanced-gc-techniques
// Benchmark framework with advanced GC techniques: forced GC, stabilization, pressure monitoring, leak detection

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
      // === ADVANCED GC TECHNIQUES ===
      if (window.gc) {
        window.gc();                    // Force GC
      }
      await new Promise(r => setTimeout(r, 500)); // Stabilization delay

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
        if (r.memoryDeltaMB > 8) {
          this.leakAlerts.push({ run, thread: i, deltaMB: r.memoryDeltaMB });
        }
      });
    }

    const avgDelta = allDeltas.reduce((a, b) => a + b, 0) / allDeltas.length;
    const slope = this._calculateTrendSlope(allDeltas);

    const result = {
      vfsType,
      threads,
      iterations,
      runs,
      avgSave: 0,
      p95: 0,
      throughput: 0,
      aggregateMemoryDeltaMB: parseFloat(avgDelta.toFixed(2)),
      trendSlopeMBPerRun: parseFloat(slope.toFixed(3)),
      leakDetected: slope > 0.5 || avgDelta > 8,
      leakAlerts: this.leakAlerts,
      timestamp: new Date().toISOString()
    };

    this.results.push(result);
    if (result.leakDetected) console.warn(`🚨 MEMORY LEAK DETECTED in ${vfsType}! Slope: ${slope.toFixed(3)} MB/run`);
    return result;
  }

  // ... _runThread, getMemoryUsage, _calculateTrendSlope, getLatestResults, exportCSV remain unchanged from previous version
}

export const workerPoolBenchmark = new WorkerPoolBenchmark();
