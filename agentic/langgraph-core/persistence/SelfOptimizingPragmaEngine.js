// agentic/langgraph-core/persistence/SelfOptimizingPragmaEngine.js
// version: 17.251.0-self-optimizing-pragma-engine
// Intelligent, adaptive PRAGMA tuner for all VFS checkpointers
// Monitors workload and dynamically adjusts settings in real time

import { workerPoolBenchmark } from './WorkerPoolBenchmark.js';

export class SelfOptimizingPragmaEngine {
  constructor(db) {
    this.db = db;
    this.lastOptimization = Date.now();
    this.optimizationInterval = 5000; // every 5 seconds
    this.metricsHistory = [];
  }

  async optimize(currentMetrics) {
    if (Date.now() - this.lastOptimization < this.optimizationInterval) return;
    this.lastOptimization = Date.now();

    this.metricsHistory.push(currentMetrics);
    if (this.metricsHistory.length > 10) this.metricsHistory.shift();

    const avgThroughput = this.metricsHistory.reduce((a, m) => a + (m.throughput || 0), 0) / this.metricsHistory.length;
    const avgMemoryMB = this.metricsHistory.reduce((a, m) => a + (m.aggregateMemoryDeltaMB || 0), 0) / this.metricsHistory.length;

    // Adaptive PRAGMA logic
    if (avgThroughput > 800) {
      // High throughput → more aggressive cache and WAL
      await this.db.run('PRAGMA cache_size=-196608;');
      await this.db.run('PRAGMA wal_autocheckpoint=200;');
    } else if (avgThroughput < 300) {
      // Low throughput → conserve memory
      await this.db.run('PRAGMA cache_size=-64000;');
      await this.db.run('PRAGMA wal_autocheckpoint=1000;');
    }

    if (avgMemoryMB > 600) {
      // High memory pressure → reduce cache and mmap
      await this.db.run('PRAGMA cache_size=-64000;');
      await this.db.run('PRAGMA mmap_size=134217728;'); // 128 MB
    }

    // Always run optimize after adjustment
    await this.db.run('PRAGMA optimize;');

    console.log(`🔧 Self-optimizing PRAGMA engine adjusted for throughput=\( {avgThroughput.toFixed(0)} ops/sec, memory= \){avgMemoryMB.toFixed(0)} MB`);
  }

  // Hook for benchmark framework to call after each run
  async onBenchmarkComplete(result) {
    await this.optimize(result);
  }
}

// Singleton
export const selfOptimizingPragmaEngine = new SelfOptimizingPragmaEngine(null);
