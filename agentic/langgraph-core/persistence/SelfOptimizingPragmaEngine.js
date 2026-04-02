// agentic/langgraph-core/persistence/SelfOptimizingPragmaEngine.js
// version: 17.252.0-ml-heuristics-expanded
// Self-optimizing PRAGMA engine with expanded ML-style heuristics
// Exponential smoothing, linear regression, adaptive thresholds, memory forecasting

import { workerPoolBenchmark } from './WorkerPoolBenchmark.js';

export class SelfOptimizingPragmaEngine {
  constructor(db) {
    this.db = db;
    this.lastOptimization = Date.now();
    this.optimizationInterval = 3000; // every 3 seconds
    this.metricsHistory = []; // rolling window of last 20 runs
    this.alpha = 0.3; // smoothing factor for exponential moving average
    this.emaThroughput = 0;
    this.emaMemory = 0;
  }

  async optimize(currentMetrics) {
    if (Date.now() - this.lastOptimization < this.optimizationInterval) return;
    this.lastOptimization = Date.now();

    this.metricsHistory.push(currentMetrics);
    if (this.metricsHistory.length > 20) this.metricsHistory.shift();

    // Update exponential moving averages
    const throughput = currentMetrics.throughput || 0;
    const memoryMB = currentMetrics.aggregateMemoryDeltaMB || 0;
    this.emaThroughput = this.alpha * throughput + (1 - this.alpha) * this.emaThroughput;
    this.emaMemory = this.alpha * memoryMB + (1 - this.alpha) * this.emaMemory;

    // Linear regression on recent history to predict next optimal cache_size
    const predictedCache = this._predictOptimalCache();

    // Adaptive PRAGMA decisions
    if (this.emaThroughput > 900) {
      await this.db.run(`PRAGMA cache_size=-${Math.min(393216, predictedCache)};`); // up to \~1.5 GB
      await this.db.run('PRAGMA wal_autocheckpoint=150;');
    } else if (this.emaThroughput < 400) {
      await this.db.run('PRAGMA cache_size=-64000;');
      await this.db.run('PRAGMA wal_autocheckpoint=1200;');
    }

    if (this.emaMemory > 650) {
      await this.db.run('PRAGMA cache_size=-96000;');
      await this.db.run('PRAGMA mmap_size=134217728;'); // 128 MB
    } else if (this.emaMemory < 300) {
      await this.db.run('PRAGMA mmap_size=536870912;'); // 512 MB
    }

    await this.db.run('PRAGMA optimize;');

    console.log(`🔧 ML-Heuristics PRAGMA engine tuned → throughput EMA: ${this.emaThroughput.toFixed(0)} ops/sec, memory EMA: ${this.emaMemory.toFixed(0)} MB`);
  }

  _predictOptimalCache() {
    if (this.metricsHistory.length < 5) return 196608; // fallback
    const n = this.metricsHistory.length;
    let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
    this.metricsHistory.forEach((m, i) => {
      const x = i;
      const y = m.aggregateMemoryDeltaMB || 0;
      sumX += x;
      sumY += y;
      sumXY += x * y;
      sumX2 += x * x;
    });
    const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
    const predictedMemory = this.metricsHistory[this.metricsHistory.length - 1].aggregateMemoryDeltaMB + slope;
    return Math.max(64000, Math.min(393216, Math.round(predictedMemory * 512))); // map to cache size
  }

  // Hook called by benchmark framework after each run
  async onBenchmarkComplete(result) {
    await this.optimize(result);
  }
}

// Singleton
export const selfOptimizingPragmaEngine = new SelfOptimizingPragmaEngine(null);
