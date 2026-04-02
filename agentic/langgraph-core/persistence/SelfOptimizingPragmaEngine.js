// agentic/langgraph-core/persistence/SelfOptimizingPragmaEngine.js
// version: 17.254.0-expanded-state-representation
// Self-optimizing PRAGMA engine with richly expanded state representation
// Now includes throughput, memory, latency, thread count, stability, and Mercy Gate margin

import { workerPoolBenchmark } from './WorkerPoolBenchmark.js';

export class SelfOptimizingPragmaEngine {
  constructor(db) {
    this.db = db;
    this.qTable = new Map(); // richer stateKey → {action → qValue}
    this.alpha = 0.15;
    this.gamma = 0.92;
    this.epsilon = 0.18;
    this.lastOptimization = Date.now();
    this.optimizationInterval = 2500;
    this.metricsHistory = [];
    this.emaThroughput = 0;
    this.emaMemory = 0;
    this.emaLatency = 0;
    this.alphaEma = 0.3;
  }

  _getStateKey(metrics) {
    // Rich multi-dimensional state representation
    const t = Math.round((metrics.throughput || 0) / 100) * 100;           // throughput bucket
    const m = Math.round((metrics.aggregateMemoryDeltaMB || 0) / 50) * 50; // memory bucket
    const l = Math.round((metrics.p95 || 2.0) * 2) / 2;                    // latency bucket (0.5 steps)
    const threads = Math.min(Math.max(Math.round((metrics.threads || 8) / 4) * 4, 4), 16); // thread bucket
    const variance = metrics.p95Variance || 0;
    const stability = Math.max(0, Math.min(1, 1 / (1 + variance)));       // stability score 0-1
    const mercyMargin = Math.max(0, (metrics.lumenasCI || 0.999) - 0.999); // how safe from violation

    return `\( {t}_ \){m}_\( {l}_ \){threads}_\( {stability.toFixed(1)}_ \){mercyMargin.toFixed(3)}`;
  }

  _getPossibleActions() {
    return [
      { name: 'high_cache', sql: 'PRAGMA cache_size=-196608;' },
      { name: 'medium_cache', sql: 'PRAGMA cache_size=-128000;' },
      { name: 'low_cache', sql: 'PRAGMA cache_size=-64000;' },
      { name: 'aggressive_wal', sql: 'PRAGMA wal_autocheckpoint=150;' },
      { name: 'conservative_wal', sql: 'PRAGMA wal_autocheckpoint=800;' },
      { name: 'high_mmap', sql: 'PRAGMA mmap_size=536870912;' },
      { name: 'medium_mmap', sql: 'PRAGMA mmap_size=268435456;' }
    ];
  }

  _computeReward(current, previous) {
    const throughputGain = (current.throughput || 0) - (previous?.throughput || 0);
    const memoryCost = Math.max(0, (current.aggregateMemoryDeltaMB || 0) - 400);
    const latencyPenalty = Math.max(0, (current.p95 || 2.0) - 2.0);
    const variance = current.p95Variance || 0;
    const stabilityBonus = 1.0 / (1 + variance);
    const mercyViolationPenalty = current.lumenasCI < 0.999 ? 50 : 0;

    return (throughputGain * 1.25) 
           - (memoryCost * 1.5) 
           - (latencyPenalty * 2.0) 
           + (stabilityBonus * 8.0)
           - mercyViolationPenalty;
  }

  async optimize(currentMetrics) {
    if (Date.now() - this.lastOptimization < this.optimizationInterval) return;
    this.lastOptimization = Date.now();

    this.metricsHistory.push(currentMetrics);
    if (this.metricsHistory.length > 20) this.metricsHistory.shift();

    const throughput = currentMetrics.throughput || 0;
    const memoryMB = currentMetrics.aggregateMemoryDeltaMB || 0;
    const latency = currentMetrics.p95 || 2.0;
    this.emaThroughput = this.alphaEma * throughput + (1 - this.alphaEma) * this.emaThroughput;
    this.emaMemory = this.alphaEma * memoryMB + (1 - this.alphaEma) * this.emaMemory;
    this.emaLatency = this.alphaEma * latency + (1 - this.alphaEma) * this.emaLatency;

    const stateKey = this._getStateKey(currentMetrics);
    if (!this.qTable.has(stateKey)) this.qTable.set(stateKey, {});

    const actions = this._getPossibleActions();
    let bestAction = actions[0];

    if (Math.random() < this.epsilon) {
      bestAction = actions[Math.floor(Math.random() * actions.length)];
    } else {
      let maxQ = -Infinity;
      for (const action of actions) {
        const q = this.qTable.get(stateKey)[action.name] || 0;
        if (q > maxQ) {
          maxQ = q;
          bestAction = action;
        }
      }
    }

    await this.db.run(bestAction.sql);
    await this.db.run('PRAGMA optimize;');

    const previous = this.metricsHistory.length > 1 ? this.metricsHistory[this.metricsHistory.length - 2] : null;
    const reward = this._computeReward(currentMetrics, previous);

    const oldQ = this.qTable.get(stateKey)[bestAction.name] || 0;
    const newQ = oldQ + this.alpha * (reward + this.gamma * 0 - oldQ);
    this.qTable.get(stateKey)[bestAction.name] = newQ;

    console.log(`🔧 Deep RL with Expanded State: ${bestAction.name} | Reward: ${reward.toFixed(2)} | State: ${stateKey}`);
  }

  async onBenchmarkComplete(result) {
    await this.optimize(result);
  }
}

// Singleton
export const selfOptimizingPragmaEngine = new SelfOptimizingPragmaEngine(null);
