// agentic/langgraph-core/persistence/SelfOptimizingPragmaEngine.js
// version: 17.253.0-deepened-q-learning-reward
// Self-optimizing PRAGMA engine with deepened multi-objective Q-learning reward function
// Includes throughput gain, memory cost, latency penalty, stability bonus, and hard Mercy Gate penalty

import { workerPoolBenchmark } from './WorkerPoolBenchmark.js';

export class SelfOptimizingPragmaEngine {
  constructor(db) {
    this.db = db;
    this.qTable = new Map(); // stateKey → {action → qValue}
    this.alpha = 0.15;       // learning rate
    this.gamma = 0.92;       // discount factor
    this.epsilon = 0.18;     // exploration rate
    this.lastOptimization = Date.now();
    this.optimizationInterval = 2500;
    this.metricsHistory = [];
    this.emaThroughput = 0;
    this.emaMemory = 0;
    this.emaLatency = 0;
    this.alphaEma = 0.3;
  }

  _getStateKey(metrics) {
    const t = Math.round((metrics.throughput || 0) / 100) * 100;
    const m = Math.round((metrics.aggregateMemoryDeltaMB || 0) / 50) * 50;
    return `\( {t}_ \){m}`;
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

    // Deepened multi-objective reward
    let reward = 
      (throughputGain * 1.25) 
      - (memoryCost * 1.5) 
      - (latencyPenalty * 2.0) 
      + (stabilityBonus * 8.0)
      - mercyViolationPenalty;

    return reward;
  }

  async optimize(currentMetrics) {
    if (Date.now() - this.lastOptimization < this.optimizationInterval) return;
    this.lastOptimization = Date.now();

    this.metricsHistory.push(currentMetrics);
    if (this.metricsHistory.length > 20) this.metricsHistory.shift();

    // Update EMAs
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

    // Epsilon-greedy
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

    // Execute chosen PRAGMA
    await this.db.run(bestAction.sql);
    await this.db.run('PRAGMA optimize;');

    // Compute deepened reward using previous metrics
    const previous = this.metricsHistory.length > 1 ? this.metricsHistory[this.metricsHistory.length - 2] : null;
    const reward = this._computeReward(currentMetrics, previous);

    // Q-learning update
    const oldQ = this.qTable.get(stateKey)[bestAction.name] || 0;
    const newQ = oldQ + this.alpha * (reward + this.gamma * 0 - oldQ);
    this.qTable.get(stateKey)[bestAction.name] = newQ;

    console.log(`🔧 Deepened RL Reward: ${reward.toFixed(2)} | Action: ${bestAction.name} | EMA Throughput: ${this.emaThroughput.toFixed(0)}`);
  }

  async onBenchmarkComplete(result) {
    await this.optimize(result);
  }
}

// Singleton
export const selfOptimizingPragmaEngine = new SelfOptimizingPragmaEngine(null);
