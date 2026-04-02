// agentic/langgraph-core/persistence/SelfOptimizingPragmaEngine.js
// version: 17.253.0-reinforcement-learning-tuning
// Self-optimizing PRAGMA engine upgraded with lightweight Q-learning
// Learns optimal PRAGMA actions from live workload feedback

import { workerPoolBenchmark } from './WorkerPoolBenchmark.js';

export class SelfOptimizingPragmaEngine {
  constructor(db) {
    this.db = db;
    this.qTable = new Map(); // stateKey → {action → qValue}
    this.alpha = 0.15;       // learning rate
    this.gamma = 0.95;       // discount factor
    this.epsilon = 0.2;      // exploration rate
    this.lastOptimization = Date.now();
    this.optimizationInterval = 2500;
    this.metricsHistory = [];
    this.emaThroughput = 0;
    this.emaMemory = 0;
    this.alphaEma = 0.3;
  }

  _getStateKey(metrics) {
    const t = Math.round(metrics.throughput || 0 / 100) * 100;
    const m = Math.round(metrics.aggregateMemoryDeltaMB || 0 / 50) * 50;
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

  async optimize(currentMetrics) {
    if (Date.now() - this.lastOptimization < this.optimizationInterval) return;
    this.lastOptimization = Date.now();

    this.metricsHistory.push(currentMetrics);
    if (this.metricsHistory.length > 15) this.metricsHistory.shift();

    const throughput = currentMetrics.throughput || 0;
    const memoryMB = currentMetrics.aggregateMemoryDeltaMB || 0;
    this.emaThroughput = this.alphaEma * throughput + (1 - this.alphaEma) * this.emaThroughput;
    this.emaMemory = this.alphaEma * memoryMB + (1 - this.alphaEma) * this.emaMemory;

    const stateKey = this._getStateKey(currentMetrics);
    if (!this.qTable.has(stateKey)) this.qTable.set(stateKey, {});

    const actions = this._getPossibleActions();
    let bestAction = actions[0];

    // Epsilon-greedy action selection
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

    // Simulate reward (throughput gain minus memory penalty)
    const reward = (this.emaThroughput * 0.8) - (this.emaMemory * 1.2);

    // Q-learning update
    const oldQ = this.qTable.get(stateKey)[bestAction.name] || 0;
    const newQ = oldQ + this.alpha * (reward + this.gamma * 0 - oldQ); // simplified next-maxQ = 0 for online
    this.qTable.get(stateKey)[bestAction.name] = newQ;

    console.log(`🔧 RL-Tuned PRAGMA: ${bestAction.name} | Reward: ${reward.toFixed(2)} | EMA Throughput: ${this.emaThroughput.toFixed(0)}`);
  }

  async onBenchmarkComplete(result) {
    await this.optimize(result);
  }
}

// Singleton
export const selfOptimizingPragmaEngine = new SelfOptimizingPragmaEngine(null);
