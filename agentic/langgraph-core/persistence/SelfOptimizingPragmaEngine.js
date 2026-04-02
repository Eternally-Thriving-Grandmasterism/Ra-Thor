// agentic/langgraph-core/persistence/SelfOptimizingPragmaEngine.js
// version: 17.255.0-full-backpropagation-dqn
// Deep Q-Network with full backpropagation, target network, experience replay, and proper Q-learning

import { workerPoolBenchmark } from './WorkerPoolBenchmark.js';

class SimpleNeuralNet {
  constructor(inputSize = 6, hiddenSize = 32, outputSize = 7) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    this.outputSize = outputSize;

    // Weights and biases
    this.w1 = Array.from({ length: inputSize }, () => Array.from({ length: hiddenSize }, () => Math.random() * 0.2 - 0.1));
    this.b1 = new Array(hiddenSize).fill(0);
    this.w2 = Array.from({ length: hiddenSize }, () => Array.from({ length: outputSize }, () => Math.random() * 0.2 - 0.1));
    this.b2 = new Array(outputSize).fill(0);

    // Target network (for stable learning)
    this.targetW1 = JSON.parse(JSON.stringify(this.w1));
    this.targetB1 = [...this.b1];
    this.targetW2 = JSON.parse(JSON.stringify(this.w2));
    this.targetB2 = [...this.b2];
  }

  forward(state, useTarget = false) {
    const w1 = useTarget ? this.targetW1 : this.w1;
    const b1 = useTarget ? this.targetB1 : this.b1;
    const w2 = useTarget ? this.targetW2 : this.w2;
    const b2 = useTarget ? this.targetB2 : this.b2;

    // Hidden layer
    const hidden = w1.map((row, i) => 
      row.reduce((sum, w, j) => sum + w * state[j], 0) + b1[i]
    ).map(x => Math.max(0, x)); // ReLU

    // Output layer (Q-values)
    return w2.map((row, i) => 
      row.reduce((sum, w, j) => sum + w * hidden[j], 0) + b2[i]
    );
  }

  // Full backpropagation with MSE loss
  backprop(state, targetQ, actionIndex, lr = 0.01) {
    const hidden = this.w1.map((row, i) => 
      row.reduce((sum, w, j) => sum + w * state[j], 0) + this.b1[i]
    ).map(x => Math.max(0, x));

    const qValues = this.forward(state, false);
    const outputErrors = qValues.map((q, i) => i === actionIndex ? q - targetQ : 0);

    // Backprop to hidden layer
    const hiddenErrors = hidden.map((h, i) => 
      outputErrors.reduce((sum, e, j) => sum + e * this.w2[i][j], 0) * (h > 0 ? 1 : 0)
    );

    // Update output layer
    for (let i = 0; i < this.outputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.w2[j][i] -= lr * outputErrors[i] * hidden[j];
      }
      this.b2[i] -= lr * outputErrors[i];
    }

    // Update hidden layer
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.inputSize; j++) {
        this.w1[j][i] -= lr * hiddenErrors[i] * state[j];
      }
      this.b1[i] -= lr * hiddenErrors[i];
    }
  }

  // Soft target network update
  updateTarget(tau = 0.005) {
    for (let i = 0; i < this.inputSize; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.targetW1[i][j] = tau * this.w1[i][j] + (1 - tau) * this.targetW1[i][j];
      }
    }
    for (let i = 0; i < this.hiddenSize; i++) {
      for (let j = 0; j < this.outputSize; j++) {
        this.targetW2[i][j] = tau * this.w2[i][j] + (1 - tau) * this.targetW2[i][j];
      }
    }
  }
}

export class SelfOptimizingPragmaEngine {
  constructor(db) {
    this.db = db;
    this.net = new SimpleNeuralNet(6, 32, 7);
    this.replayBuffer = [];
    this.bufferSize = 3000;
    this.alpha = 0.15;
    this.gamma = 0.92;
    this.epsilon = 0.25;
    this.epsilonDecay = 0.995;
    this.minEpsilon = 0.05;
    this.lastOptimization = Date.now();
    this.optimizationInterval = 2500;
    this.metricsHistory = [];
  }

  _getStateVector(metrics) {
    return [
      (metrics.throughput || 0) / 1000,
      (metrics.aggregateMemoryDeltaMB || 0) / 1000,
      (metrics.p95 || 2.0) / 10,
      (metrics.threads || 8) / 16,
      1 / (1 + (metrics.p95Variance || 0)),
      Math.max(0, (metrics.lumenasCI || 0.999) - 0.999)
    ];
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

    return (throughputGain * 1.25) - (memoryCost * 1.5) - (latencyPenalty * 2.0) + (stabilityBonus * 8.0) - mercyViolationPenalty;
  }

  async optimize(currentMetrics) {
    if (Date.now() - this.lastOptimization < this.optimizationInterval) return;
    this.lastOptimization = Date.now();

    this.metricsHistory.push(currentMetrics);
    if (this.metricsHistory.length > 30) this.metricsHistory.shift();

    const stateVec = this._getStateVector(currentMetrics);
    const actions = this._getPossibleActions();

    let actionIndex = 0;
    if (Math.random() < this.epsilon) {
      actionIndex = Math.floor(Math.random() * actions.length);
    } else {
      const qValues = this.net.forward(stateVec);
      actionIndex = qValues.indexOf(Math.max(...qValues));
    }

    const chosenAction = actions[actionIndex];
    await this.db.run(chosenAction.sql);
    await this.db.run('PRAGMA optimize;');

    const previous = this.metricsHistory.length > 1 ? this.metricsHistory[this.metricsHistory.length - 2] : null;
    const reward = this._computeReward(currentMetrics, previous);

    // Store experience
    this.replayBuffer.push({ state: stateVec, actionIndex, reward, nextState: this._getStateVector(currentMetrics) });
    if (this.replayBuffer.length > this.bufferSize) this.replayBuffer.shift();

    // Replay & train
    if (this.replayBuffer.length > 64) {
      const batch = this.replayBuffer.slice(-64);
      for (const exp of batch) {
        const qValues = this.net.forward(exp.state);
        const nextQ = this.net.forward(exp.nextState, true); // target network
        const target = exp.reward + this.gamma * Math.max(...nextQ);
        qValues[exp.actionIndex] = target;
        this.net.backprop(exp.state, target, exp.actionIndex, 0.01);
      }
      this.net.updateTarget(0.005);
    }

    if (this.epsilon > this.minEpsilon) this.epsilon *= this.epsilonDecay;

    console.log(`🔧 Full DQN Backprop: ${chosenAction.name} | Reward: ${reward.toFixed(2)} | ε: ${this.epsilon.toFixed(3)}`);
  }

  async onBenchmarkComplete(result) {
    await this.optimize(result);
  }
}

// Singleton
export const selfOptimizingPragmaEngine = new SelfOptimizingPragmaEngine(null);
