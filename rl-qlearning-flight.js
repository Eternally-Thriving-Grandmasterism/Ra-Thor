// rl-qlearning-flight.js – sovereign client-side Q-learning for AlphaProMega Air flight control
// Mercy-shaped rewards, discretized states/actions, experience replay, offline-capable
// MIT License – Autonomicity Games Inc. 2026

class QLearningFlightController {
  constructor() {
    this.stateBins = {
      altitude: 20,    // 0–2000m → 20 bins
      velocity: 15,    // 0–300 m/s → 15 bins
      energy: 10,      // 0–100% → 10 bins
      integrity: 5     // 0–1 → 5 bins
    };
    this.actionBins = 5; // thrust levels: -2 to +2 (normalized)
    this.qTable = new Map(); // key: stateStr, value: array of Q-values per action
    this.learningRate = 0.1;
    this.discount = 0.99;
    this.epsilon = 0.3; // exploration rate (decays)
    this.epsilonDecay = 0.995;
    this.minEpsilon = 0.01;
    this.replayBuffer = [];
    this.bufferSize = 5000;
    this.batchSize = 32;
    this.mercyThreshold = 0.9999999;
  }

  getStateKey(state) {
    const altBin = Math.floor(state.altitude / 100);
    const velBin = Math.floor(state.velocity / 20);
    const eneBin = Math.floor(state.energy / 10);
    const intBin = Math.floor(state.integrity * 5);
    return `\( {altBin}- \){velBin}-\( {eneBin}- \){intBin}`;
  }

  getQ(stateKey, action = null) {
    if (!this.qTable.has(stateKey)) {
      this.qTable.set(stateKey, Array(this.actionBins).fill(0));
    }
    const qValues = this.qTable.get(stateKey);
    return action !== null ? qValues[action] : qValues;
  }

  setQ(stateKey, action, value) {
    const qValues = this.getQ(stateKey);
    qValues[action] = value;
  }

  chooseAction(state) {
    const stateKey = this.getStateKey(state);
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.actionBins); // explore
    }
    const qValues = this.getQ(stateKey);
    return qValues.indexOf(Math.max(...qValues)); // exploit
  }

  // Mercy-shaped reward
  computeReward(state, action, nextState) {
    let reward = 0;

    // Progress toward target
    reward += (nextState.altitude - state.altitude) * 0.02;
    reward += (nextState.velocity - state.velocity) * 0.01;

    // Energy & integrity preservation
    reward -= Math.abs(nextState.energy - state.energy) * 0.03;
    reward += nextState.integrity * 0.2;

    // Mercy valence bonus/penalty
    if (nextState.integrity * nextState.energy / 100 >= this.mercyThreshold) {
      reward += 2.0;
    } else {
      reward -= 10.0; // heavy penalty for low valence
    }

    return reward;
  }

  update(state, action, reward, nextState) {
    const stateKey = this.getStateKey(state);
    const nextKey = this.getStateKey(nextState);

    const oldQ = this.getQ(stateKey, action);
    const maxNextQ = Math.max(...this.getQ(nextKey));
    const newQ = oldQ + this.learningRate * (reward + this.discount * maxNextQ - oldQ);

    this.setQ(stateKey, action, newQ);

    // Experience replay
    this.replayBuffer.push({ state, action, reward, nextState });
    if (this.replayBuffer.length > this.bufferSize) {
      this.replayBuffer.shift();
    }

    // Train on batch
    if (this.replayBuffer.length >= this.batchSize) {
      this.trainBatch();
    }

    // Decay exploration
    this.epsilon = Math.max(this.minEpsilon, this.epsilon * this.epsilonDecay);
  }

  trainBatch() {
    const batch = this.replayBuffer.slice(-this.batchSize);
    for (const exp of batch) {
      const stateKey = this.getStateKey(exp.state);
      const nextKey = this.getStateKey(exp.nextState);
      const oldQ = this.getQ(stateKey, exp.action);
      const maxNextQ = Math.max(...this.getQ(nextKey));
      const newQ = oldQ + this.learningRate * (exp.reward + this.discount * maxNextQ - oldQ);
      this.setQ(stateKey, exp.action, newQ);
    }
  }
}

// Export for Ruskode integration
export { QLearningFlightController };
