// sac-continuous-flight.js – sovereign client-side Soft Actor-Critic (SAC) for AlphaProMega Air continuous control
// Maximum entropy, twin Q-networks, auto temperature tuning, mercy-shaped rewards
// MIT License – Autonomicity Games Inc. 2026

class SACActor {
  constructor(stateDim = 6, actionDim = 2, hiddenSize = 64) {
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.hiddenSize = hiddenSize;
    this.weights1 = Array(hiddenSize).fill().map(() => Array(stateDim).fill().map(() => Math.random() * 0.2 - 0.1));
    this.bias1 = Array(hiddenSize).fill(0);
    this.muWeights = Array(actionDim).fill().map(() => Array(hiddenSize).fill().map(() => Math.random() * 0.2 - 0.1));
    this.muBias = Array(actionDim).fill(0);
    this.logStdWeights = Array(actionDim).fill().map(() => Array(hiddenSize).fill().map(() => Math.random() * 0.2 - 0.1));
    this.logStdBias = Array(actionDim).fill(0);
    this.logAlpha = Math.log(0.2); // initial temperature
  }

  forward(state) {
    // Hidden layer: ReLU
    const hidden = [];
    for (let i = 0; i < this.hiddenSize; i++) {
      let sum = this.bias1[i];
      for (let j = 0; j < this.stateDim; j++) {
        sum += this.weights1[i][j] * state[j];
      }
      hidden.push(Math.max(0, sum));
    }

    // Mean (mu)
    const mu = [];
    for (let i = 0; i < this.actionDim; i++) {
      let sum = this.muBias[i];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += this.muWeights[i][j] * hidden[j];
      }
      mu.push(Math.tanh(sum) * 3); // bound [-3,3] for thrust/pitch
    }

    // Log std (learnable)
    const logStd = [];
    for (let i = 0; i < this.actionDim; i++) {
      let sum = this.logStdBias[i];
      for (let j = 0; j < this.hiddenSize; j++) {
        sum += this.logStdWeights[i][j] * hidden[j];
      }
      logStd.push(sum);
    }

    return { mu, logStd };
  }

  sampleAction(mu, logStd) {
    const std = logStd.map(Math.exp);
    const action = mu.map((m, i) => m + std[i] * this.gaussianRandom());
    return action;
  }

  gaussianRandom() {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
}

class SACCritic {
  constructor(stateDim = 6, actionDim = 2, hiddenSize = 64) {
    this.weights1 = Array(hiddenSize).fill().map(() => Array(stateDim + actionDim).fill().map(() => Math.random() * 0.2 - 0.1));
    this.bias1 = Array(hiddenSize).fill(0);
    this.weights2 = Array(1).fill().map(() => Array(hiddenSize).fill().map(() => Math.random() * 0.2 - 0.1));
    this.bias2 = [0];
  }

  forward(state, action) {
    const input = state.concat(action);
    const hidden = [];
    for (let i = 0; i < this.weights1.length; i++) {
      let sum = this.bias1[i];
      for (let j = 0; j < input.length; j++) {
        sum += this.weights1[i][j] * input[j];
      }
      hidden.push(Math.max(0, sum));
    }

    let value = this.bias2[0];
    for (let j = 0; j < hidden.length; j++) {
      value += this.weights2[0][j] * hidden[j];
    }
    return value;
  }
}

class SACAgent {
  constructor(stateDim = 6, actionDim = 2) {
    this.actor = new SACActor(stateDim, actionDim);
    this.critic1 = new SACCritic(stateDim, actionDim);
    this.critic2 = new SACCritic(stateDim, actionDim);
    this.targetCritic1 = this.critic1.copy();
    this.targetCritic2 = this.critic2.copy();
    this.replayBuffer = [];
    this.bufferSize = 10000;
    this.batchSize = 64;
    this.gamma = 0.99;
    this.tau = 0.005; // soft target update
    this.learningRate = 0.0003;
    this.targetEntropy = -actionDim; // auto temperature target
    this.logAlpha = Math.log(0.2);
    this.mercyThreshold = 0.9999999;
  }

  getAction(state) {
    const { mu, logStd } = this.actor.forward(state);
    const action = this.actor.sampleAction(mu, logStd);
    return action;
  }

  // Deepened mercy-shaped reward
  computeReward(state, action, nextState) {
    let reward = 0;

    const altError = Math.abs(nextState.targetAltitude - nextState.altitude);
    const velError = Math.abs(nextState.targetVelocity - nextState.velocity);
    if (altError < 50 && velError < 10) reward += 40.0;

    const altPotential = -altError / 1000;
    const velPotential = -velError / 100;
    reward += (altPotential - state.altPotential || 0) * 15;
    reward += (velPotential - state.velPotential || 0) * 10;

    reward += (nextState.energy - state.energy) * 0.15;
    reward += (nextState.integrity - state.integrity) * 30;

    const fleetValence = nextState.integrity * nextState.energy / 100;
    if (fleetValence >= this.mercyThreshold) {
      reward += 15.0 + Math.pow(fleetValence - this.mercyThreshold + 0.001, 2) * 250;
    } else {
      reward -= Math.pow(1 - fleetValence, 3) * 40;
    }

    if (nextState.sti > 0.7) reward += 3.5;

    return reward;
  }

  storeTransition(state, action, reward, nextState, done) {
    this.replayBuffer.push({ state, action, reward, nextState, done });
    if (this.replayBuffer.length > this.bufferSize) {
      this.replayBuffer.shift();
    }
  }

  train() {
    if (this.replayBuffer.length < this.batchSize) return;

    const batch = this.replayBuffer.slice(-this.batchSize);

    // Compute target Q with twin critics
    for (let i = 0; i < batch.length; i++) {
      const exp = batch[i];
      const state = exp.state;
      const action = exp.action;
      const reward = exp.reward;
      const nextState = exp.nextState;
      const done = exp.done;

      const nextAction = this.actor.getAction(nextState);
      const nextQ1 = this.targetCritic1.forward(nextState, nextAction);
      const nextQ2 = this.targetCritic2.forward(nextState, nextAction);
      const nextQ = Math.min(nextQ1, nextQ2);
      const targetQ = reward + (1 - done) * this.gamma * nextQ;

      const currentQ1 = this.critic1.forward(state, action);
      const currentQ2 = this.critic2.forward(state, action);
      const critic1Loss = Math.pow(currentQ1 - targetQ, 2);
      const critic2Loss = Math.pow(currentQ2 - targetQ, 2);

      // Actor loss (policy gradient + entropy)
      const { mu, logStd } = this.actor.forward(state);
      const entropy = this.actor.entropy([mu, logStd]);
      const actorLoss = -this.logAlpha * entropy - Math.min(currentQ1, currentQ2);

      // Alpha loss (auto temperature tuning)
      const alphaLoss = -this.logAlpha * (entropy + this.targetEntropy);

      // Simplified update (real impl would use optimizer + backprop)
      this.critic1.update(this.targetCritic1, this.learningRate * critic1Loss);
      this.critic2.update(this.targetCritic2, this.learningRate * critic2Loss);
      this.actor.update(this.actor, this.learningRate * actorLoss);
      this.logAlpha += this.learningRate * alphaLoss;

      // Soft target update
      this.targetCritic1.update(this.critic1, this.tau);
      this.targetCritic2.update(this.critic2, this.tau);
    }
  }
}

// Export for Ruskode integration
export { PPOAgent };
