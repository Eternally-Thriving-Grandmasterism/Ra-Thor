// src/core/dqn-mcts-hybrid.ts – DQN + MCTS Hybrid Engine v1.0
// Deep Q-Network guided MCTS + experience replay + target network
// Valence-shaped Q-bonus, mercy gating, lattice-integrated planning
// MIT License – Autonomicity Games Inc. 2026

import MCTS from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const DQN_GAMMA = 0.99;
const DQN_EPSILON_START = 1.0;
const DQN_EPSILON_END = 0.05;
const DQN_EPSILON_DECAY = 0.995;
const REPLAY_BUFFER_SIZE = 100000;
const BATCH_SIZE = 64;
const TARGET_UPDATE_FREQ = 1000;
const VALENCE_Q_BONUS = 2.5;
const MAX_TRAJECTORY_LENGTH = 256;
const LEARNING_RATE = 1e-4;

interface Transition {
  state: any;
  action: string;
  reward: number;
  nextState: any;
  done: boolean;
  valence: number;
  qValue?: number;
}

interface ReplayBuffer {
  buffer: Transition[];
  index: number;
  size: number;
}

const replayBuffer: ReplayBuffer = {
  buffer: [],
  index: 0,
  size: 0
};

let stepsSinceTargetUpdate = 0;
let epsilon = DQN_EPSILON_START;

export class DQNMCTSHybrid {
  private mcts: MCTS;
  private qNetwork: NeuralNetwork & { 
    predictQ: (state: any, action?: string) => Promise<number>;
    train: (batch: Transition[]) => Promise<void>;
  };
  private targetNetwork: NeuralNetwork & { predictQ: (state: any, action?: string) => Promise<number> };

  constructor(
    initialState: any,
    initialActions: string[],
    qNetwork: any,
    targetNetwork: any
  ) {
    this.mcts = new MCTS(initialState, initialActions, qNetwork);
    this.qNetwork = qNetwork;
    this.targetNetwork = targetNetwork;
  }

  /**
   * Collect trajectory using ε-greedy + MCTS-guided exploration
   */
  async collectTrajectory(maxSteps: number = MAX_TRAJECTORY_LENGTH): Promise<Transition[]> {
    const trajectory: Transition[] = [];
    let state = this.mcts.root.state;

    for (let step = 0; step < maxSteps; step++) {
      const epsilonCurrent = epsilon * Math.pow(DQN_EPSILON_DECAY, step);
      const explore = Math.random() < epsilonCurrent;

      let action: string;
      let qValue: number | undefined;

      if (explore) {
        action = this.mcts.root.untriedActions[Math.floor(Math.random() * this.mcts.root.untriedActions.length)];
      } else {
        const { bestAction } = await this.mcts.search();
        action = bestAction;
        qValue = await this.qNetwork.predictQ(state, action);
      }

      const nextState = this.mcts.applyAction(state, action);
      const done = this.mcts.isTerminal(nextState);
      const valence = currentValence.get();

      const reward = this.computeReward(nextState, valence, done);

      trajectory.push({
        state,
        action,
        reward,
        nextState,
        done,
        valence,
        qValue
      });

      if (done) break;
      state = nextState;
    }

    return trajectory;
  }

  /**
   * Compute valence-shaped reward
   */
  private computeReward(nextState: any, valence: number, done: boolean): number {
    let reward = done ? (nextState.isWinning ? 1 : -1) : 0;
    reward += valence * VALENCE_Q_BONUS;
    return reward;
  }

  /**
   * Store transition in replay buffer
   */
  private storeTransition(transition: Transition) {
    replayBuffer.buffer[replayBuffer.index] = transition;
    replayBuffer.index = (replayBuffer.index + 1) % REPLAY_BUFFER_SIZE;
    replayBuffer.size = Math.min(replayBuffer.size + 1, REPLAY_BUFFER_SIZE);
  }

  /**
   * Sample batch & compute targets / advantages
   */
  private async sampleAndComputeTargets(): Promise<{
    states: any[];
    actions: string[];
    targets: number[];
    advantages: number[];
  }> {
    const indices = new Set<number>();
    while (indices.size < BATCH_SIZE && indices.size < replayBuffer.size) {
      indices.add(Math.floor(Math.random() * replayBuffer.size));
    }

    const batch = Array.from(indices).map(i => replayBuffer.buffer[i]);

    const states: any[] = [];
    const actions: string[] = [];
    const targets: number[] = [];
    const advantages: number[] = [];

    for (const t of batch) {
      states.push(t.state);
      actions.push(t.action);

      const nextMaxQ = t.done ? 0 : await this.targetNetwork.predictQ(t.nextState);
      const targetQ = t.reward + DQN_GAMMA * nextMaxQ;

      const currentQ = t.qValue ?? await this.qNetwork.predictQ(t.state, t.action);
      const advantage = targetQ - currentQ;

      targets.push(targetQ);
      advantages.push(advantage);
    }

    return { states, actions, targets, advantages };
  }

  /**
   * PPO-style update (clipped surrogate + value loss)
   */
  async update() {
    const actionName = 'DQN-MCTS PPO-style update';
    if (!await mercyGate(actionName)) return;

    const { states, actions, targets, advantages } = await this.sampleAndComputeTargets();

    const stats = await this.qNetwork.train(
      states,
      actions,
      targets,
      advantages
    );

    // Update target network periodically
    stepsSinceTargetUpdate += BATCH_SIZE;
    if (stepsSinceTargetUpdate >= TARGET_UPDATE_FREQ) {
      // Soft update (polyak averaging) or hard copy
      await this.softUpdateTargetNetwork(0.005);
      stepsSinceTargetUpdate = 0;
    }

    mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
    console.log("[DQN-MCTS] Update stats:", stats);
    return stats;
  }

  private async softUpdateTargetNetwork(tau: number = 0.005) {
    // Polyak averaging for target network
    console.log("[DQN-MCTS] Soft updating target network");
    // Real impl: θ_target = τ * θ + (1-τ) * θ_target
  }

  /**
   * Full self-play + DQN training loop
   */
  async runTrainingLoop(episodes: number = 20) {
    const actionName = 'Run DQN-guided MCTS training loop';
    if (!await mercyGate(actionName)) return;

    for (let e = 0; e < episodes; e++) {
      console.log(`[DQN-MCTS] Episode \( {e+1}/ \){episodes}`);
      const trajectory = await this.collectTrajectory();

      for (const step of trajectory) {
        this.storeTransition(step);
      }

      if (replayBuffer.size >= BATCH_SIZE) {
        await this.update();
      }
    }

    console.log("[DQN-MCTS] Training loop complete");
  }
}

// Mock Q-network with training stub (replace with real implementation)
class MockQNetwork {
  async predictQ(state: any, action?: string) {
    return currentValence.get();
  }

  async train(states: any[], actions: string[], targets: number[], advantages: number[]) {
    console.log(`[MockQNetwork] Training on ${states.length} steps`);
    return {
      qLoss: 0.08,
      advantageLoss: -0.12
    };
  }
}

export default DQNMCTSHybrid;
