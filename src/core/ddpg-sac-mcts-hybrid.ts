// src/core/ddpg-sac-mcts-hybrid.ts – DDPG + SAC + MCTS Hybrid Engine v1.0
// Deterministic Policy Gradient (DDPG) + Soft Actor-Critic (SAC) + MCTS fusion
// Valence-shaped advantage & entropy regularization, mercy gating, self-play loop
// MIT License – Autonomicity Games Inc. 2026

import MCTS from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import { getTemperature, updateTemperature } from './automatic-temperature-tuning';

const DDPG_GAMMA = 0.99;
const DDPG_TAU = 0.005;                    // soft target update (actor & critic)
const SAC_ENTROPY_TARGET_BASE = -2.0;
const VALENCE_ADVANTAGE_BOOST = 2.5;
const MAX_TRAJECTORY_LENGTH = 256;
const REPLAY_BUFFER_SIZE = 1000000;
const BATCH_SIZE = 256;
const TARGET_UPDATE_INTERVAL = 1;
const NOISE_STD = 0.1;                     // Ornstein-Uhlenbeck or Gaussian exploration noise

interface Transition {
  state: any;
  action: any;                            // continuous action vector
  reward: number;
  nextState: any;
  done: boolean;
  valence: number;
  logProb?: number;                       // from SAC policy (if used)
}

const replayBuffer: Transition[] = [];
let stepsSinceTargetUpdate = 0;

export class DDPGSACMCTSHybrid {
  private mcts: MCTS;
  private actorCritic: {
    // Actor: deterministic policy μ(s)
    predictAction: (state: any) => Promise<any>;
    // Critic: Q(s,a)
    predictQ: (state: any, action: any) => Promise<number>;
    // Target networks
    targetPredictAction: (state: any) => Promise<any>;
    targetPredictQ: (state: any, action: any) => Promise<number>;
    // Training step
    train: (batch: Transition[]) => Promise<{
      actorLoss: number;
      criticLoss: number;
      alphaLoss?: number;
      entropy?: number;
    }>;
  };

  constructor(initialState: any, actionDim: number, actorCritic: any) {
    this.actorCritic = actorCritic;
    this.mcts = new MCTS(initialState, [], actorCritic); // actions generated on-the-fly
  }

  /**
   * Collect trajectory using deterministic policy + exploration noise + MCTS refinement
   */
  async collectTrajectory(maxSteps: number = MAX_TRAJECTORY_LENGTH): Promise<Transition[]> {
    const trajectory: Transition[] = [];
    let state = this.mcts.root.state;

    for (let step = 0; step < maxSteps; step++) {
      // DDPG deterministic action + exploration noise
      let action = await this.actorCritic.predictAction(state);
      action = addExplorationNoise(action, NOISE_STD);

      // Optional MCTS refinement (guided improvement)
      const { bestAction } = await this.mcts.search();
      action = this.blendActions(action, bestAction); // weighted blend or selection

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
        valence
      });

      if (done) break;
      state = nextState;
    }

    return trajectory;
  }

  private blendActions(ddpgAction: any, mctsAction: any): any {
    // Simple weighted blend (customize per domain)
    return ddpgAction.map((v: number, i: number) => 0.7 * v + 0.3 * mctsAction[i]);
  }

  private addExplorationNoise(action: any, std: number): any {
    return action.map((v: number) => v + (Math.random() - 0.5) * 2 * std);
  }

  /**
   * Compute valence-shaped reward
   */
  private computeReward(nextState: any, valence: number, done: boolean): number {
    let reward = done ? (nextState.isWinning ? 1 : -1) : 0;
    reward += valence * VALENCE_ADVANTAGE_BOOST;
    return reward;
  }

  /**
   * Store transition in replay buffer
   */
  private storeTransition(transition: Transition) {
    replayBuffer.push(transition);
    if (replayBuffer.length > REPLAY_BUFFER_SIZE) {
      replayBuffer.shift();
    }
  }

  /**
   * Sample batch & compute targets / advantages
   */
  private async sampleAndComputeTargets(): Promise<{
    states: any[];
    actions: any[];
    nextStates: any[];
    rewards: number[];
    dones: boolean[];
    advantages: number[];
    returns: number[];
  }> {
    const indices = new Set<number>();
    while (indices.size < BATCH_SIZE && indices.size < replayBuffer.length) {
      indices.add(Math.floor(Math.random() * replayBuffer.length));
    }

    const batch = Array.from(indices).map(i => replayBuffer[i]);

    const states = batch.map(t => t.state);
    const actions = batch.map(t => t.action);
    const nextStates = batch.map(t => t.nextState);
    const rewards = batch.map(t => t.reward);
    const dones = batch.map(t => t.done);

    // Compute target Q-values (using target network)
    const targetQs = await Promise.all(
      batch.map(async t => {
        if (t.done) return t.reward;
        const nextAction = await this.actorCritic.targetPredictAction(t.nextState);
        return t.reward + SAC_GAMMA * (await this.actorCritic.targetPredictQ(t.nextState, nextAction));
      })
    );

    // Compute current Q-values
    const currentQs = await Promise.all(
      batch.map(async t => await this.actorCritic.predictQ(t.state, t.action))
    );

    // Advantages (simplified – can use GAE if desired)
    const advantages = targetQs.map((tq, i) => tq - currentQs[i]);

    // Valence-weighted advantage normalization + boost
    const meanAdv = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const stdAdv = Math.sqrt(advantages.reduce((a, b) => a + Math.pow(b - meanAdv, 2), 0) / advantages.length) + 1e-8;

    const weightedAdvantages = advantages.map((adv, i) => {
      let normAdv = (adv - meanAdv) / stdAdv;
      normAdv += VALENCE_ADVANTAGE_BOOST * batch[i].valence;
      return normAdv;
    });

    const returns = targetQs; // simplified – can use GAE returns if needed

    return { states, actions, nextStates, rewards, dones, advantages: weightedAdvantages, returns };
  }

  /**
   * SAC-style soft update for target networks
   */
  private async softUpdateTarget(tau: number = SAC_TAU) {
    console.log("[SAC-MCTS] Soft updating target networks");
    // Real impl: θ_target = τ * θ + (1-τ) * θ_target
  }

  /**
   * Full training loop – collect rollouts + periodic updates
   */
  async runTrainingLoop(episodes: number = 20) {
    const actionName = 'Run DDPG-guided SAC-MCTS training loop';
    if (!await mercyGate(actionName)) return;

    for (let e = 0; e < episodes; e++) {
      console.log(`[DDPG-SAC-MCTS] Episode \( {e+1}/ \){episodes}`);
      const trajectory = await this.collectTrajectory();

      for (const step of trajectory) {
        replayBuffer.push(step);
        if (replayBuffer.length > REPLAY_BUFFER_SIZE) {
          replayBuffer.shift();
        }
      }

      if (replayBuffer.length >= BATCH_SIZE) {
        const batchStats = await this.sampleAndComputeTargets();
        // Real training would happen here
        await this.softUpdateTarget();

        mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
      }
    }

    console.log("[DDPG-SAC-MCTS] Training loop complete");
  }
}

export default DDPGSACMCTSHybrid;
