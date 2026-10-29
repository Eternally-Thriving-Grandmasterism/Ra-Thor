// src/core/a2c-mcts-hybrid.ts – A2C + MCTS Hybrid Engine v1.0
// Advantage Actor-Critic guided MCTS + self-play training loop
// Valence-shaped advantage, mercy gating, lattice-integrated planning
// MIT License – Autonomicity Games Inc. 2026

import MCTS from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const A2C_GAMMA = 0.99;
const A2C_LAMBDA = 0.95;              // GAE lambda
const A2C_ENTROPY_COEF = 0.01;
const A2C_VALUE_LOSS_COEF = 0.5;
const A2C_POLICY_LOSS_COEF = 1.0;
const VALENCE_ADVANTAGE_BOOST = 2.5;
const MAX_TRAJECTORY_LENGTH = 256;
const TRAINING_BATCH_SIZE = 64;

interface TrajectoryStep {
  state: any;
  action: string;
  logProb: number;
  value: number;
  reward: number;
  nextState: any;
  done: boolean;
  valence: number;
}

interface A2CUpdateStats {
  policyLoss: number;
  valueLoss: number;
  entropy: number;
}

export class A2CMCTSHybrid {
  private mcts: MCTS;
  private actorCritic: {
    predict: (state: any) => Promise<{ policy: Map<string, number>; value: number }>;
    train: (batch: TrajectoryStep[], advantages: number[], returns: number[]) => Promise<A2CUpdateStats>;
  };

  constructor(initialState: any, initialActions: string[], actorCritic: any) {
    this.actorCritic = actorCritic;
    this.mcts = new MCTS(initialState, initialActions, actorCritic);
  }

  /**
   * Collect trajectory using MCTS-guided policy (self-play episode)
   */
  async collectTrajectory(maxSteps: number = MAX_TRAJECTORY_LENGTH): Promise<TrajectoryStep[]> {
    const trajectory: TrajectoryStep[] = [];
    let state = this.mcts.root.state;

    for (let step = 0; step < maxSteps; step++) {
      const { bestAction, policy } = await this.mcts.search();

      const actionLogProb = Math.log(policy.get(bestAction) || 1e-8);
      const nextState = this.mcts.applyAction(state, bestAction);
      const done = this.mcts.isTerminal(nextState);
      const valence = currentValence.get();

      const reward = this.computeReward(nextState, valence, done);

      trajectory.push({
        state,
        action: bestAction,
        logProb: actionLogProb,
        value: 0, // filled later via bootstrapping
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

  /**
   * Compute GAE advantages & discounted returns (A2C standard)
   */
  computeAdvantagesAndReturns(trajectory: TrajectoryStep[]): { advantages: number[]; returns: number[] } {
    const advantages: number[] = new Array(trajectory.length);
    const returns: number[] = new Array(trajectory.length);

    let nextValue = 0;
    let nextAdvantage = 0;

    for (let t = trajectory.length - 1; t >= 0; t--) {
      const step = trajectory[t];
      const delta = step.reward + A2C_GAMMA * nextValue * (step.done ? 0 : 1) - step.value;
      nextAdvantage = delta + A2C_GAMMA * A2C_LAMBDA * (step.done ? 0 : 1) * nextAdvantage;

      advantages[t] = nextAdvantage;
      returns[t] = advantages[t] + step.value;

      nextValue = step.value;
    }

    // Valence-shaped advantage normalization
    const meanAdv = advantages.reduce((a, b) => a + b, 0) / advantages.length;
    const stdAdv = Math.sqrt(advantages.reduce((a, b) => a + Math.pow(b - meanAdv, 2), 0) / advantages.length) + 1e-8;

    for (let i = 0; i < advantages.length; i++) {
      advantages[i] = (advantages[i] - meanAdv) / stdAdv;
      advantages[i] += VALENCE_ADVANTAGE_BOOST * trajectory[i].valence;
    }

    return { advantages, returns };
  }

  /**
   * PPO-style update (clipped surrogate + value loss + entropy)
   */
  async update(trajectory: TrajectoryStep[]) {
    const actionName = 'A2C-PPO style update';
    if (!await mercyGate(actionName)) return;

    const { advantages, returns } = this.computeAdvantagesAndReturns(trajectory);

    const stats = await this.actorCritic.train(trajectory, advantages, returns);

    mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
    console.log("[A2C-MCTS] Update stats:", stats);
    return stats;
  }

  /**
   * Full self-play + A2C training loop
   */
  async runTrainingLoop(episodes: number = 20) {
    const actionName = 'Run A2C-guided MCTS training loop';
    if (!await mercyGate(actionName)) return;

    for (let e = 0; e < episodes; e++) {
      console.log(`[A2C-MCTS] Episode \( {e+1}/ \){episodes}`);
      const trajectory = await this.collectTrajectory();

      if (trajectory.length > 0) {
        await this.update(trajectory);
      }
    }

    console.log("[A2C-MCTS] Training loop complete");
  }

  private computeReward(nextState: any, valence: number, done: boolean): number {
    let reward = done ? (nextState.isWinning ? 1 : -1) : 0;
    reward += valence * 0.8; // valence shaping
    return reward;
  }
}

// Mock actor-critic net with A2C/PPO training stub (replace with real impl)
class MockActorCritic {
  async predict(state: any) {
    return {
      policy: new Map([['action1', 0.4], ['action2', 0.3], ['action3', 0.3]]),
      value: currentValence.get()
    };
  }

  async train(batch: any[], advantages: number[], returns: number[]) {
    console.log(`[MockActorCritic] Training on ${batch.length} steps`);
    // Real impl: PPO clipped surrogate + value loss + entropy
    return {
      policyLoss: -0.12,
      valueLoss: 0.08,
      entropy: 1.15
    };
  }
}

export default A2CMCTSHybrid;
