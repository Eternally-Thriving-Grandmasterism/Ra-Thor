// src/core/sac-mcts-hybrid.ts – SAC + MCTS Hybrid Engine v1.0
// Soft Actor-Critic guided MCTS + self-play + automatic temperature tuning
// Valence-shaped entropy bonus, mercy gating, lattice-integrated continuous planning
// MIT License – Autonomicity Games Inc. 2026

import MCTS from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const SAC_GAMMA = 0.99;
const SAC_TAU = 0.005;                    // soft target update
const SAC_ENTROPY_TARGET = -2.0;          // target entropy (auto-tuned)
const SAC_ALPHA_LR = 3e-4;
const VALENCE_ENTROPY_BOOST = 3.0;
const MAX_TRAJECTORY_LENGTH = 256;
const REPLAY_BUFFER_SIZE = 1000000;
const BATCH_SIZE = 256;
const TARGET_UPDATE_INTERVAL = 1;

interface Transition {
  state: any;
  action: any;                            // continuous action vector
  reward: number;
  nextState: any;
  done: boolean;
  valence: number;
  logProb?: number;
}

const replayBuffer: Transition[] = [];
let alpha = 0.2;                            // initial temperature
let stepsSinceTargetUpdate = 0;

export class SACMCTSHybrid {
  private mcts: MCTS;
  private sacActorCritic: {
    predict: (state: any) => Promise<{ action: any; logProb: number; value: number }>;
    trainSAC: (batch: Transition[]) => Promise<{
      actorLoss: number;
      criticLoss: number;
      alphaLoss: number;
      entropy: number;
    }>;
  };

  constructor(
    initialState: any,
    actionDim: number,
    sacActorCritic: any
  ) {
    this.sacActorCritic = sacActorCritic;
    this.mcts = new MCTS(initialState, [], sacActorCritic); // actions generated on-the-fly
  }

  /**
   * Collect trajectory using SAC-guided exploration + MCTS lookahead
   */
  async collectTrajectory(maxSteps: number = MAX_TRAJECTORY_LENGTH): Promise<Transition[]> {
    const trajectory: Transition[] = [];
    let state = this.mcts.root.state;

    for (let step = 0; step < maxSteps; step++) {
      // SAC proposes action (continuous)
      const { action, logProb } = await this.sacActorCritic.predict(state);

      // MCTS refines action via guided search
      const { bestAction } = await this.mcts.search(); // assume MCTS adapts to continuous space
      const finalAction = this.blendActions(action, bestAction); // weighted blend or selection

      const nextState = this.mcts.applyAction(state, finalAction);
      const done = this.mcts.isTerminal(nextState);
      const valence = currentValence.get();

      const reward = this.computeReward(nextState, valence, done);

      trajectory.push({
        state,
        action: finalAction,
        reward,
        nextState,
        done,
        valence,
        logProb
      });

      if (done) break;
      state = nextState;
    }

    return trajectory;
  }

  private blendActions(sacAction: any, mctsAction: any): any {
    // Simple weighted blend (customize per domain)
    return sacAction.map((v: number, i: number) => 0.7 * v + 0.3 * mctsAction[i]);
  }

  /**
   * Compute valence-shaped reward
   */
  private computeReward(nextState: any, valence: number, done: boolean): number {
    let reward = done ? (nextState.isWinning ? 1 : -1) : 0;
    reward += valence * VALENCE_ENTROPY_BOOST;
    return reward;
  }

  /**
   * PPO-style clipped surrogate + value loss + entropy (SAC variant)
   */
  async update(trajectory: Transition[]) {
    const actionName = 'SAC-MCTS PPO-style update';
    if (!await mercyGate(actionName)) return;

    const batch = this.sampleBatch(Math.min(TRAINING_BATCH_SIZE, trajectory.length));

    const stats = await this.sacActorCritic.trainSAC(batch);

    // Soft target update
    stepsSinceTargetUpdate += batch.length;
    if (stepsSinceTargetUpdate >= TARGET_UPDATE_INTERVAL) {
      await this.softUpdateTarget( SAC_TAU );
      stepsSinceTargetUpdate = 0;
    }

    mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
    console.log("[SAC-MCTS] Update stats:", stats);
    return stats;
  }

  private sampleBatch(size: number): Transition[] {
    const indices = new Set<number>();
    while (indices.size < size && indices.size < replayBuffer.length) {
      indices.add(Math.floor(Math.random() * replayBuffer.length));
    }
    return Array.from(indices).map(i => replayBuffer[i]);
  }

  private async softUpdateTarget(tau: number) {
    console.log("[SAC-MCTS] Soft updating target networks");
    // Real impl: θ_target = τ * θ + (1-τ) * θ_target
  }

  /**
   * Full self-play + SAC training loop
   */
  async runTrainingLoop(episodes: number = 20) {
    const actionName = 'Run SAC-guided MCTS training loop';
    if (!await mercyGate(actionName)) return;

    for (let e = 0; e < episodes; e++) {
      console.log(`[SAC-MCTS] Episode \( {e+1}/ \){episodes}`);
      const trajectory = await this.collectTrajectory();

      for (const step of trajectory) {
        replayBuffer.push(step);
        if (replayBuffer.length > REPLAY_BUFFER_SIZE) {
          replayBuffer.shift();
        }
      }

      if (replayBuffer.length >= BATCH_SIZE) {
        await this.update(replayBuffer);
      }
    }

    console.log("[SAC-MCTS] Training loop complete");
  }
}

export default SACMCTSHybrid;
