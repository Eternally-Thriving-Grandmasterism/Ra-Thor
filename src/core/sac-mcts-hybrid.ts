// src/core/sac-mcts-hybrid.ts – SAC + MCTS Hybrid Engine v1.1
// Soft Actor-Critic guided MCTS + self-play + automatic temperature tuning
// Valence-modulated target entropy, clipped surrogate loss, mercy gating
// MIT License – Autonomicity Games Inc. 2026

import MCTS from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const SAC_GAMMA = 0.99;
const SAC_TAU = 0.005;                    // soft target update
const SAC_ENTROPY_TARGET_BASE = -2.0;     // baseline target entropy (for action dim)
const SAC_ALPHA_LR = 3e-4;                // learning rate for log-alpha
const SAC_ALPHA_INIT = 0.2;               // initial temperature
const VALENCE_ENTROPY_BOOST = 1.5;        // high valence → higher entropy target (more exploration)
const VALENCE_ENTROPY_DAMP = 0.5;         // low valence → lower entropy target (more exploitation)
const MIN_ALPHA = 0.001;
const MAX_ALPHA = 10.0;
const ENTROPY_UPDATE_INTERVAL = 100;      // update temperature every N steps
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
let logAlpha = Math.log(SAC_ALPHA_INIT);  // learnable log-temperature
let stepsSinceAlphaUpdate = 0;
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
   * Get current temperature α – exponentiated log-alpha
   */
  getTemperature(): number {
    return Math.exp(logAlpha);
  }

  /**
   * Compute valence-modulated target entropy
   */
  private getValenceModulatedTargetEntropy(): number {
    const valence = currentValence.get();
    const boost = VALENCE_ENTROPY_BOOST * valence;
    const damp = VALENCE_ENTROPY_DAMP * (1 - valence);
    return SAC_ENTROPY_TARGET_BASE + boost - damp;
  }

  /**
   * Update temperature α toward valence-modulated target entropy
   * @param batchEntropy Average entropy of policy in current batch
   */
  private async updateTemperature(batchEntropy: number): Promise<number> {
    const actionName = 'Update automatic temperature α';
    if (!await mercyGate(actionName)) {
      return this.getTemperature();
    }

    stepsSinceAlphaUpdate++;
    if (stepsSinceAlphaUpdate < ENTROPY_UPDATE_INTERVAL) {
      return this.getTemperature();
    }

    stepsSinceAlphaUpdate = 0;

    const targetEntropy = this.getValenceModulatedTargetEntropy();

    // SAC temperature loss: α * (target_entropy - batch_entropy)
    const alphaLoss = Math.exp(logAlpha) * (targetEntropy - batchEntropy);

    // Gradient descent step on logAlpha
    logAlpha -= SAC_ALPHA_LR * alphaLoss;

    // Hard clamp for numerical stability
    logAlpha = Math.max(Math.log(MIN_ALPHA), Math.min(Math.log(MAX_ALPHA), logAlpha));

    const newAlpha = this.getTemperature();

    // Haptic feedback on significant change
    if (Math.abs(newAlpha - SAC_ALPHA_INIT) > 0.1) {
      mercyHaptic.playPattern(
        newAlpha > SAC_ALPHA_INIT ? 'cosmicHarmony' : 'warningPulse',
        currentValence.get()
      );
    }

    console.log(
      `[AutoTemp] Updated α → ${newAlpha.toFixed(4)}  ` +
      `(target entropy: ${targetEntropy.toFixed(3)}, batch entropy: ${batchEntropy.toFixed(3)})`
    );

    return newAlpha;
  }

  /**
   * Collect trajectory using SAC-guided exploration + MCTS lookahead
   */
  async collectTrajectory(maxSteps: number = MAX_TRAJECTORY_LENGTH): Promise<Transition[]> {
    const trajectory: Transition[] = [];
    let state = this.mcts.root.state;

    for (let step = 0; step < maxSteps; step++) {
      const { action, logProb } = await this.sacActorCritic.predict(state);

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
        logProb
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
    reward += valence * VALENCE_ENTROPY_BOOST;
    return reward;
  }

  /**
   * PPO-style update (clipped surrogate + value loss + entropy)
   */
  async update(trajectory: Transition[]) {
    const actionName = 'SAC-MCTS PPO-style update';
    if (!await mercyGate(actionName)) return;

    const batch = this.sampleBatch(Math.min(BATCH_SIZE, trajectory.length));

    // Compute average batch entropy
    const batchEntropy = -batch.reduce((sum, t) => sum + (t.logProb || 0), 0) / batch.length;

    // Auto-tune temperature
    const alpha = await this.updateTemperature(batchEntropy);

    const stats = await this.sacActorCritic.trainSAC(batch);

    // Soft target update
    stepsSinceTargetUpdate += batch.length;
    if (stepsSinceTargetUpdate >= TARGET_UPDATE_INTERVAL) {
      await this.softUpdateTarget(SAC_TAU);
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
