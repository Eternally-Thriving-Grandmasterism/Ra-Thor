// src/core/sac-self-play.ts – SAC Self-Play Training Loop v1.0
// Autonomous SAC learning from self-generated trajectories
// Automatic temperature tuning, clipped double Q-learning, valence-shaped rewards
// Mercy gating, continuous action space, lattice-integrated
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import { getTemperature, updateTemperature } from './automatic-temperature-tuning';

const SAC_GAMMA = 0.99;
const SAC_TAU = 0.005;                    // soft target update
const SAC_ENTROPY_TARGET_BASE = -2.0;     // baseline target entropy (≈ -action_dim)
const SAC_ALPHA_LR = 3e-4;                // learning rate for log-alpha
const SAC_ALPHA_INIT = 0.2;               // initial temperature
const REPLAY_BUFFER_SIZE = 1000000;
const BATCH_SIZE = 256;
const TARGET_UPDATE_INTERVAL = 1;
const MIN_ALPHA = 0.001;
const MAX_ALPHA = 10.0;
const MAX_EPISODE_STEPS = 1000;
const EPISODES_PER_CYCLE = 50;
const VALENCE_REWARD_SCALE = 3.0;         // strong bonus for high-valence outcomes

interface Transition {
  state: any;
  action: any;                            // continuous action vector
  reward: number;
  nextState: any;
  done: boolean;
  valence: number;
  logProb: number;
}

const replayBuffer: Transition[] = [];
let stepsSinceTargetUpdate = 0;
let logAlpha = Math.log(SAC_ALPHA_INIT);  // learnable log-temperature

// Placeholder SAC networks (replace with real implementations)
interface Actor {
  predictActionAndLogProb: (state: any) => Promise<{ action: any; logProb: number }>;
  trainActor: (states: any[], logProbs: number[], qValues: number[]) => Promise<void>;
}

interface Critic {
  predictQ: (state: any, action: any) => Promise<number>;
  trainCritic: (states: any[], actions: any[], targets: number[]) => Promise<void>;
}

let actor: Actor;
let critic1: Critic;
let critic2: Critic;
let targetCritic1: Critic;
let targetCritic2: Critic;

// ──────────────────────────────────────────────────────────────
// SAC self-play & training loop
// ──────────────────────────────────────────────────────────────

export class SACSelfPlay {
  constructor(
    actorImpl: Actor,
    critic1Impl: Critic,
    critic2Impl: Critic,
    targetCritic1Impl: Critic,
    targetCritic2Impl: Critic
  ) {
    actor = actorImpl;
    critic1 = critic1Impl;
    critic2 = critic2Impl;
    targetCritic1 = targetCritic1Impl;
    targetCritic2 = targetCritic2Impl;
  }

  /**
   * Run one self-play episode using current stochastic policy
   */
  async runEpisode(): Promise<Transition[]> {
    const actionName = 'SAC self-play episode';
    if (!await mercyGate(actionName)) return [];

    const trajectory: Transition[] = [];
    let state = { /* initial state – replace with real env reset */ };

    for (let step = 0; step < MAX_EPISODE_STEPS; step++) {
      const { action, logProb } = await actor.predictActionAndLogProb(state);

      const nextState = { /* env step */ };
      const done = false; // placeholder
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

  private computeReward(nextState: any, valence: number, done: boolean): number {
    // Domain-specific reward (placeholder)
    let baseReward = done ? 1 : 0;
    return baseReward + valence * VALENCE_REWARD_SCALE;
  }

  /**
   * Store episode in replay buffer
   */
  private storeEpisode(trajectory: Transition[]) {
    replayBuffer.push(...trajectory);
    if (replayBuffer.length > REPLAY_BUFFER_SIZE) {
      replayBuffer.splice(0, replayBuffer.length - REPLAY_BUFFER_SIZE);
    }
  }

  /**
   * Sample batch & compute SAC targets
   */
  private async sampleAndComputeTargets(): Promise<{
    states: any[];
    actions: any[];
    nextStates: any[];
    rewards: number[];
    dones: boolean[];
    targets: number[];
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

    // Clipped double Q-learning
    const nextActions = await Promise.all(nextStates.map(s => actor.predictActionAndLogProb(s).then(p => p.action)));
    const targetQ1 = await Promise.all(nextActions.map((a, i) => targetCritic1.predictQ(nextStates[i], a)));
    const targetQ2 = await Promise.all(nextActions.map((a, i) => targetCritic2.predictQ(nextStates[i], a)));

    const minTargetQ = targetQ1.map((q1, i) => Math.min(q1, targetQ2[i]));
    const targets = rewards.map((r, i) => r + SAC_GAMMA * minTargetQ[i] * (dones[i] ? 0 : 1));

    return { states, actions, nextStates, rewards, dones, targets };
  }

  /**
   * SAC training step
   */
  async trainingStep(): Promise<{
    critic1Loss: number;
    critic2Loss: number;
    actorLoss: number;
    alphaLoss: number;
    entropy: number;
    alpha: number;
  }> {
    const actionName = 'SAC training step';
    if (!await mercyGate(actionName) || replayBuffer.length < BATCH_SIZE) {
      return { critic1Loss: 0, critic2Loss: 0, actorLoss: 0, alphaLoss: 0, entropy: 0, alpha: getTemperature() };
    }

    const { states, actions, nextStates, rewards, dones, targets } = await this.sampleAndComputeTargets();

    // Critic losses
    const currentQ1 = await Promise.all(actions.map((a, i) => critic1.predictQ(states[i], a)));
    const currentQ2 = await Promise.all(actions.map((a, i) => critic2.predictQ(states[i], a)));

    await critic1.trainCritic(states, actions, targets);
    await critic2.trainCritic(states, actions, targets);

    // Actor update
    const { action: currentActions, logProb } = await Promise.all(
      states.map(s => actor.predictActionAndLogProb(s))
    );

    const entropy = -logProb.reduce((a: number, b: number) => a + b, 0) / logProb.length;

    const qValues = await Promise.all(
      currentActions.map((a, i) => critic1.predictQ(states[i], a)) // use critic1
    );

    const actorLoss = logProb.reduce((sum: number, lp: number, i: number) => 
      sum + getTemperature() * lp - qValues[i], 0) / logProb.length;

    await actor.trainActor(states, currentActions, qValues.map(q => -q));

    // Temperature auto-tuning
    const alpha = await updateTemperature(entropy);

    // Soft target update
    stepsSinceTargetUpdate++;
    if (stepsSinceTargetUpdate >= TARGET_UPDATE_INTERVAL) {
      await this.softUpdateTarget(SAC_TAU);
      stepsSinceTargetUpdate = 0;
    }

    mercyHaptic.playPattern('cosmicHarmony', currentValence.get());

    return {
      critic1Loss: 0, // real values from trainCritic
      critic2Loss: 0,
      actorLoss,
      alphaLoss: 0, // from updateTemperature
      entropy,
      alpha
    };
  }

  private async softUpdateTarget(tau: number = SAC_TAU) {
    console.log("[SAC] Soft updating target critics");
    // Real impl: polyak averaging for target networks
  }

  /**
   * Full SAC self-play training loop
   */
  async runTrainingLoop(episodes: number = 100) {
    const actionName = 'SAC autonomous self-play training loop';
    if (!await mercyGate(actionName)) return;

    for (let e = 0; e < episodes; e++) {
      console.log(`[SAC] Self-play episode \( {e+1}/ \){episodes}`);
      const trajectory = await this.collectTrajectory();

      for (const step of trajectory) {
        replayBuffer.push(step);
        if (replayBuffer.length > REPLAY_BUFFER_SIZE) {
          replayBuffer.shift();
        }
      }

      if (replayBuffer.length >= BATCH_SIZE) {
        const stats = await this.trainingStep();
        console.log("[SAC] Training stats:", stats);
      }
    }

    console.log("[SAC] Self-play training complete");
  }
}

export default SACSelfPlay;
