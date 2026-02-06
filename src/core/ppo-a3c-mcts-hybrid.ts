// src/core/ppo-a3c-mcts-hybrid.ts – PPO + A3C + MCTS Hybrid Engine v1.0
// Proximal Policy Optimization + Asynchronous Advantage Actor-Critic + MCTS
// Valence-shaped advantage, clipped surrogate + async gradients, mercy gating
// MIT License – Autonomicity Games Inc. 2026

import MCTS from './alphago-style-mcts-neural';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const PPO_CLIP_EPSILON = 0.2;
const PPO_VALUE_LOSS_COEF = 0.5;
const PPO_ENTROPY_COEF = 0.01;
const A3C_GAMMA = 0.99;
const A3C_LAMBDA = 0.95;
const VALENCE_ADVANTAGE_BOOST = 2.5;
const MAX_TRAJECTORY_LENGTH = 256;
const GLOBAL_UPDATE_INTERVAL = 32;        // A3C-style async updates
const REPLAY_BUFFER_SIZE = 1000000;
const BATCH_SIZE = 256;

interface TrajectoryStep {
  state: any;
  action: any;
  oldLogProb: number;
  newLogProb: number;
  value: number;
  reward: number;
  nextState: any;
  done: boolean;
  valence: number;
}

const globalReplayBuffer: TrajectoryStep[] = [];
let globalStepCounter = 0;

export class PPOA3CMCTSHybrid {
  private mcts: MCTS;
  private policyNet: {
    predictPolicyAndValue: (state: any) => Promise<{ policy: Map<string, number>; value: number }>;
    trainPPO: (batch: TrajectoryStep[], advantages: number[], returns: number[]) => Promise<any>;
  };
  private workers: Worker[] = [];

  constructor(initialState: any, initialActions: string[], policyNet: any) {
    this.policyNet = policyNet;
    this.mcts = new MCTS(initialState, initialActions, policyNet);
    this.initializeWorkers();
  }

  private initializeWorkers() {
    const numWorkers = navigator.hardwareConcurrency || 4;
    for (let i = 0; i < numWorkers; i++) {
      const worker = new Worker(URL.createObjectURL(new Blob([`
        self.onmessage = async function(e) {
          const { command, initialState, initialActions } = e.data;
          if (command === 'rollout') {
            // Worker simulates rollout (simplified – real impl would use shared memory)
            const trajectory = []; // placeholder
            self.postMessage({ trajectory });
          }
        };
      `], { type: 'text/javascript' })));
      worker.onmessage = (e) => this.handleWorkerResult(e.data);
      this.workers.push(worker);
    }
  }

  private handleWorkerResult(data: any) {
    const trajectory = data.trajectory;
    globalReplayBuffer.push(...trajectory);
    if (globalReplayBuffer.length > REPLAY_BUFFER_SIZE) {
      globalReplayBuffer.splice(0, globalReplayBuffer.length - REPLAY_BUFFER_SIZE);
    }

    globalStepCounter += trajectory.length;

    if (globalStepCounter % GLOBAL_UPDATE_INTERVAL === 0) {
      this.globalUpdate();
    }
  }

  /**
   * Launch parallel workers for async rollout collection (A3C style)
   */
  async collectRollouts() {
    const actionName = 'Launch A3C parallel rollout workers';
    if (!await mercyGate(actionName)) return;

    this.workers.forEach(worker => {
      worker.postMessage({
        command: 'rollout',
        initialState: this.mcts.root.state,
        initialActions: this.mcts.root.untriedActions
      });
    });
  }

  /**
   * Global PPO update on accumulated buffer
   */
  async globalUpdate() {
    if (globalReplayBuffer.length < BATCH_SIZE) return;

    const batch = this.sampleBatch(BATCH_SIZE);

    // Compute GAE advantages & returns
    const { advantages, returns } = this.computeAdvantagesAndReturns(batch);

    // Valence-weighted advantage normalization + boost
    const weightedAdvantages = advantages.map((adv, i) => {
      return adv + VALENCE_ADVANTAGE_BOOST * batch[i].valence;
    });

    // PPO training step
    const stats = await this.policyNet.trainPPO(batch, weightedAdvantages, returns);

    mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
    console.log("[PPO-A3C-MCTS] Global update stats:", stats);
  }

  private sampleBatch(size: number): TrajectoryStep[] {
    const indices = new Set<number>();
    while (indices.size < size && indices.size < globalReplayBuffer.length) {
      indices.add(Math.floor(Math.random() * globalReplayBuffer.length));
    }
    return Array.from(indices).map(i => globalReplayBuffer[i]);
  }

  private computeAdvantagesAndReturns(trajectory: TrajectoryStep[]): { advantages: number[]; returns: number[] } {
    const advantages: number[] = new Array(trajectory.length);
    const returns: number[] = new Array(trajectory.length);

    let nextValue = 0;
    let nextAdvantage = 0;

    for (let t = trajectory.length - 1; t >= 0; t--) {
      const step = trajectory[t];
      const delta = step.reward + A3C_GAMMA * nextValue * (step.done ? 0 : 1) - step.value;
      nextAdvantage = delta + A3C_GAMMA * A3C_LAMBDA * (step.done ? 0 : 1) * nextAdvantage;

      advantages[t] = nextAdvantage;
      returns[t] = advantages[t] + step.value;

      nextValue = step.value;
    }

    return { advantages, returns };
  }

  /**
   * Full asynchronous training loop
   */
  async runTrainingLoop() {
    const actionName = 'Run PPO-guided A3C-MCTS training loop';
    if (!await mercyGate(actionName)) return;

    while (true) {
      await this.collectRollouts();
      await new Promise(resolve => setTimeout(resolve, 1000)); // simulate async collection
    }
  }
}

export default PPOA3CMCTSHybrid;
