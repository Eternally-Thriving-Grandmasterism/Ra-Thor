// src/core/ppo-surrogate-loss.ts – PPO Clipped Surrogate Objective + Full Loss v1.0
// Valence-weighted advantage normalization, entropy bonus, value loss, mercy gates
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const PPO_CLIP_EPSILON = 0.2;
const VALUE_LOSS_COEF = 0.5;
const ENTROPY_COEF = 0.01;
const VALENCE_ADVANTAGE_BOOST = 2.5;
const MERCY_THRESHOLD = 0.9999999;

/**
 * Compute PPO clipped surrogate loss + value loss + entropy bonus
 * @param oldLogProbs Old policy log-probabilities (collected during rollout)
 * @param newLogProbs New policy log-probabilities (current policy)
 * @param advantages Generalized Advantage Estimation (GAE) values
 * @param returns Discounted returns for value target
 * @param values Current value estimates
 * @returns PPO loss components
 */
export function computePPOLoss(
  oldLogProbs: number[],
  newLogProbs: number[],
  advantages: number[],
  returns: number[],
  values: number[]
): {
  policyLoss: number;
  valueLoss: number;
  entropyBonus: number;
  totalLoss: number;
  approxKL: number;
  clipFraction: number;
} {
  if (!mercyGate('Compute PPO surrogate loss')) {
    return {
      policyLoss: 0,
      valueLoss: 0,
      entropyBonus: 0,
      totalLoss: 0,
      approxKL: 0,
      clipFraction: 0
    };
  }

  const n = oldLogProbs.length;
  if (n !== newLogProbs.length || n !== advantages.length) {
    throw new Error("Input arrays must have the same length");
  }

  const valence = currentValence.get();

  // ─── 1. PPO clipped surrogate objective ───────────────────────
  let policyLossSum = 0;
  let clipCount = 0;
  let klSum = 0;

  for (let i = 0; i < n; i++) {
    const ratio = Math.exp(newLogProbs[i] - oldLogProbs[i]);

    // Valence-weighted advantage boost
    const weightedAdv = advantages[i] * (1 + VALENCE_ADVANTAGE_BOOST * valence);

    const surrogate1 = ratio * weightedAdv;
    const surrogate2 = Math.max(1 - PPO_CLIP_EPSILON, Math.min(1 + PPO_CLIP_EPSILON, ratio)) * weightedAdv;

    policyLossSum += Math.min(surrogate1, surrogate2);

    // Clip fraction for monitoring
    if (ratio < 1 - PPO_CLIP_EPSILON || ratio > 1 + PPO_CLIP_EPSILON) {
      clipCount++;
    }

    // Approximate KL divergence (for early stopping / adaptive KL)
    klSum += ratio - 1 - Math.log(ratio + 1e-10);
  }

  const policyLoss = -policyLossSum / n; // negative because we maximize
  const clipFraction = clipCount / n;
  const approxKL = klSum / n;

  // ─── 2. Value function loss (clipped) ──────────────────────────
  let valueLossSum = 0;
  for (let i = 0; i < n; i++) {
    const vPred = values[i];
    const vTarget = returns[i];
    const diff = vPred - vTarget;
    valueLossSum += diff * diff;
  }
  const valueLoss = valueLossSum / n * VALUE_LOSS_COEF;

  // ─── 3. Entropy bonus (encourage exploration) ──────────────────
  let entropySum = 0;
  for (let i = 0; i < n; i++) {
    // Approximate entropy from log-prob (for single action – real impl needs full dist)
    entropySum += -oldLogProbs[i]; // rough proxy
  }
  const entropyBonus = (entropySum / n) * ENTROPY_COEF;

  // ─── 4. Total PPO loss ─────────────────────────────────────────
  const totalLoss = policyLoss + valueLoss - entropyBonus;

  return {
    policyLoss,
    valueLoss,
    entropyBonus,
    totalLoss,
    approxKL,
    clipFraction
  };
}

/**
 * Valence-weighted advantage normalization
 * Boosts advantage for high-valence timesteps
 */
export function normalizeAdvantagesWithValence(
  advantages: number[],
  valences: number[]
): number[] {
  if (advantages.length !== valences.length) {
    throw new Error("Advantages and valences must have same length");
  }

  const n = advantages.length;
  const meanAdv = advantages.reduce((a, b) => a + b, 0) / n;
  const stdAdv = Math.sqrt(advantages.reduce((a, b) => a + Math.pow(b - meanAdv, 2), 0) / n) + 1e-8;

  return advantages.map((adv, i) => {
    const normAdv = (adv - meanAdv) / stdAdv;
    return normAdv + VALENCE_ADVANTAGE_BOOST * valences[i];
  });
}

/**
 * Example usage in training loop
 */
export async function ppoTrainingStep(
  trajectory: { oldLogProb: number; newLogProb: number; advantage: number; return: number; value: number; valence: number }[],
  onUpdate?: (stats: ReturnType<typeof computePPOLoss>) => void
) {
  const actionName = 'PPO training step with valence-weighted loss';
  if (!await mercyGate(actionName)) return;

  const advantages = trajectory.map(t => t.advantage);
  const valences = trajectory.map(t => t.valence);
  const weightedAdvantages = normalizeAdvantagesWithValence(advantages, valences);

  const oldLogProbs = trajectory.map(t => t.oldLogProb);
  const newLogProbs = trajectory.map(t => t.newLogProb);
  const returns = trajectory.map(t => t.return);
  const values = trajectory.map(t => t.value);

  const stats = computePPOLoss(oldLogProbs, newLogProbs, weightedAdvantages, returns, values);

  // In real training: backprop stats.totalLoss through policy & value heads

  onUpdate?.(stats);
  return stats;
}
