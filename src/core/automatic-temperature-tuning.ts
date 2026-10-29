// src/core/automatic-temperature-tuning.ts – Automatic Entropy Temperature Tuning v1.0
// SAC-style auto-α learning with valence-modulated target entropy
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const ALPHA_LR = 3e-4;                      // Learning rate for log-alpha
const ALPHA_INIT = 0.2;                     // Initial temperature
const TARGET_ENTROPY_BASE = -2.0;           // Default SAC target entropy (for action dim)
const VALENCE_ENTROPY_BOOST = 1.5;          // High valence → higher entropy target (more exploration)
const VALENCE_ENTROPY_DAMP = 0.5;           // Low valence → lower entropy target (more exploitation)
const MIN_ALPHA = 0.001;
const MAX_ALPHA = 10.0;
const ENTROPY_UPDATE_INTERVAL = 100;        // Update temperature every N steps

let logAlpha = Math.log(ALPHA_INIT);        // Learnable log-temperature (more stable)
let stepsSinceUpdate = 0;

/**
 * Get current temperature (alpha) – exponentiated log-alpha
 */
export function getTemperature(): number {
  return Math.exp(logAlpha);
}

/**
 * Compute valence-modulated target entropy
 * High valence → encourage exploration (higher entropy target)
 * Low valence → encourage exploitation (lower entropy target)
 */
function getValenceModulatedTargetEntropy(): number {
  const valence = currentValence.get();
  const boost = VALENCE_ENTROPY_BOOST * valence;
  const damp = VALENCE_ENTROPY_DAMP * (1 - valence);
  return TARGET_ENTROPY_BASE + boost - damp;
}

/**
 * Update temperature α toward valence-modulated target entropy
 * @param batchEntropy Average entropy of policy in current batch
 */
export async function updateTemperature(batchEntropy: number): Promise<number> {
  const actionName = 'Update automatic temperature α';
  if (!await mercyGate(actionName)) {
    return getTemperature();
  }

  stepsSinceUpdate++;
  if (stepsSinceUpdate < ENTROPY_UPDATE_INTERVAL) {
    return getTemperature();
  }

  stepsSinceUpdate = 0;

  const targetEntropy = getValenceModulatedTargetEntropy();

  // SAC temperature loss: α * (target_entropy - batch_entropy)
  const alphaLoss = Math.exp(logAlpha) * (targetEntropy - batchEntropy);

  // Gradient descent step on logAlpha
  logAlpha -= ALPHA_LR * alphaLoss;

  // Hard clamp for numerical stability
  logAlpha = Math.max(Math.log(MIN_ALPHA), Math.min(Math.log(MAX_ALPHA), logAlpha));

  const newAlpha = getTemperature();

  // Haptic feedback on significant change
  if (Math.abs(newAlpha - ALPHA_INIT) > 0.1) {
    mercyHaptic.playPattern(
      newAlpha > ALPHA_INIT ? 'cosmicHarmony' : 'warningPulse',
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
 * Compute entropy bonus for policy (used in actor loss)
 * @param logProbs Log-probabilities from current policy
 */
export function entropyBonus(logProbs: number[]): number {
  const avgLogProb = logProbs.reduce((a, b) => a + b, 0) / logProbs.length;
  return -avgLogProb * getTemperature() * PPO_ENTROPY_COEF;
}

/**
 * Example usage in training loop
 */
export async function trainingStepWithAutoTemp(
  batchEntropy: number,
  onUpdate?: (newAlpha: number) => void
): Promise<number> {
  const newAlpha = await updateTemperature(batchEntropy);
  onUpdate?.(newAlpha);
  return newAlpha;
}

export default {
  getTemperature,
  updateTemperature,
  entropyBonus,
  trainingStepWithAutoTemp
};
