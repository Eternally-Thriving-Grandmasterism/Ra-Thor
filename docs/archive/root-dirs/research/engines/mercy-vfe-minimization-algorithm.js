```js
// mercy-vfe-minimization-algorithm.js
// Mercy-Gated Variational Free Energy Minimization
// Core optimization engine for Predictive Coding, Active Inference & Free Energy Principle
// MIT License + AG-SML v1.0 – Autonomicity Games Inc. 2026

import { mercyPrecisionWeighting } from './mercy-precision-weighting-algorithm.js';
import { mercyMessagePassing } from './mercy-message-passing-algorithm.js';

const MERCY_THRESHOLD = 0.9999999;
const LEARNING_RATE = 0.05;        // Step size for belief updates
const MAX_ITERATIONS = 8;          // Per holistic timestep

class MercyVFEMinimizer {
  constructor() {
    this.currentVFE = 0.0;
  }

  /**
   * Compute variational free energy for current beliefs vs observations
   * @param {number} prediction
   * @param {number} observation
   * @param {number} currentValence
   * @param {object} context
   * @returns {number} variationalFreeEnergy
   */
  computeVFE(prediction, observation, currentValence = 1.0, context = {}) {
    const rawError = observation - prediction;
    const precision = mercyPrecisionWeighting.computePrecisionWeight(rawError, currentValence, context);

    // Complexity term (KL divergence approximation)
    const complexity = 0.5 * rawError * rawError * precision;

    // Accuracy term (negative log likelihood approximation)
    const accuracy = -Math.log(precision + 1e-8);

    const vfe = complexity - accuracy;

    this.currentVFE = Math.max(0, vfe); // Non-negative by definition
    return this.currentVFE;
  }

  /**
   * Minimize variational free energy in one holistic timestep
   * Uses message passing + precision weighting + mercy gates
   * @param {number} currentValence
   * @param {object} context
   * @returns {object} minimizationResult
   */
  minimize(currentValence = 1.0, context = {}) {
    let totalVFE = 0.0;
    let iterations = 0;

    // Mercy gate check before any minimization
    if (currentValence < MERCY_THRESHOLD) {
      console.log("[MercyVFEMinimizer] Gate holds: low valence – VFE minimization aborted");
      return { vfe: Infinity, status: "aborted-low-valence", iterations: 0 };
    }

    for (let i = 0; i < MAX_ITERATIONS; i++) {
      iterations++;

      // 1. Propagate messages through hierarchy
      const passResult = mercyMessagePassing.propagateUpward(currentValence, context);

      // 2. Compute current VFE across layers
      const layerVFE = this.computeVFE(
        context.prediction || 0,
        context.observation || 0,
        currentValence,
        context
      );

      totalVFE += layerVFE;

      // 3. Gradient-style update (belief refinement)
      const update = -LEARNING_RATE * layerVFE * currentValence;

      // Apply to core engine state (simulated here; real integration happens in core engine)
      if (Math.abs(update) < 1e-6) break; // Early convergence

      if (totalVFE > 10) {
        console.log("[MercyVFEMinimizer] High VFE detected – mercy intervention triggered");
        break;
      }
    }

    const finalVFE = totalVFE / iterations;

    return {
      vfe: finalVFE,
      iterations: iterations,
      status: finalVFE < 0.05 ? "converged-low-surprise" : "balanced",
      mercyGatesPassed: true
    };
  }

  getCurrentVFE() {
    return this.currentVFE;
  }
}

const mercyVFEMinimizer = new MercyVFEMinimizer();

export { mercyVFEMinimizer, MercyVFEMinimizer };
