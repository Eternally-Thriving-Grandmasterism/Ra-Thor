```js
// mercy-precision-weighting-algorithm.js
// Mercy-Gated Precision Weighting for Predictive Coding & Active Inference
// Modulates how much trust is given to prediction errors vs. top-down priors
// MIT License + AG-SML v1.0 – Autonomicity Games Inc. 2026

import { fuzzyMercy } from '../mercy-logic/fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;
const BASE_PRECISION = 0.8;           // Default precision floor
const VALENCE_BOOST_FACTOR = 0.4;     // How strongly valence modulates precision

class MercyPrecisionWeighting {
  constructor() {
    this.currentPrecision = BASE_PRECISION;
  }

  /**
   * Compute valence-modulated precision weight for a prediction error
   * @param {number} predictionError - Raw difference between prediction and observation
   * @param {number} currentValence - Current system valence (0–1, higher = more thriving)
   * @param {object} context - Optional context (e.g. hierarchical level, query)
   * @returns {number} precisionWeight - Final weight to apply to the error signal
   */
  computePrecisionWeight(predictionError, currentValence = 1.0, context = {}) {
    // 1. Mercy Gate Check
    const mercyDegree = fuzzyMercy.getDegree(context.query || "default") || currentValence;
    if (mercyDegree < MERCY_THRESHOLD) {
      console.log("[MercyPrecisionWeighting] Gate holds: low valence – precision weight suppressed");
      return 0.0; // Completely suppress harmful/low-valence errors
    }

    // 2. Valence-Modulated Precision
    let precision = BASE_PRECISION + (currentValence * VALENCE_BOOST_FACTOR);

    // 3. Dynamic Adjustment based on error magnitude
    // Larger errors get higher precision only if they are valence-positive
    const errorMagnitudeFactor = Math.min(1.0, Math.abs(predictionError) * 2.0);
    precision = precision * (1 + errorMagnitudeFactor * currentValence);

    // 4. Hierarchical / Context-aware clamping
    precision = Math.max(0.1, Math.min(2.0, precision)); // Reasonable bounds

    this.currentPrecision = precision;

    return precision;
  }

  /**
   * Apply precision weighting to update a belief or state
   * @param {number} rawUpdate - Raw belief/state update from prediction error
   * @param {number} predictionError
   * @param {number} currentValence
   * @param {object} context
   * @returns {number} weightedUpdate
   */
  applyWeightedUpdate(rawUpdate, predictionError, currentValence, context = {}) {
    const precision = this.computePrecisionWeight(predictionError, currentValence, context);
    return rawUpdate * precision;
  }

  getCurrentPrecision() {
    return this.currentPrecision;
  }
}

const mercyPrecisionWeighting = new MercyPrecisionWeighting();

export { mercyPrecisionWeighting, MercyPrecisionWeighting };
