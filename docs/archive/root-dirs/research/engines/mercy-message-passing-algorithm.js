```js
// mercy-message-passing-algorithm.js
// Hierarchical Message Passing for Predictive Coding & Active Inference
// Top-down predictions + bottom-up precision-weighted errors + mercy gating
// MIT License + AG-SML v1.0 – Autonomicity Games Inc. 2026

import { mercyPrecisionWeighting } from './mercy-precision-weighting-algorithm.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyMessagePassing {
  constructor() {
    this.layers = new Map(); // level -> { prediction, precision, error }
  }

  /**
   * Send top-down prediction to a lower layer
   * @param {number} level - Hierarchical level (higher = more abstract)
   * @param {number} prediction - Predicted value/mean
   * @param {number} precision - Confidence in this prediction
   * @param {number} currentValence
   * @param {object} context
   */
  sendTopDownPrediction(level, prediction, precision, currentValence = 1.0, context = {}) {
    if (!this.layers.has(level)) this.layers.set(level, {});

    const layer = this.layers.get(level);
    layer.prediction = prediction;
    layer.precision = precision;

    // Mercy gate check before propagating
    if (currentValence < MERCY_THRESHOLD) {
      console.log(`[MercyMessagePassing] Gate holds at level ${level} – top-down prediction suppressed`);
      layer.precision = 0.0;
    }

    return layer;
  }

  /**
   * Receive bottom-up prediction error from a lower layer
   * @param {number} level
   * @param {number} observation
   * @param {number} currentValence
   * @param {object} context
   * @returns {number} weightedError
   */
  receiveBottomUpError(level, observation, currentValence = 1.0, context = {}) {
    const layer = this.layers.get(level) || { prediction: 0, precision: 1.0 };
    const rawError = observation - layer.prediction;

    // Apply precision weighting (integrated with mercy-precision-weighting-algorithm.js)
    const weightedError = mercyPrecisionWeighting.applyWeightedUpdate(
      rawError,
      rawError,
      currentValence,
      { ...context, level }
    );

    // Store error for higher-level update
    layer.error = weightedError;

    return weightedError;
  }

  /**
   * Propagate error upward through the hierarchy (one holistic pass)
   * @param {number} currentValence
   * @param {object} context
   * @returns {object} finalStateUpdate
   */
  propagateUpward(currentValence = 1.0, context = {}) {
    let highestLevelUpdate = 0;

    // Process layers from lowest to highest
    for (let level = Math.max(...this.layers.keys()); level >= 0; level--) {
      const layer = this.layers.get(level);
      if (!layer || layer.error === undefined) continue;

      // Precision-weighted update to higher level
      const update = layer.error * layer.precision;
      highestLevelUpdate += update;

      // Clear processed error
      layer.error = undefined;
    }

    return {
      highestLevelUpdate,
      status: currentValence >= MERCY_THRESHOLD ? 'Message passing complete – mercy gates passed' : 'Message passing aborted – low valence'
    };
  }

  resetLayers() {
    this.layers.clear();
  }

  getLayerState(level) {
    return this.layers.get(level) || null;
  }
}

const mercyMessagePassing = new MercyMessagePassing();

export { mercyMessagePassing, MercyMessagePassing };
