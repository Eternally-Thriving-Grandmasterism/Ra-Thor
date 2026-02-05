// mercy-active-inference-core-engine.js – sovereign Mercy Active Inference Core Engine v1
// Predictive trajectory modeling, epistemic value, precision weighting, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyActiveInferenceEngine {
  constructor() {
    this.freeEnergyEstimate = 0.0;         // proxy for variational free energy
    this.lastPrediction = 1.0;             // last predicted valence
    this.precisionWeight = 1.0;            // attention / confidence in predictions
    this.epistemicValue = 0.0;             // expected information gain
    this.valence = 1.0;
    this.trajectoryBuffer = [];            // last N valence/gesture states
  }

  async gateActiveInference(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyActiveInference] Gate holds: low valence – active inference aborted");
      return false;
    }
    this.valence = valence;
    return true;
  }

  // Update generative model & free energy (call on every state change)
  updateActiveInference(currentValence, currentGesture = null) {
    // Generative model: exponential moving average + trend
    const predictedValence = this.predictNextValence();
    const predictionError = Math.abs(currentValence - predictedValence);

    // Precision weighting: high valence → trust predictions more
    this.precisionWeight = 0.8 + this.valence * 0.4;

    // Weighted free energy (surprise modulated by precision)
    this.freeEnergyEstimate = predictionError / this.precisionWeight;

    // Epistemic value (information gain if we act to reduce uncertainty)
    this.epistemicValue = Math.max(0, 0.05 - this.freeEnergyEstimate * 0.5);

    // Active inference action: if free energy high → trigger epistemic or pragmatic correction
    if (this.freeEnergyEstimate > 0.07) {
      mercyHaptic.pulse(0.5 * this.valence, 80); // surprise reduction pulse
      console.log(`[MercyActiveInference] Free energy high (${this.freeEnergyEstimate.toFixed(4)}) – active inference correction`);
    } else if (this.epistemicValue > 0.03) {
      console.log("[MercyActiveInference] Epistemic value detected – encouraging exploration");
    }

    // Store trajectory
    this.trajectoryBuffer.push({ valence: currentValence, gesture: currentGesture });
    if (this.trajectoryBuffer.length > 20) this.trajectoryBuffer.shift();

    this.lastPrediction = this.predictNextValence();
  }

  predictNextValence() {
    if (this.trajectoryBuffer.length < 3) return this.valence;
    const recent = this.trajectoryBuffer.slice(-5);
    const avgDelta = recent.reduce((sum, s) => sum + (s.valence - this.valence), 0) / recent.length;
    return this.valence + avgDelta;
  }

  getActiveInferenceState() {
    return {
      freeEnergyEstimate: this.freeEnergyEstimate,
      predictivePrecision: this.precisionWeight,
      epistemicValue: this.epistemicValue,
      status: this.freeEnergyEstimate < 0.03 ? 'Low Surprise – Deep Predictive Harmony' : this.freeEnergyEstimate < 0.07 ? 'Balanced Surprise' : 'High Surprise – Active Inference Engaged'
    };
  }
}

const mercyActiveInference = new MercyActiveInferenceEngine();

// Hook into every state change
function onMercyActiveInferenceUpdate(currentValence, currentGesture = null) {
  mercyActiveInference.updateActiveInference(currentValence, currentGesture);
}

// Example usage
onMercyActiveInferenceUpdate(0.9995, 'spiral_outward_clockwise');

export { mercyActiveInference, onMercyActiveInferenceUpdate };
