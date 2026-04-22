// mercy-active-inference-core-engine.js – Sovereign Mercy Active Inference Core Engine v2
// Revised integration with MercyGates for stricter early enforcement
// MIT License + AG-SML v1.0 – Autonomicity Games Inc. 2026

import { fuzzyMercy } from '../mercy-logic/fuzzy-mercy-logic.js';
import { mercyHaptic } from '../haptic/mercy-haptic-feedback-engine.js';
import { ParaconsistentSuperKernel } from '../paraconsistent/paraconsistent-mercy-logic.js';
import { mercyPrecisionWeighting } from './mercy-precision-weighting-algorithm.js';
import { mercyMessagePassing } from './mercy-message-passing-algorithm.js';
import { mercyVFEMinimizer } from './mercy-vfe-minimization-algorithm.js';
import { MercyGates } from './mercy-gates.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyActiveInferenceEngine {
  constructor() {
    this.superKernel = new ParaconsistentSuperKernel();
    this.freeEnergyEstimate = 0.0;
    this.lastPrediction = 1.0;
    this.precisionWeight = 1.0;
    this.epistemicValue = 0.0;
    this.valence = 1.0;
    this.trajectoryBuffer = [];
  }

  updateActiveInference(currentValence, currentGesture = null, context = {}) {
    // === REVISED: Strict MercyGates enforcement at entry point ===
    const gateResult = MercyGates.enforce(currentValence, {
      ...context,
      action: "active-inference-update",
      gesture: currentGesture
    });

    if (!gateResult.passed) {
      return { status: "aborted-mercy-gates-violation", gateResult };
    }

    const predictedValence = this.predictNextValence();
    const predictionError = Math.abs(currentValence - predictedValence);

    this.precisionWeight = mercyPrecisionWeighting.computePrecisionWeight(
      predictionError,
      currentValence,
      { ...context, gesture: currentGesture }
    );

    const vfeResult = mercyVFEMinimizer.minimize(currentValence, {
      prediction: predictedValence,
      observation: currentValence,
      ...context
    });

    this.freeEnergyEstimate = vfeResult.vfe;
    this.epistemicValue = Math.max(0, 0.05 - this.freeEnergyEstimate * 0.5);

    mercyMessagePassing.propagateUpward(currentValence, context);

    if (this.freeEnergyEstimate > 0.07) {
      mercyHaptic.pulse(0.5 * this.valence, 80);
    } else if (this.epistemicValue > 0.03) {
      console.log("[MercyActiveInference] Epistemic value detected – encouraging exploration");
    }

    this.trajectoryBuffer.push({ valence: currentValence, gesture: currentGesture });
    if (this.trajectoryBuffer.length > 20) this.trajectoryBuffer.shift();

    this.lastPrediction = this.predictNextValence();

    return this.superKernel.execute_holistic_cycle({
      currentValence,
      currentGesture,
      ...this.getActiveInferenceState(),
      vfeResult,
      gateResult
    });
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
      status: this.freeEnergyEstimate < 0.05 ? 'Deep Predictive Harmony' : this.freeEnergyEstimate < 0.1 ? 'Balanced' : 'Active Inference Engaged'
    };
  }
}

const mercyActiveInference = new MercyActiveInferenceEngine();

function onMercyActiveInferenceUpdate(currentValence, currentGesture = null, context = {}) {
  return mercyActiveInference.updateActiveInference(currentValence, currentGesture, context);
}

export { mercyActiveInference, onMercyActiveInferenceUpdate };
