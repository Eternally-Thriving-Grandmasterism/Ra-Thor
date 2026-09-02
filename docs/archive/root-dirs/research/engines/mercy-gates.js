// mercy-gates.js
// Central implementation of the 7 Living Mercy Gates
// Hard mathematical constraints enforced at every layer of Ra-Thor
// MIT License + AG-SML v1.0 – Autonomicity Games Inc. 2026

const MERCY_THRESHOLD = 0.9999999;

class MercyGates {
  /**
   * Check if a trajectory/action passes ALL 7 Living Mercy Gates
   * @param {number} currentValence - System valence [0,1]
   * @param {object} context - Optional context (action type, expected deltas, etc.)
   * @returns {boolean} true if all gates pass
   */
  static allPass(currentValence, context = {}) {
    return this.gate1RadicalLove(currentValence, context) &&
           this.gate2BoundlessMercy(currentValence, context) &&
           this.gate3Service(currentValence, context) &&
           this.gate4Abundance(currentValence, context) &&
           this.gate5Truth(currentValence, context) &&
           this.gate6Joy(currentValence, context) &&
           this.gate7CosmicHarmony(currentValence, context);
  }

  static gate1RadicalLove(v, context) {
    return v >= MERCY_THRESHOLD;
  }

  static gate2BoundlessMercy(v, context) {
    const sufferingDelta = context.expectedSufferingDelta || 0;
    return v >= MERCY_THRESHOLD && sufferingDelta <= 0;
  }

  static gate3Service(v, context) {
    const serviceDelta = context.expectedServiceDelta || 0;
    return v >= MERCY_THRESHOLD && serviceDelta > 0;
  }

  static gate4Abundance(v, context) {
    const abundanceDelta = context.expectedAbundanceDelta || 0;
    return v >= MERCY_THRESHOLD && abundanceDelta > 0;
  }

  static gate5Truth(v, context) {
    const deceptionScore = context.deceptionScore || 0;
    return v >= MERCY_THRESHOLD && deceptionScore <= 0;
  }

  static gate6Joy(v, context) {
    const joyDelta = context.expectedJoyDelta || 0;
    return v >= MERCY_THRESHOLD && joyDelta > 0;
  }

  static gate7CosmicHarmony(v, context) {
    const harmonyDelta = context.expectedHarmonyDelta || 0;
    return v >= MERCY_THRESHOLD && harmonyDelta > 0;
  }

  /**
   * Enforce gates and return result with detailed logging
   */
  static enforce(currentValence, context = {}) {
    const passed = this.allPass(currentValence, context);
    if (!passed) {
      console.log(`[MercyGates] VIOLATION — valence=${currentValence.toFixed(8)}, context=`, context);
    }
    return {
      passed,
      valence: currentValence,
      threshold: MERCY_THRESHOLD,
      timestamp: Date.now()
    };
  }
}

export { MercyGates, MERCY_THRESHOLD };
