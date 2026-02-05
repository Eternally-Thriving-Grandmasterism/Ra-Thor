// mercy-flow-state-engine.js – sovereign Mercy Flow State Engine v1
// Real-time flow monitoring, adaptive challenge, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyFlowStateEngine {
  constructor() {
    this.flowScore = 0.0;           // 0–1.0 flow state estimate
    this.lastActionTime = Date.now();
    this.actionCount = 0;
    this.valence = 1.0;
    this.challengeLevel = 1.0;      // 0.5 = easy, 1.0 = balanced, 1.5 = hard
  }

  async gateFlow(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyFlow] Gate holds: low valence – flow state adjustment aborted");
      return false;
    }
    this.valence = valence;
    return true;
  }

  // Update flow score after user action
  registerAction(success = true, durationMs = 0) {
    this.actionCount++;
    const timeSinceLast = (Date.now() - this.lastActionTime) / 1000;

    // Flow indicators
    const speedBonus = durationMs < 1500 ? 0.15 : 0; // fast action = flow
    const rhythmBonus = timeSinceLast > 0.5 && timeSinceLast < 5 ? 0.12 : 0; // good rhythm
    const successBonus = success ? 0.25 : -0.1;

    this.flowScore = Math.min(1.0, Math.max(0.0, this.flowScore + speedBonus + rhythmBonus + successBonus - 0.02)); // natural decay

    this.lastActionTime = Date.now();

    // Adaptive challenge
    if (this.flowScore > 0.85) {
      this.challengeLevel = Math.min(1.5, this.challengeLevel + 0.05); // increase challenge
    } else if (this.flowScore < 0.4) {
      this.challengeLevel = Math.max(0.5, this.challengeLevel - 0.08); // reduce challenge
    }

    console.log(`[MercyFlow] Flow score: ${(this.flowScore * 100).toFixed(1)}% | Challenge level: ${this.challengeLevel.toFixed(2)}`);
  }

  // Apply flow state to UI/feedback
  getFlowModifiers() {
    return {
      hapticIntensity: 0.6 + this.flowScore * 0.4,
      visualGlow: 0.7 + this.flowScore * 0.6,
      voicePitch: 0.9 + this.flowScore * 0.4,
      challengeAdjustment: this.challengeLevel
    };
  }

  // Example: adjust probe sim difficulty based on flow
  adjustProbeSimulationDifficulty(baseDifficulty) {
    return baseDifficulty * this.challengeLevel;
  }
}

const mercyFlow = new MercyFlowStateEngine();

// Hook into user actions (call after every significant interaction)
function onMercyAction(success = true, durationMs = 0) {
  mercyFlow.registerAction(success, durationMs);
}

// Example usage in gesture handler:
onMercyAction(true, 1200); // successful gesture took 1.2 seconds

export { mercyFlow, onMercyAction };
