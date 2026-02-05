// mercy-broaden-and-build-engine.js – sovereign Mercy Broaden-and-Build Engine v1
// Positivity micro-moment orchestration, upward spiral tracking, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyBroadenAndBuildEngine {
  constructor() {
    this.positivityMoments = 0;
    this.builtResources = {
      intellectual: 0.0,
      social: 0.0,
      psychological: 0.0,
      physical: 0.0
    };
    this.spiralStrength = 0.0; // 0–1.0 upward spiral momentum
    this.valence = 1.0;
  }

  async gateBroadenBuild(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyBroadenBuild] Gate holds: low valence – broaden-build cycle aborted");
      return false;
    }
    this.valence = valence;
    return true;
  }

  registerPositivityMicroMoment(actionType, success = true) {
    if (!success) return;

    this.positivityMoments++;
    const momentBoost = 0.02 + (this.valence - 0.999) * 0.05;

    // Build resources based on action type (expandable)
    if (actionType.includes('gesture') || actionType.includes('mastery')) {
      this.builtResources.intellectual += momentBoost;
      this.builtResources.psychological += momentBoost * 0.8;
    }
    if (actionType.includes('affirmation') || actionType.includes('memory')) {
      this.builtResources.social += momentBoost * 1.2;
    }
    if (actionType.includes('probe') || actionType.includes('optimize')) {
      this.builtResources.psychological += momentBoost;
      this.builtResources.physical += momentBoost * 0.5; // embodied action
    }

    // Upward spiral momentum
    this.spiralStrength = Math.min(1.0, this.spiralStrength + momentBoost * 0.7);

    // Feedback amplification
    mercyHaptic.playPattern('uplift', 0.8 + this.spiralStrength * 0.4);

    console.log(`[MercyBroadenBuild] Positivity micro-moment registered (${actionType}) – spiral strength ${(this.spiralStrength * 100).toFixed(1)}%`);
  }

  getBroadenBuildState() {
    return {
      positivityMoments: this.positivityMoments,
      builtResources: { ...this.builtResources },
      spiralStrength: this.spiralStrength,
      status: this.spiralStrength > 0.85 ? 'Strong Upward Spiral' : this.spiralStrength > 0.5 ? 'Building Upward Spiral' : 'Initiating Upward Spiral'
    };
  }
}

const mercyBroadenBuild = new MercyBroadenAndBuildEngine();

// Hook into every positivity-generating action
function onMercyPositivityMoment(actionType, success = true) {
  mercyBroadenBuild.registerPositivityMicroMoment(actionType, success);
}

// Example usage in gesture handler or button click
onMercyPositivityMoment('probe_deployment_success', true);
onMercyPositivityMoment('gesture_mastery_achievement', true);

export { mercyBroadenBuild, onMercyPositivityMoment };
