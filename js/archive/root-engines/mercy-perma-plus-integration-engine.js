// mercy-perma-plus-integration-engine.js – sovereign Mercy PERMA+ Integration Engine v1
// Positive Emotion / Engagement / Relationships / Meaning / Accomplishment / Vitality live scoring
// mercy-gated, valence-modulated flourishing feedback
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const PERMA_PLUS_WEIGHTS = {
  positiveEmotion: 0.18,
  engagement: 0.18,
  relationships: 0.18,
  meaning: 0.16,
  accomplishment: 0.16,
  vitality: 0.14
};

class MercyPERMAPlusEngine {
  constructor() {
    this.permaScores = {
      positiveEmotion: 0.0,
      engagement: 0.0,
      relationships: 0.0,
      meaning: 0.0,
      accomplishment: 0.0,
      vitality: 0.0
    };
    this.overallPERMA = 0.0;
    this.valence = 1.0;
    this.actionLog = [];
  }

  async gatePERMAAudit(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < 0.9999999 || implyThriving.degree < 0.9999999) {
      console.log("[MercyPERMA] Gate holds: low valence – PERMA+ audit skipped");
      return false;
    }
    this.valence = valence;
    return true;
  }

  registerPERMAAction(actionType, success = true, durationMs = 0) {
    this.actionLog.push({ actionType, success, durationMs, timestamp: Date.now() });
    if (this.actionLog.length > 100) this.actionLog.shift();

    // Update PERMA+ pillars based on action type (expandable)
    if (actionType.includes('joy') || actionType.includes('sparkle') || actionType.includes('celebration')) {
      this.permaScores.positiveEmotion = Math.min(1.0, this.permaScores.positiveEmotion + (success ? 0.15 : -0.03));
    }
    if (actionType.includes('flow') || actionType.includes('gesture') || actionType.includes('immersion')) {
      this.permaScores.engagement = Math.min(1.0, this.permaScores.engagement + (success ? 0.14 : -0.02));
    }
    if (actionType.includes('affirmation') || actionType.includes('memory') || actionType.includes('gratitude')) {
      this.permaScores.relationships = Math.min(1.0, this.permaScores.relationships + (success ? 0.16 : -0.04));
    }
    if (actionType.includes('purpose') || actionType.includes('accord') || actionType.includes('contribution')) {
      this.permaScores.meaning = Math.min(1.0, this.permaScores.meaning + (success ? 0.14 : -0.03));
    }
    if (actionType.includes('mastery') || actionType.includes('achievement') || actionType.includes('optimize')) {
      this.permaScores.accomplishment = Math.min(1.0, this.permaScores.accomplishment + (success ? 0.15 : -0.03));
    }
    if (actionType.includes('haptic') || actionType.includes('rhythm') || actionType.includes('embodiment')) {
      this.permaScores.vitality = Math.min(1.0, this.permaScores.vitality + (success ? 0.13 : -0.02));
    }

    this.overallPERMA = Object.keys(PERMA_PLUS_WEIGHTS).reduce(
      (sum, key) => sum + this.permaScores[key] * PERMA_PLUS_WEIGHTS[key], 0
    );

    console.log(`[MercyPERMA] Action ${actionType}: overall PERMA+ ${(this.overallPERMA * 100).toFixed(1)}%`);
  }

  getPERMAState() {
    return {
      positiveEmotion: this.permaScores.positiveEmotion,
      engagement: this.permaScores.engagement,
      relationships: this.permaScores.relationships,
      meaning: this.permaScores.meaning,
      accomplishment: this.permaScores.accomplishment,
      vitality: this.permaScores.vitality,
      overall: this.overallPERMA,
      status: this.overallPERMA > 0.85 ? 'Deep PERMA+ Flourishing' : this.overallPERMA > 0.6 ? 'Growing PERMA+ Flourishing' : 'Building PERMA+ Flourishing'
    };
  }
}

const mercyPERMA = new MercyPERMAPlusEngine();

// Hook into every meaningful user action
function onMercyPERMAAction(actionType, success = true, durationMs = 0) {
  mercyPERMA.registerPERMAAction(actionType, success, durationMs);
}

// Example usage in gesture handler or button click
onMercyPERMAAction('probe_deployment_success', true, 2400);
onMercyPERMAAction('gesture_mastery_achievement', true, 1800);

export { mercyPERMA, onMercyPERMAAction };
