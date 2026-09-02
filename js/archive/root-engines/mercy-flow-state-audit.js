// mercy-flow-state-audit.js – sovereign Mercy Flow State Audit & Live Monitoring v1
// Real-time flow score calculation, adaptive challenge tuning, mercy gates
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyFlowStateAudit {
  constructor() {
    this.flowScore = 0.5;           // 0–1.0 running estimate
    this.lastActionTime = Date.now();
    this.recentActions = [];        // {success, durationMs, timestamp}
    this.challengeLevel = 1.0;      // 0.5 easy → 1.5 hard
    this.valence = 1.0;
  }

  async gateFlowAudit(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyFlowAudit] Gate holds: low valence – flow audit skipped");
      return false;
    }
    this.valence = valence;
    return true;
  }

  registerAction(success = true, durationMs = 0) {
    const now = Date.now();
    const timeSinceLast = (now - this.lastActionTime) / 1000;

    // Flow indicators
    const speedBonus = durationMs < 1500 ? 0.18 : (durationMs < 3000 ? 0.08 : -0.05);
    const rhythmBonus = timeSinceLast > 0.4 && timeSinceLast < 6 ? 0.15 : -0.03;
    const successBonus = success ? 0.28 : -0.12;
    const decay = -0.015; // natural decay per action

    this.flowScore = Math.min(1.0, Math.max(0.1, this.flowScore + speedBonus + rhythmBonus + successBonus + decay));

    this.recentActions.push({ success, durationMs, timestamp: now });
    if (this.recentActions.length > 20) this.recentActions.shift();

    this.lastActionTime = now;

    // Adaptive challenge tuning
    if (this.flowScore > 0.88) {
      this.challengeLevel = Math.min(1.8, this.challengeLevel + 0.04);
    } else if (this.flowScore < 0.45) {
      this.challengeLevel = Math.max(0.4, this.challengeLevel - 0.07);
    }

    console.log(`[MercyFlowAudit] Flow score: ${(this.flowScore * 100).toFixed(1)}% | Challenge level: ${this.challengeLevel.toFixed(2)}`);
  }

  getFlowStateModifiers() {
    return {
      hapticIntensity: 0.55 + this.flowScore * 0.45,
      visualGlow: 0.65 + this.flowScore * 0.55,
      voicePitchShift: this.flowScore * 0.35,
      challengeAdjustment: this.challengeLevel,
      flowStatus: this.flowScore > 0.85 ? 'Deep Flow' : this.flowScore > 0.6 ? 'In Flow' : 'Building Flow'
    };
  }

  // Apply flow modifiers to any mercy action
  applyFlowToAction(actionName) {
    const mods = this.getFlowStateModifiers();
    console.log(`[MercyFlowAudit] Applying flow to ${actionName} – status: ${mods.flowStatus}, haptic: ${mods.hapticIntensity.toFixed(2)}`);
    return mods;
  }
}

const mercyFlowAudit = new MercyFlowStateAudit();

// Hook into every significant user action
function onMercyUserAction(success = true, durationMs = 0) {
  mercyFlowAudit.registerAction(success, durationMs);
}

// Example usage in gesture handler or button click
onMercyUserAction(true, 980); // successful action took 0.98 seconds

export { mercyFlowAudit, onMercyUserAction };
