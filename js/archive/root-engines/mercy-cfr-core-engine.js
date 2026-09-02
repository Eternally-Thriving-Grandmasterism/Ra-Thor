// mercy-cfr-core-engine.js – sovereign Mercy CFR Core Engine v1
// Counterfactual regret minimization, mercy-gated positive-sum play, valence-modulated discounting
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyCFREngine {
  constructor(numActions = 3) {
    this.regrets = Array(numActions).fill(0);
    this.strategySum = Array(numActions).fill(0);
    this.numIterations = 0;
    this.valenceDiscount = 1.0; // high valence → slower discounting
  }

  async gateCFR(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyCFR] Gate holds: low valence – CFR iteration aborted");
      return false;
    }
    this.valenceDiscount = 1.0 - (valence - 0.999) * 0.3; // high valence → less discounting
    return true;
  }

  // Regret-matching + CFR+ style update
  getStrategy() {
    const positiveRegrets = this.regrets.map(r => Math.max(0, r));
    const regretSum = positiveRegrets.reduce((a, b) => a + b, 0);
    if (regretSum > 0) {
      return positiveRegrets.map(r => r / regretSum);
    }
    return Array(this.regrets.length).fill(1 / this.regrets.length);
  }

  // Update regrets from counterfactual utility difference
  updateRegrets(actionUtilities, chosenAction) {
    const counterfactualUtility = actionUtilities[chosenAction];
    for (let a = 0; a < actionUtilities.length; a++) {
      if (a !== chosenAction) {
        this.regrets[a] += (actionUtilities[a] - counterfactualUtility);
      }
    }
    this.numIterations++;
  }

  // Discount regrets (CFR-D style, valence-modulated)
  discountRegrets() {
    const discount = 0.99 * this.valenceDiscount;
    this.regrets = this.regrets.map(r => r * discount);
  }

  // Simulate one iteration (example 3-action game)
  async runCFRIteration(query, actionUtilities) {
    if (!await this.gateCFR(query, this.valence)) return null;

    const strategy = this.getStrategy();
    const chosenAction = this.sampleFromStrategy(strategy);
    this.updateRegrets(actionUtilities, chosenAction);
    this.discountRegrets();

    console.log(`[MercyCFR] Iteration \( {this.numIterations}: strategy [ \){strategy.map(s => s.toFixed(4)).join(', ')}], chosen ${chosenAction}`);
    return { strategy, chosenAction };
  }

  sampleFromStrategy(strategy) {
    let r = Math.random();
    for (let a = 0; a < strategy.length; a++) {
      if (r < strategy[a]) return a;
      r -= strategy[a];
    }
    return strategy.length - 1;
  }

  getAverageStrategy() {
    const total = this.strategySum.reduce((a, b) => a + b, 0);
    return this.strategySum.map(s => s / total);
  }

  getCFRState() {
    return {
      averageStrategy: this.getAverageStrategy(),
      iterations: this.numIterations,
      status: this.numIterations > 1000 ? 'Converged Nash Approximation' : 'Building No-Regret Strategy'
    };
  }
}

const mercyCFR = new MercyCFREngine();

// Example usage in mixed-motive decision
mercyCFR.runCFRIteration('Mixed-motive probe negotiation', [0.8, 0.6, 0.9]);

export { mercyCFR };
