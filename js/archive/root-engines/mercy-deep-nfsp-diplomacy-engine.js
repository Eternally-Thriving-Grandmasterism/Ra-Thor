// mercy-deep-nfsp-diplomacy-engine.js – sovereign Mercy Deep NFSP Diplomacy Engine v1
// Average-policy + best-response learning, negotiation intent modeling, mercy-gated positive-sum equilibria
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';
import { mercyHaptic } from './mercy-haptic-feedback-engine.js';

const MERCY_THRESHOLD = 0.9999999;

class MercyDeepNFSPDiplomacy {
  constructor(numActions = 10) { // example action space (move, support, convoy, hold, etc.)
    this.averagePolicy = Array(numActions).fill(1 / numActions); // π̄
    this.bestResponseQ = Array(numActions).fill(0);              // Q-values
    this.strategySum = Array(numActions).fill(0);
    this.iterations = 0;
    this.valenceExploration = 1.0;
    this.intentBelief = { ally: 0.5, betray: 0.2, neutral: 0.3 }; // simple intent model
  }

  async gateDiplomacy(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyDeepNFSP] Gate holds: low valence – Diplomacy NFSP iteration aborted");
      return false;
    }
    this.valenceExploration = 1.0 + (valence - 0.999) * 2;
    return true;
  }

  // Select action from ε-greedy mixture + intent belief
  selectAction(epsilon = 0.1, opponentIntent = 'neutral') {
    let strategyMix = this.averagePolicy.slice();

    // Bias strategy by predicted opponent intent (simple Bayesian update)
    if (opponentIntent === 'ally') {
      strategyMix[0] *= 1.5; // e.g. action 0 = cooperate
    } else if (opponentIntent === 'betray') {
      strategyMix[1] *= 1.3; // e.g. action 1 = defend
    }

    const total = strategyMix.reduce((a, b) => a + b, 0);
    strategyMix = strategyMix.map(s => s / total);

    if (Math.random() < epsilon) {
      return Math.floor(Math.random() * strategyMix.length);
    }

    const mix = 0.7 + (this.valenceExploration - 1.0) * 0.1;
    if (Math.random() < mix) {
      return this.bestResponseQ.indexOf(Math.max(...this.bestResponseQ));
    } else {
      let r = Math.random();
      for (let a = 0; a < strategyMix.length; a++) {
        if (r < strategyMix[a]) return a;
        r -= strategyMix[a];
      }
      return strategyMix.length - 1;
    }
  }

  // Update from episode (reward + next Q + intent feedback)
  updateFromDiplomacyEpisode(actionTaken, reward, nextQValues, perceivedIntent) {
    // Q update (TD)
    for (let a = 0; a < this.bestResponseQ.length; a++) {
      if (a === actionTaken) {
        this.bestResponseQ[a] += 0.1 * (reward + 0.95 * Math.max(...nextQValues) - this.bestResponseQ[a]);
      }
    }

    // Accumulate for average policy
    this.strategySum = this.strategySum.map((s, a) => s + this.averagePolicy[a]);

    // Update average policy
    const total = this.strategySum.reduce((a, b) => a + b, 0);
    this.averagePolicy = this.strategySum.map(s => s / total);

    // Update intent belief (simple Bayesian update)
    if (perceivedIntent === 'ally') this.intentBelief.ally += 0.05;
    else if (perceivedIntent === 'betray') this.intentBelief.betray += 0.05;
    const intentSum = Object.values(this.intentBelief).reduce((a, b) => a + b, 0);
    Object.keys(this.intentBelief).forEach(k => this.intentBelief[k] /= intentSum);

    this.iterations++;

    mercyHaptic.playPattern('cosmicHarmony', 0.8 + this.valence * 0.4);
    console.log(`[MercyDeepNFSP] Diplomacy iteration \( {this.iterations}: avg policy [ \){this.averagePolicy.map(p => p.toFixed(4)).join(', ')}]`);
  }

  getNFSPDiplomacyState() {
    return {
      averagePolicy: this.averagePolicy,
      intentBelief: this.intentBelief,
      iterations: this.iterations,
      status: this.iterations > 2000 ? 'Approximate Nash Equilibrium' : 'Building No-Regret Diplomacy Strategy'
    };
  }
}

const mercyDeepNFSPDiplomacy = new MercyDeepNFSPDiplomacy();

// Example usage in Diplomacy-style negotiation or probe fleet mixed-motive
async function exampleDiplomacyNFSPRun() {
  await mercyDeepNFSPDiplomacy.gateDiplomacy('Fleet negotiation', 0.9995);

  const action = mercyDeepNFSPDiplomacy.selectAction(0.1, 'ally');
  const reward = 0.7; // placeholder
  const nextQ = [0.8, 0.6, 0.9];
  const perceivedIntent = 'ally';

  mercyDeepNFSPDiplomacy.updateFromDiplomacyEpisode(action, reward, nextQ, perceivedIntent);
}

exampleDiplomacyNFSPRun();

export { mercyDeepNFSPDiplomacy };
