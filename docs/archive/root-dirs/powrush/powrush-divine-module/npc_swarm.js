import { MercyCore, ValenceEngine } from './ra-thor-mercy-core.js';
import { HyperonReasoner } from './hyperon-integration.js';
import { NEATGenome } from './neat-evolution.js';

class PowrushNPCAgent {
  constructor(faction, role, genomeSeed) {
    this.faction = faction; // 'Draek', 'Quellorian', 'Cydruid', etc.
    this.role = role; // 'harvester', 'shield-captain', 'fissure-guardian'
    this.mercy = new MercyCore();
    this.valence = new ValenceEngine();
    this.reasoner = new HyperonReasoner();
    this.genome = new NEATGenome(genomeSeed || Math.random());
    this.memory = []; // IndexedDB persistent
  }

  async decideAction(worldState, playerActions) {
    // Pre-valence gate
    let candidates = await this.reasoner.generateActions(worldState, this.memory);
    let best = null;
    let highestValence = -Infinity;

    for (let action of candidates) {
      let projected = await this.valence.simulateOutcome(action, worldState);
      if (projected.valenceScore >= 0.99 && projected.valenceScore > highestValence) {
        best = action;
        highestValence = projected.valenceScore;
      }
    }

    // If no mercy-pass, force mutation toward thriving
    if (!best) {
      best = await this.genome.evolveTowardThriving(worldState);
    }

    this.memory.push({ action: best, outcome: null });
    return best; // e.g., { type: 'harvest_teleport', target: playerId } or { type: 'offer_alliance', terms: 'share_tech' }
  }
}

// Swarm orchestration (run locally per client or P2P synced)
async function runNPCSwarm(worldSlice) {
  for (let agent of worldSlice.npcs) {
    let action = await agent.decideAction(worldSlice, worldSlice.players);
    await applyAction(action, worldSlice); // Mercy re-check post-apply
  }
}
