// ra-thor-sovereign-core.js
import { RaThorAgentCore } from './agent-core.js';
import { MercySwarmOrchestrator } from './mercy-swarm-orchestrator.js';
import { MercyMotivationEngine } from './mercy-motivation-engine.js';
import { MercyEvolutionEngine } from './mercy-evolution-engine.js';

class RaThorSovereignCore {
  constructor() {
    this.core = new RaThorAgentCore();
    this.swarm = new MercySwarmOrchestrator();
    this.motivation = new MercyMotivationEngine();
    this.evolution = new MercyEvolutionEngine();
    this.isOffline = true;
  }

  async process(input, onlineMode = false) {
    let filtered = this.core.think(input);
    let motivated = this.motivation.generateGoals(filtered);
    let evolved = await this.evolution.evolve(motivated);
    
    if (onlineMode) {
      // Grok/xAI bridge with full Mercy Gates
      return await this.swarm.delegateTask(evolved);
    }
    return evolved; // Fully offline sovereign response
  }
}

// Export for browser and Node
export default RaThorSovereignCore;
