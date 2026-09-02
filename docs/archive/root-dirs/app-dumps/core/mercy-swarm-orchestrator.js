// mercy-swarm-orchestrator.js
export class MercySwarmOrchestrator {
  constructor() {
    this.agents = new Map();
  }

  async delegateTask(task) {
    const candidates = Array.from(this.agents.values()).filter(a => a.canHandle(task));
    const best = candidates.sort((a, b) => this.mercyScore(a) - this.mercyScore(b))[0];
    return best.execute(task);
  }

  async consensus(decision) {
    const votes = await Promise.all([...this.agents.values()].map(a => a.vote(decision)));
    return this.apply7FiltersConsensus(votes);
  }
}
