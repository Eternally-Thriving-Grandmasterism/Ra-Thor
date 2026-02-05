// mercy-von-neumann-swarm-simulator.js – sovereign von Neumann swarm growth simulator v1
// Exponential + logistic constraint, mercy-gated activation, valence-modulated params
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const mercyThreshold = 0.9999999;

class MercyVonNeumannSwarmSimulator {
  constructor() {
    this.defaultParams = {
      generations: 30,
      replicationFactor: 2,
      initialCount: 1,
      resourceLimit: 1e12, // galaxy-scale cap (e.g., habitable systems)
      growthModel: 'logistic' // 'exponential' or 'logistic'
    };
  }

  gateSimulation(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < mercyThreshold || implyThriving.degree < mercyThreshold) {
      return { status: "Mercy gate holds – simulation skipped without eternal thriving alignment" };
    }
    return { status: "Mercy gate passes – swarm simulation activated" };
  }

  simulate(params = {}) {
    const {
      generations = this.defaultParams.generations,
      replicationFactor = this.defaultParams.replicationFactor,
      initialCount = this.defaultParams.initialCount,
      resourceLimit = this.defaultParams.resourceLimit,
      growthModel = this.defaultParams.growthModel
    } = params;

    const counts = [initialCount];
    for (let i = 1; i <= generations; i++) {
      let next = counts[counts.length - 1] * replicationFactor;
      if (growthModel === 'logistic') {
        // Logistic growth cap (carrying capacity)
        next = resourceLimit / (1 + (resourceLimit / initialCount - 1) * Math.exp(-replicationFactor * i));
      }
      counts.push(next);
    }

    // Console table output
    console.group("[MercySwarm] Von Neumann Growth Simulation");
    console.table(counts.map((c, g) => ({
      Generation: g,
      Probes: c.toLocaleString('en-US', { maximumFractionDigits: 0 })
    })));
    console.groupEnd();

    return {
      generations: counts,
      finalCount: counts[generations],
      saturated: counts[generations] >= resourceLimit * 0.99,
      status: "Simulation complete – mercy abundance propagated"
    };
  }

  // Valence-modulated param tweak (high valence → faster replication)
  modulateParams(valence) {
    return {
      replicationFactor: 2 + (valence - 0.999) * 10, // slight boost for high thriving
      resourceLimit: 1e12 * (1 + (valence - 0.999) * 100) // expanded carrying capacity
    };
  }
}

const swarmSimulator = new MercyVonNeumannSwarmSimulator();

export { swarmSimulator };
