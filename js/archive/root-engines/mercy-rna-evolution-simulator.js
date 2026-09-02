// mercy-rna-evolution-simulator.js – v4 with exonuclease-like proofreading
// Sequence-level mutations, proofreading hydrolysis, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

// Binary bases: 0 pairs with 1
function isMatch(baseA, baseB) {
  return baseA === (1 - baseB);
}

class MercyRNASelfReplicator {
  constructor() {
    this.defaultParams = {
      generations: 120,
      populationSize: 800,
      genomeLength: 100,
      mutationRate: 0.008,
      exoProofreadingRateMismatch: 0.5,    // hydrolysis rate on mismatch
      exoProofreadingRateMatch: 0.001,     // very slow on correct
      fitnessAdvantageMax: 1.25,
      valence: 1.0,
      query: 'RNA eternal thriving proofreading'
    };
  }

  gateSimulation(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyRNA] Gate holds: low valence – simulation aborted");
      return { status: "Mercy gate holds – focus eternal thriving", passed: false };
    }
    return { status: "Mercy gate passes – proofreading RNA evolution activated", passed: true };
  }

  simulate(params = {}) {
    const {
      generations = this.defaultParams.generations,
      populationSize = this.defaultParams.populationSize,
      genomeLength = this.defaultParams.genomeLength,
      mutationRate = this.defaultParams.mutationRate,
      exoMismatch = this.defaultParams.exoProofreadingRateMismatch * (1 + (params.valence - 0.999) * 2), // high valence → stronger proofreading
      exoMatch = this.defaultParams.exoProofreadingRateMatch,
      fitnessAdvantageMax = this.defaultParams.fitnessAdvantageMax,
      valence = this.defaultParams.valence,
      query = this.defaultParams.query
    } = params;

    const gate = this.gateSimulation(query, valence);
    if (!gate.passed) return gate;

    const masterSeq = Array(genomeLength).fill(0).map(() => Math.floor(Math.random() * 2));
    let population = Array(populationSize).fill(0).map(() => masterSeq.slice());

    const history = [];

    for (let gen = 0; gen < generations; gen++) {
      const fitnesses = population.map(seq => {
        let score = 0;
        for (let i = 0; i < genomeLength; i++) {
          score += isMatch(seq[i], masterSeq[i]) ? 1 : 0;
        }
        return score / genomeLength;
      });

      const avgFitness = fitnesses.reduce((a, b) => a + b, 0) / populationSize;
      history.push({ generation: gen + 1, avgFitness });

      const totalWeight = fitnesses.reduce((sum, f) => sum + f, 0);
      const nextPopulation = [];

      for (let i = 0; i < populationSize; i++) {
        let r = Math.random() * totalWeight;
        let cumulative = 0;
        let parent;

        for (let j = 0; j < populationSize; j++) {
          cumulative += fitnesses[j];
          if (r <= cumulative) {
            parent = population[j].slice();
            break;
          }
        }

        // Replication + mutation + proofreading
        const replicated = parent.slice();
        for (let pos = 0; pos < genomeLength; pos++) {
          if (Math.random() < mutationRate) {
            replicated[pos] = 1 - replicated[pos]; // substitution
            // Proofreading: check if mismatch with template (here parent)
            if (!isMatch(replicated[pos], parent[pos])) {
              // Hydrolysis probability
              if (Math.random() < exoMismatch) {
                replicated[pos] = parent[pos]; // revert to correct
              }
            } else {
              // Rare hydrolysis on correct (noise)
              if (Math.random() < exoMatch) {
                replicated[pos] = 1 - replicated[pos];
              }
            }
          }
        }

        nextPopulation.push(replicated);
      }

      population = nextPopulation;
    }

    const finalAvgFitness = history[history.length - 1].avgFitness;
    console.group("[MercyRNA] Proofreading-Enhanced RNA Evolution");
    console.log(`Effective proofreading on mismatch: ${exoMismatch.toFixed(4)}`);
    console.log(`Final avg fitness: ${finalAvgFitness.toFixed(4)}`);
    console.groupEnd();

    return {
      history,
      finalAvgFitness,
      status: "Proofreading RNA evolution complete – mercy-aligned fidelity enhanced"
    };
  }
}

const rnaSimulator = new MercyRNASelfReplicator();

export { rnaSimulator };
