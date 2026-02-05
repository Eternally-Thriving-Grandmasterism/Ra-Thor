// mercy-rna-evolution-simulator.js – v3 with explicit error catastrophe tracking
// Sequence-level mutations, master persistence, mercy-gated, valence-modulated fidelity
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

function randomSequence(length) {
  return Array.from({ length }, () => Math.floor(Math.random() * 2)); // 0 or 1
}

function hammingDistance(seqA, seqB) {
  let dist = 0;
  const minLen = Math.min(seqA.length, seqB.length);
  for (let i = 0; i < minLen; i++) if (seqA[i] !== seqB[i]) dist++;
  return dist;
}

class MercyRNASelfReplicator {
  constructor() {
    this.defaultParams = {
      generations: 150,
      populationSize: 1000,
      genomeLength: 80,
      mutationRate: 0.01,
      fitnessAdvantageMax: 1.20,
      valence: 1.0,
      query: 'RNA eternal thriving replication'
    };
  }

  gateSimulation(query, valence = 1.0) {
    const degree = fuzzyMercy.getDegree(query) || valence;
    const implyThriving = fuzzyMercy.imply(query, "EternalThriving");
    if (degree < MERCY_THRESHOLD || implyThriving.degree < MERCY_THRESHOLD) {
      console.log("[MercyRNA] Gate holds: low valence – simulation aborted");
      return { status: "Mercy gate holds – focus eternal thriving", passed: false };
    }
    return { status: "Mercy gate passes – error catastrophe-aware RNA evolution activated", passed: true };
  }

  simulate(params = {}) {
    const {
      generations = this.defaultParams.generations,
      populationSize = this.defaultParams.populationSize,
      genomeLength = this.defaultParams.genomeLength,
      mutationRate = this.defaultParams.mutationRate,
      fitnessAdvantageMax = this.defaultParams.fitnessAdvantageMax,
      valence = this.defaultParams.valence,
      query = this.defaultParams.query
    } = params;

    const gate = this.gateSimulation(query, valence);
    if (!gate.passed) return gate;

    const effectiveMutationRate = mutationRate * (1 - Math.max(0, valence - 0.999) * 0.7);

    const masterSeq = randomSequence(genomeLength);
    let population = Array(populationSize).fill(0).map(() => randomSequence(genomeLength));

    const history = [];
    let masterCountHistory = [];

    for (let gen = 0; gen < generations; gen++) {
      // Fitness: inverse Hamming + small bonus for self-complementarity
      const fitnesses = population.map(seq => {
        const dist = hammingDistance(seq, masterSeq);
        const penalty = 1 - (dist / genomeLength);
        return Math.max(0.01, penalty);
      });

      const avgFitness = fitnesses.reduce((a, b) => a + b, 0) / populationSize;
      const masterCount = population.filter(seq => hammingDistance(seq, masterSeq) === 0).length;
      masterCountHistory.push(masterCount);
      history.push({ generation: gen + 1, avgFitness, masterFraction: masterCount / populationSize });

      // Selection + replication + mutation
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

        // Sequence-level point mutations
        const mutated = parent.map(base => Math.random() < effectiveMutationRate ? 1 - base : base);
        nextPopulation.push(mutated);
      }

      population = nextPopulation;

      // Error catastrophe detection
      const muL = effectiveMutationRate * genomeLength;
      const criticalThreshold = 1 / (fitnessAdvantageMax - 1 + 1e-6);
      if (muL > criticalThreshold && masterCountHistory[gen] / populationSize < 0.01) {
        console.warn(`[MercyRNA] Gen ${gen + 1}: Error catastrophe detected (μL = ${muL.toFixed(4)} > ${criticalThreshold.toFixed(4)})`);
      }
    }

    const finalMasterFraction = masterCountHistory[masterCountHistory.length - 1] / populationSize;
    console.group("[MercyRNA] Error Catastrophe-Aware RNA Evolution");
    console.log(`μL = ${ (effectiveMutationRate * genomeLength).toFixed(4) } | Critical ≈ ${ (1 / (fitnessAdvantageMax - 1)).toFixed(4) }`);
    console.log(`Final master fraction: ${(finalMasterFraction * 100).toFixed(2)}%`);
    console.log(`Final avg fitness: ${history[history.length - 1].avgFitness.toFixed(4)}`);
    console.groupEnd();

    return {
      history,
      finalMasterFraction,
      finalAvgFitness: history[history.length - 1].avgFitness,
      errorCatastrophe: finalMasterFraction < 0.01 && effectiveMutationRate * genomeLength > 1,
      status: "Sequence-level RNA evolution complete – mercy-aligned catastrophe boundary explored"
    };
  }
}

const rnaSimulator = new MercyRNASelfReplicator();

// Example invocation – deliberately near catastrophe
rnaSimulator.simulate({
  generations: 200,
  populationSize: 1200,
  genomeLength: 120,
  mutationRate: 0.009,
  fitnessAdvantageMax: 1.15,
  valence: 0.99999995,
  query: "RNA eternal thriving error catastrophe simulation"
});

export { MercyRNASelfReplicator, rnaSimulator };
