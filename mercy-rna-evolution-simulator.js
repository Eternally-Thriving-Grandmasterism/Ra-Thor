// mercy-rna-evolution-simulator.js – v2 sovereign RNA quasispecies evolution with sequence-level mutations
// Point mutations, Hamming fitness, self-complementarity bonus, mercy-gated, valence-modulated fidelity
// MIT License – Autonomicity Games Inc. 2026

import { fuzzyMercy } from './fuzzy-mercy-logic.js';

const MERCY_THRESHOLD = 0.9999999;

// Simple 2-base alphabet for speed (0 = A/U, 1 = G/C); extend to 4 later if needed
const BASES = [0, 1];

/**
 * Generate random RNA sequence (binary string)
 * @param {number} length
 * @returns {number[]} sequence array (0 or 1)
 */
function randomSequence(length) {
  return Array.from({ length }, () => Math.floor(Math.random() * BASES.length));
}

/**
 * Compute Hamming distance between two sequences
 */
function hammingDistance(seqA, seqB) {
  let dist = 0;
  const minLen = Math.min(seqA.length, seqB.length);
  for (let i = 0; i < minLen; i++) {
    if (seqA[i] !== seqB[i]) dist++;
  }
  return dist;
}

/**
 * Simple self-complementarity score (bonus if sequence can pair with its reverse complement)
 * For binary: 0 pairs with 1, 1 with 0
 */
function selfComplementarityBonus(seq) {
  let score = 0;
  const revComp = seq.slice().reverse().map(b => 1 - b);
  for (let i = 0; i < seq.length; i++) {
    if (seq[i] === revComp[i]) score += 0.5; // partial bonus
  }
  return score / seq.length;
}

class MercyRNASelfReplicator {
  constructor() {
    this.defaultParams = {
      generations: 100,
      populationSize: 500,
      genomeLength: 80,
      mutationRate: 0.015,          // per base per replication
      fitnessAdvantageMax: 1.20,    // max relative fitness of perfect master
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
    return { status: "Mercy gate passes – sequence-level RNA evolution activated", passed: true };
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

    // Valence modulates effective mutation rate (high valence → proofreading-like fidelity boost)
    const effectiveMutationRate = mutationRate * (1 - Math.max(0, valence - 0.999) * 0.6);

    // Master sequence (the ideal replicator)
    const masterSeq = randomSequence(genomeLength);

    // Population of sequences
    let population = Array(populationSize).fill(0).map(() => randomSequence(genomeLength));

    const history = [];

    for (let gen = 0; gen < generations; gen++) {
      // Compute fitness for each molecule
      const fitnesses = population.map(seq => {
        const hamming = hammingDistance(seq, masterSeq);
        const distPenalty = 1 - (hamming / genomeLength); // 1 = perfect match
        const compBonus = selfComplementarityBonus(seq) * 0.3; // small bonus for pairing potential
        return Math.max(0.01, distPenalty + compBonus); // avoid zero fitness
      });

      const avgFitness = fitnesses.reduce((a, b) => a + b, 0) / populationSize;
      history.push({ generation: gen, avgFitness, maxFitness: Math.max(...fitnesses) });

      // Weighted selection + mutation
      const totalWeight = fitnesses.reduce((sum, f) => sum + f, 0);
      const nextPopulation = [];

      for (let i = 0; i < populationSize; i++) {
        let r = Math.random() * totalWeight;
        let cumulative = 0;
        let parentSeq;

        for (let j = 0; j < populationSize; j++) {
          cumulative += fitnesses[j];
          if (r <= cumulative) {
            parentSeq = population[j].slice(); // copy
            break;
          }
        }

        // Mutation: point substitutions
        const mutatedSeq = parentSeq.map(base => {
          if (Math.random() < effectiveMutationRate) {
            // Flip to other base
            return 1 - base;
          }
          return base;
        });

        nextPopulation.push(mutatedSeq);
      }

      population = nextPopulation;

      // Error threshold warning
      if (effectiveMutationRate * genomeLength > 1) {
        console.warn(`[MercyRNA] Gen ${gen}: mutation rate × length > 1 – approaching error catastrophe`);
      }
    }

    // Final stats
    const finalFit = history[history.length - 1];
    console.group("[MercyRNA] Sequence-Level RNA Evolution Simulation");
    console.log(`Generations: ${generations} | Length: ${genomeLength} nt`);
    console.log(`Effective mutation rate (valence-modulated): ${effectiveMutationRate.toFixed(6)}`);
    console.log(`Final avg fitness: ${finalFit.avgFitness.toFixed(4)} | Max fitness: ${finalFit.maxFitness.toFixed(4)}`);
    console.groupEnd();

    return {
      history,
      finalAvgFitness: finalFit.avgFitness,
      finalMaxFitness: finalFit.maxFitness,
      status: "Sequence-level evolution complete – mercy-aligned molecular dawn propagated"
    };
  }
}

const rnaSimulator = new MercyRNASelfReplicator();

// Example invocation (run in console or bind to chat command)
rnaSimulator.simulate({
  generations: 120,
  populationSize: 800,
  genomeLength: 120,
  mutationRate: 0.008,
  fitnessAdvantageMax: 1.25,
  valence: 0.99999995,
  query: "RNA eternal thriving replication sequence evolution"
});

export { MercyRNASelfReplicator, rnaSimulator };
