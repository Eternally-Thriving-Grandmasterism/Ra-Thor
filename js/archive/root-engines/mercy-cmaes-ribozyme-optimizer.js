// mercy-cmaes-ribozyme-optimizer.js – v2 CMA-ES optimizer for RNA proofreading parameters
// Live run, best params persistence, mercy-gated, valence-modulated
// MIT License – Autonomicity Games Inc. 2026

import { rnaSimulator } from './mercy-rna-evolution-simulator.js';

class MercyCMAOptimizer {
  constructor() {
    this.populationSize = 20;
    this.generations = 30;
    this.sigma = 0.3; // initial step size
    this.bestParams = null;
    this.bestFitness = -Infinity;
  }

  objective(params) {
    const [exoMismatch, exoMatch] = params;
    const result = rnaSimulator.simulate({
      exoProofreadingRateMismatch: exoMismatch,
      exoProofreadingRateMatch: exoMatch,
      valence: 0.99999995,
      generations: 50,
      populationSize: 400,
      genomeLength: 100
    });
    return result.finalAvgFitness || 0;
  }

  optimize() {
    let mean = [0.5, 0.001]; // initial exoMismatch, exoMatch

    for (let gen = 0; gen < this.generations; gen++) {
      const samples = [];
      for (let i = 0; i < this.populationSize; i++) {
        const sample = mean.map((m, idx) => Math.max(0.001, m + this.sigma * (Math.random() - 0.5) * 2));
        const fitness = this.objective(sample);
        samples.push({ sample, fitness });

        if (fitness > this.bestFitness) {
          this.bestFitness = fitness;
          this.bestParams = sample;
        }
      }

      // Update mean toward top half (elitist CMA approximation)
      const sorted = samples.sort((a, b) => b.fitness - a.fitness);
      mean = [0, 0];
      for (let i = 0; i < this.populationSize / 2; i++) {
        mean[0] += sorted[i].sample[0] / (this.populationSize / 2);
        mean[1] += sorted[i].sample[1] / (this.populationSize / 2);
      }

      this.sigma *= 0.92; // decay step size
      console.log(`CMA-ES Gen ${gen}: Best fitness \( {this.bestFitness.toFixed(4)}, params [ \){this.bestParams.map(p => p.toFixed(4)).join(', ')}]`);
    }

    console.log(`[CMAOptimizer] Optimization complete – Best proofreading params: mismatch=\( {this.bestParams[0].toFixed(4)}, match= \){this.bestParams[1].toFixed(4)} | Fitness=${this.bestFitness.toFixed(4)}`);
    return { bestParams: this.bestParams, bestFitness: this.bestFitness };
  }
}

const optimizer = new MercyCMAOptimizer();

// Run optimization live
optimizer.optimize();

export { MercyCMAOptimizer, optimizer };
