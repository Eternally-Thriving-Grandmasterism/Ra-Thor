// evolutionary-strategies-engine.js – sovereign client-side Evolutionary Strategies (ES)
// MIT License – Autonomicity Games Inc. 2026

class ESIndividual {
  constructor(dimension, sigma = 0.1) {
    this.params = new Float64Array(dimension);
    this.sigma = sigma; // mutation step size (self-adaptive)
    for (let i = 0; i < dimension; i++) {
      this.params[i] = Math.random() * 2 - 1; // [-1, 1] init
    }
    this.fitness = 0;
    this.sti = 0.1; // attention from Hyperon
  }

  mutate() {
    const newInd = new ESIndividual(this.params.length, this.sigma);
    for (let i = 0; i < this.params.length; i++) {
      newInd.params[i] = this.params[i] + this.sigma * gaussianRandom();
    }
    // Self-adaptation of sigma (log-normal)
    newInd.sigma = this.sigma * Math.exp(0.1 * gaussianRandom());
    return newInd;
  }

  copy() {
    const copy = new ESIndividual(this.params.length, this.sigma);
    copy.params.set(this.params);
    copy.fitness = this.fitness;
    copy.sti = this.sti;
    return copy;
  }
}

// Simple Gaussian random (Box-Muller transform)
function gaussianRandom() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

class ESEngine {
  constructor(dimension = 10, mu = 10, lambda = 50, generations = 100) {
    this.dimension = dimension;
    this.mu = mu;           // parents
    this.lambda = lambda;   // offspring
    this.generations = generations;
    this.population = [];
    for (let i = 0; i < lambda; i++) {
      this.population.push(new ESIndividual(dimension));
    }
  }

  async evolve(fitnessFunction) {
    for (let gen = 0; gen < this.generations; gen++) {
      // Evaluate current population
      for (const ind of this.population) {
        ind.fitness = await fitnessFunction(ind.params);
        ind.sti = Math.min(1.0, ind.fitness * 0.6 + ind.sti * 0.4);
      }

      // (μ, λ)-ES: select μ best parents
      this.population.sort((a, b) => (b.fitness + b.sti) - (a.fitness + a.sti));
      const parents = this.population.slice(0, this.mu);

      // Generate λ offspring from parents
      const offspring = [];
      for (let i = 0; i < this.lambda; i++) {
        const parent = parents[Math.floor(Math.random() * this.mu)];
        offspring.push(parent.mutate());
      }

      // Replace population with offspring
      this.population = offspring;
    }

    // Return best individual
    this.population.sort((a, b) => (b.fitness + b.sti) - (a.fitness + a.sti));
    const best = this.population[0];
    return {
      params: Array.from(best.params),
      fitness: best.fitness.toFixed(4),
      sti: best.sti.toFixed(4),
      sigma: best.sigma.toFixed(4)
    };
  }
}

// Example fitness – can be task-specific (higher = better)
async function exampleFitness(params) {
  // Dummy: reward parameter vector close to [0.5, 0.5, ..., 0.5]
  let score = 0;
  for (const p of params) {
    score -= Math.abs(p - 0.5);
  }
  // Boost with Hyperon attention if integrated
  // const highAtt = await updateAttention(params.join(","));
  // score += highAtt.length * 0.1;
  return Math.max(0, Math.min(1, (score + params.length) / params.length));
}

// Export for index.html integration
export { ESEngine, exampleFitness };
