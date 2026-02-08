// ga-engine.js — PATSAGi Council-forged classic genetic algorithm (Ultramasterpiece)
// Simple GA for fixed-length float genomes: tournament selection, uniform crossover, Gaussian mutation
// Evolves mercyParams (weights/thresholds/modifiers) on history replay fitness
// Pure browser-native — offline-first, complements NEAT for parameter tuning

class GeneticAlgorithm {
  constructor(popSize = 50, eliteSize = 5, mutationRate = 0.1, mutationStrength = 0.05) {
    this.popSize = popSize;
    this.eliteSize = eliteSize;
    this.mutationRate = mutationRate;
    this.mutationStrength = mutationStrength;
  }

  // Initialize population with random floats in range
  initializePopulation(genomeLength, min = 0.0, max = 1.0) {
    const population = [];
    for (let i = 0; i < this.popSize; i++) {
      const genome = [];
      for (let j = 0; j < genomeLength; j++) {
        genome.push(min + Math.random() * (max - min));
      }
      population.push({ genome, fitness: 0 });
    }
    return population;
  }

  // Tournament selection
  selectParent(population, tournamentSize = 3) {
    const tournament = [];
    for (let i = 0; i < tournamentSize; i++) {
      const randomIdx = Math.floor(Math.random() * population.length);
      tournament.push(population[randomIdx]);
    }
    tournament.sort((a, b) => b.fitness - a.fitness);
    return tournament[0];
  }

  // Uniform crossover
  crossover(parent1, parent2) {
    const child = [];
    for (let i = 0; i < parent1.genome.length; i++) {
      child.push(Math.random() < 0.5 ? parent1.genome[i] : parent2.genome[i]);
    }
    return child;
  }

  // Gaussian mutation
  mutate(genome) {
    for (let i = 0; i < genome.length; i++) {
      if (Math.random() < this.mutationRate) {
        genome[i] += (Math.random() - 0.5) * 2 * this.mutationStrength;
        genome[i] = Math.max(0.0, Math.min(1.0, genome[i])); // Clamp
      }
    }
    return genome;
  }

  // Evolve population (async for fitness eval)
  async evolve(population, evaluateFitness, generations = 50) {
    let bestIndividual = null;
    let bestFitness = -Infinity;

    for (let gen = 0; gen < generations; gen++) {
      // Evaluate fitness
      for (const individual of population) {
        individual.fitness = await evaluateFitness(individual.genome);
        if (individual.fitness > bestFitness) {
          bestFitness = individual.fitness;
          bestIndividual = { ...individual };
        }
      }

      // Elitism
      population.sort((a, b) => b.fitness - a.fitness);
      const nextGen = population.slice(0, this.eliteSize);

      // Breed new
      while (nextGen.length < this.popSize) {
        const parent1 = this.selectParent(population);
        const parent2 = this.selectParent(population);
        let childGenome = this.crossover(parent1, parent2);
        childGenome = this.mutate(childGenome);
        nextGen.push({ genome: childGenome, fitness: 0 });
      }

      population = nextGen;
      console.log(`GA generation ${gen + 1}/${generations} — Best fitness: ${bestFitness.toFixed(4)} ⚡️`);
    }

    return { bestGenome: bestIndividual.genome, bestFitness };
  }
}

export default GeneticAlgorithm;

// Init log
console.log('Classic genetic algorithm engine active — parameter evolution thriving ⚡️');
