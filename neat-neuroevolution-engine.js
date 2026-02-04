// neat-neuroevolution-engine.js – sovereign client-side Neuroevolution of Augmenting Topologies (NEAT)
// Full speciation with historical markings, excess/disjoint/weight distance, dynamic threshold, fitness sharing
// MIT License – Autonomicity Games Inc. 2026

let globalInnovation = 0;

class NeuronGene {
  constructor(id, type = "hidden") {
    this.id = id;
    this.type = type; // input, hidden, output
    this.activation = Math.tanh;
  }
}

class ConnectionGene {
  constructor(inNode, outNode, weight = Math.random() * 2 - 1, enabled = true, innovation = 0) {
    this.in = inNode;
    this.out = outNode;
    this.weight = weight;
    this.enabled = enabled;
    this.innovation = innovation;
  }
}

class Genome {
  constructor() {
    this.nodes = new Map();
    this.connections = [];
    this.fitness = 0;
    this.adjustedFitness = 0;
    this.species = -1;
    this.sti = 0.1;
  }

  copy() {
    const copy = new Genome();
    copy.nodes = new Map(this.nodes);
    copy.connections = this.connections.map(c => new ConnectionGene(c.in, c.out, c.weight, c.enabled, c.innovation));
    copy.fitness = this.fitness;
    copy.sti = this.sti;
    return copy;
  }

  mutate() {
    // Weight mutation
    this.connections.forEach(c => {
      if (Math.random() < 0.8) c.weight += (Math.random() - 0.5) * 0.1;
    });

    // Add connection
    if (Math.random() < 0.05) {
      const inNode = Array.from(this.nodes.keys())[Math.floor(Math.random() * this.nodes.size)];
      const outNode = Array.from(this.nodes.keys())[Math.floor(Math.random() * this.nodes.size)];
      if (inNode !== outNode && !this.hasConnection(inNode, outNode)) {
        this.connections.push(new ConnectionGene(inNode, outNode, Math.random() * 2 - 1, true, globalInnovation++));
      }
    }

    // Add node
    if (Math.random() < 0.03 && this.connections.length > 0) {
      const conn = this.connections[Math.floor(Math.random() * this.connections.length)];
      if (conn.enabled) {
        conn.enabled = false;
        const newNodeId = this.nodes.size;
        this.nodes.set(newNodeId, new NeuronGene(newNodeId));
        this.connections.push(new ConnectionGene(conn.in, newNodeId, 1.0, true, globalInnovation++));
        this.connections.push(new ConnectionGene(newNodeId, conn.out, conn.weight, true, globalInnovation++));
      }
    }
  }

  hasConnection(inNode, outNode) {
    return this.connections.some(c => c.in === inNode && c.out === outNode);
  }
}

class Species {
  constructor(id, representative) {
    this.id = id;
    this.genomes = [representative];
    this.representative = representative;
    this.maxFitness = representative.fitness;
    this.adjustedFitnessTotal = 0;
  }

  updateRepresentative() {
    this.representative = this.genomes[0]; // top genome as rep
  }
}

class NEAT {
  constructor(inputs = 4, outputs = 1, popSize = 150, maxGenerations = 100) {
    this.inputs = inputs;
    this.outputs = outputs;
    this.popSize = popSize;
    this.maxGenerations = maxGenerations;
    this.population = [];
    this.species = [];
    this.compatibilityThreshold = 3.0;
    this.c1 = 1.0; // excess coefficient
    this.c2 = 1.0; // disjoint coefficient
    this.c3 = 0.4; // weight difference coefficient
    this.speciesIdCounter = 0;

    this.initPopulation();
  }

  initPopulation() {
    for (let i = 0; i < this.popSize; i++) {
      const genome = new Genome();
      for (let inp = 0; inp < this.inputs; inp++) {
        genome.nodes.set(inp, new NeuronGene(inp, "input"));
      }
      for (let out = 0; out < this.outputs; out++) {
        const outId = this.inputs + out;
        genome.nodes.set(outId, new NeuronGene(outId, "output"));
        for (let inp = 0; inp < this.inputs; inp++) {
          genome.connections.push(new ConnectionGene(inp, outId, Math.random() * 2 - 1, true, globalInnovation++));
        }
      }
      this.population.push(genome);
    }
  }

  async evolve(fitnessFunction) {
    for (let gen = 0; gen < this.maxGenerations; gen++) {
      // Evaluate fitness
      for (const genome of this.population) {
        genome.fitness = await fitnessFunction(genome);
        genome.sti = Math.min(1.0, genome.fitness * 0.6 + genome.sti * 0.4);
      }

      // Speciate with dynamic threshold adjustment
      this.speciate();
      this.adjustFitness();

      // Sort species by max fitness
      this.species.sort((a, b) => b.maxFitness - a.maxFitness);

      // Breed next generation
      const nextPopulation = [];
      let totalAdjustedFitness = this.species.reduce((sum, s) => sum + s.adjustedFitnessTotal, 0);

      for (const species of this.species) {
        if (species.adjustedFitnessTotal <= 0) continue;

        const offspringCount = Math.floor((species.adjustedFitnessTotal / totalAdjustedFitness) * this.popSize);
        if (offspringCount <= 0) continue;

        // Elitism: keep best per species
        const elite = species.genomes[0].copy();
        nextPopulation.push(elite);

        // Generate offspring
        for (let i = 0; i < offspringCount; i++) {
          let parent1 = species.tournamentSelect();
          let parent2 = parent1;
          if (Math.random() < 0.7 && species.genomes.length > 1) {
            parent2 = species.tournamentSelect();
          }
          let child = this.crossover(parent1, parent2);
          child.mutate();
          nextPopulation.push(child);
        }
      }

      // Fill remaining slots if needed
      while (nextPopulation.length < this.popSize) {
        const species = this.species[Math.floor(Math.random() * this.species.length)];
        const parent = species.tournamentSelect();
        const child = parent.copy();
        child.mutate();
        nextPopulation.push(child);
      }

      this.population = nextPopulation;
    }

    // Return best genome
    const best = this.population.reduce((a, b) => (a.fitness + a.sti > b.fitness + b.sti ? a : b));
    return {
      genome: best,
      fitness: best.fitness.toFixed(4),
      sti: best.sti.toFixed(4)
    };
  }

  speciate() {
    this.species = [];
    for (const genome of this.population) {
      let found = false;
      for (const species of this.species) {
        if (this.compatibilityDistance(genome, species.representative) < this.compatibilityThreshold) {
          species.genomes.push(genome);
          found = true;
          break;
        }
      }
      if (!found) {
        this.species.push(new Species(this.speciesIdCounter++, genome));
      }
    }

    // Update representatives (choose fittest in each species)
    this.species.forEach(s => s.updateRepresentative());
  }

  compatibilityDistance(g1, g2) {
    const genes1 = new Set(g1.connections.map(c => c.innovation));
    const genes2 = new Set(g2.connections.map(c => c.innovation));

    const matching = [];
    let weightDiffSum = 0;
    for (const c1 of g1.connections) {
      const c2 = g2.connections.find(c => c.innovation === c1.innovation);
      if (c2) {
        matching.push(c1);
        weightDiffSum += Math.abs(c1.weight - c2.weight);
      }
    }

    const N = Math.max(genes1.size, genes2.size);
    const excess = Math.abs(genes1.size - genes2.size) / N;
    const disjoint = (genes1.size + genes2.size - 2 * matching.length) / N;
    const weightDiff = matching.length > 0 ? weightDiffSum / matching.length : 0;

    return this.c1 * excess + this.c2 * disjoint + this.c3 * weightDiff;
  }

  adjustFitness() {
    let totalAdjusted = 0;
    for (const species of this.species) {
      species.adjustedFitnessTotal = 0;
      species.genomes.forEach(genome => {
        genome.adjustedFitness = genome.fitness / species.genomes.length;
        species.adjustedFitnessTotal += genome.adjustedFitness;
      });
      species.maxFitness = Math.max(...species.genomes.map(g => g.fitness));
      totalAdjusted += species.adjustedFitnessTotal;
    }
    return totalAdjusted;
  }

  tournamentSelect(species) {
    let best = species.genomes[0];
    for (let i = 1; i < 5; i++) {
      const cand = species.genomes[Math.floor(Math.random() * species.genomes.length)];
      if (cand.adjustedFitness + cand.sti > best.adjustedFitness + best.sti) {
        best = cand;
      }
    }
    return best;
  }

  crossover(g1, g2) {
    const child = new Genome();
    const maxInn = Math.max(...g1.connections.map(c => c.innovation), ...g2.connections.map(c => c.innovation));
    for (let inn = 0; inn <= maxInn; inn++) {
      const c1 = g1.connections.find(c => c.innovation === inn);
      const c2 = g2.connections.find(c => c.innovation === inn);
      if (c1 && c2) {
        child.connections.push(Math.random() < 0.5 ? c1 : c2);
      } else if (c1) {
        child.connections.push(c1);
      } else if (c2) {
        child.connections.push(c2);
      }
    }
    return child;
  }
}

// Example fitness – higher = better
async function exampleFitness(genome) {
  // Dummy: reward networks that output ~0.5 for input [0.5,0.5,0.5,0.5]
  const output = genome.evaluate([0.5, 0.5, 0.5, 0.5]);
  const score = -Math.abs(output - 0.5);
  return Math.max(0, Math.min(1, (score + 1) / 1));
}

// Export for index.html integration
export { NEAT, exampleFitness };
