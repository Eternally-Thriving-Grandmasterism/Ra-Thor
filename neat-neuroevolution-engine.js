// neat-neuroevolution-engine.js – sovereign client-side Neuroevolution of Augmenting Topologies (NEAT)
// Topology + weight evolution, speciation, historical markings, crossover compatibility
// MIT License – Autonomicity Games Inc. 2026

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
      if (Math.random() < 0.8) {
        c.weight += (Math.random() - 0.5) * 0.1;
      }
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

let globalInnovation = 0;

class NEAT {
  constructor(inputs = 4, outputs = 1, popSize = 150, maxGenerations = 100) {
    this.inputs = inputs;
    this.outputs = outputs;
    this.popSize = popSize;
    this.maxGenerations = maxGenerations;
    this.population = [];
    this.species = [];
    this.compatibilityThreshold = 3.0;
    this.c1 = 1.0; // excess
    this.c2 = 1.0; // disjoint
    this.c3 = 0.4; // weight diff

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
      for (const genome of this.population) {
        genome.fitness = await fitnessFunction(genome);
        genome.sti = Math.min(1.0, genome.fitness * 0.6 + genome.sti * 0.4);
      }

      this.speciate();
      this.adjustFitness();

      this.species.sort((a, b) => b.maxFitness - a.maxFitness);

      const nextPopulation = [];
      for (const species of this.species) {
        const offspringCount = Math.floor(species.adjustedFitnessTotal / this.getTotalAdjustedFitness() * this.popSize);
        const elite = species.genomes[0];
        nextPopulation.push(elite.copy());

        for (let i = 0; i < offspringCount; i++) {
          const parent1 = species.tournamentSelect();
          let parent2 = parent1;
          if (Math.random() < 0.7 && species.genomes.length > 1) {
            parent2 = species.tournamentSelect();
          }
          let child = this.crossover(parent1, parent2);
          child.mutate();
          nextPopulation.push(child);
        }
      }

      while (nextPopulation.length < this.popSize) {
        const species = this.species[Math.floor(Math.random() * this.species.length)];
        const parent = species.tournamentSelect();
        const child = parent.copy();
        child.mutate();
        nextPopulation.push(child);
      }

      this.population = nextPopulation;
    }

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
        const rep = species.genomes[0];
        if (this.compatibilityDistance(genome, rep) < this.compatibilityThreshold) {
          species.genomes.push(genome);
          found = true;
          break;
        }
      }
      if (!found) {
        this.species.push({ genomes: [genome], maxFitness: genome.fitness, adjustedFitnessTotal: 0 });
      }
    }
  }

  compatibilityDistance(g1, g2) {
    const genes1 = new Set(g1.connections.map(c => c.innovation));
    const genes2 = new Set(g2.connections.map(c => c.innovation));
    const disjoint = genes1.symmetricDifference(genes2).size;
    const excess = Math.abs(genes1.size - genes2.size);
    let weightDiff = 0;
    let matching = 0;
    g1.connections.forEach(c1 => {
      const c2 = g2.connections.find(c => c.innovation === c1.innovation);
      if (c2) {
        weightDiff += Math.abs(c1.weight - c2.weight);
        matching++;
      }
    });
    weightDiff /= matching || 1;
    return this.c1 * excess + this.c2 * disjoint + this.c3 * weightDiff;
  }

  adjustFitness() {
    for (const species of this.species) {
      species.adjustedFitnessTotal = 0;
      species.genomes.forEach(genome => {
        genome.adjustedFitness = genome.fitness / species.genomes.length;
        species.adjustedFitnessTotal += genome.adjustedFitness;
      });
      species.maxFitness = Math.max(...species.genomes.map(g => g.fitness));
    }
  }

  tournamentSelect() {
    const total = this.species.reduce((sum, s) => sum + s.adjustedFitnessTotal, 0);
    let r = Math.random() * total;
    let selectedSpecies;
    for (const species of this.species) {
      r -= species.adjustedFitnessTotal;
      if (r <= 0) {
        selectedSpecies = species;
        break;
      }
    }
    return selectedSpecies.genomes[Math.floor(Math.random() * selectedSpecies.genomes.length)];
  }

  crossover(g1, g2) {
    const child = new Genome();
    const maxInnovation = Math.max(...g1.connections.map(c => c.innovation), ...g2.connections.map(c => c.innovation));
    for (let inn = 0; inn <= maxInnovation; inn++) {
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
  // Dummy: reward networks that output ~0.5 for input [0.5, 0.5, 0.5, 0.5]
  const output = genome.evaluate([0.5, 0.5, 0.5, 0.5]);
  const score = -Math.abs(output - 0.5);
  return Math.max(0, Math.min(1, (score + 1) / 1));
}

// Export for index.html integration
export { NEAT, exampleFitness };
