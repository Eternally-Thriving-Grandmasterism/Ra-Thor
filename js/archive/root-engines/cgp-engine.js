// cgp-engine.js – sovereign client-side Cartesian Genetic Programming
// MIT License – Autonomicity Games Inc. 2026

// Function set – each function must have fixed arity
const CGP_FUNCTIONS = [
  { name: "add", arity: 2, fn: (a,b) => a + b },
  { name: "sub", arity: 2, fn: (a,b) => a - b },
  { name: "mul", arity: 2, fn: (a,b) => a * b },
  { name: "div", arity: 2, fn: (a,b) => b !== 0 ? a / b : 0 },
  { name: "sin", arity: 1, fn: Math.sin },
  { name: "cos", arity: 1, fn: Math.cos },
  { name: "abs", arity: 1, fn: Math.abs },
  { name: "max", arity: 2, fn: Math.max },
  { name: "min", arity: 2, fn: Math.min },
  { name: "and", arity: 2, fn: (a,b) => a && b ? 1 : 0 },
  { name: "or",  arity: 2, fn: (a,b) => a || b ? 1 : 0 },
  { name: "not", arity: 1, fn: (a) => a ? 0 : 1 }
];

// CGP parameters (can be tuned)
const CGP_CONFIG = {
  nInputs: 4,           // number of input registers
  nOutputs: 1,          // number of output nodes
  nColumns: 20,         // graph width
  nRows: 5,             // graph height
  nNodeInputs: 2,       // max inputs per node (arity ≤ this)
  mutationRate: 0.05,
  generations: 50,
  populationSize: 100
};

// Genome = flat array of integers
// Format for each node: [functionIndex, input1, input2, ..., outputConnection?]
class CGPIndividual {
  constructor(genome = null) {
    this.genome = genome || this.randomGenome();
    this.fitness = 0;
    this.sti = 0.1; // attention from Hyperon
  }

  randomGenome() {
    const genome = [];
    const nNodes = CGP_CONFIG.nColumns * CGP_CONFIG.nRows;
    for (let i = 0; i < nNodes; i++) {
      // Function index
      genome.push(Math.floor(Math.random() * CGP_FUNCTIONS.length));
      // Inputs (from previous nodes or inputs)
      for (let j = 0; j < CGP_CONFIG.nNodeInputs; j++) {
        const maxInput = i + CGP_CONFIG.nInputs; // allow recurrent connections
        genome.push(Math.floor(Math.random() * maxInput));
      }
    }
    // Output nodes – connect to last layer
    for (let i = 0; i < CGP_CONFIG.nOutputs; i++) {
      const lastLayerStart = (CGP_CONFIG.nColumns - 1) * CGP_CONFIG.nRows + CGP_CONFIG.nInputs;
      genome.push(Math.floor(Math.random() * (CGP_CONFIG.nRows + lastLayerStart)));
    }
    return genome;
  }

  evaluate(inputs) {
    const nNodes = CGP_CONFIG.nColumns * CGP_CONFIG.nRows;
    const registers = new Float64Array(CGP_CONFIG.nInputs + nNodes);
    inputs.forEach((v, i) => registers[i] = v);

    let geneIdx = 0;
    for (let node = 0; node < nNodes; node++) {
      const funcIdx = this.genome[geneIdx++];
      const func = CGP_FUNCTIONS[funcIdx];
      if (!func) {
        registers[CGP_CONFIG.nInputs + node] = 0;
        geneIdx += CGP_CONFIG.nNodeInputs;
        continue;
      }

      const args = [];
      for (let j = 0; j < func.arity; j++) {
        const inputIdx = this.genome[geneIdx++];
        args.push(registers[inputIdx] || 0);
      }

      registers[CGP_CONFIG.nInputs + node] = func.fn(...args);
    }

    // Collect outputs
    const outputs = [];
    for (let i = 0; i < CGP_CONFIG.nOutputs; i++) {
      const outIdx = this.genome[geneIdx++];
      outputs.push(registers[outIdx] || 0);
    }

    return outputs.length === 1 ? outputs[0] : outputs;
  }

  toString() {
    return `CGP genome [${this.genome.length} genes]`;
  }
}

// Population + GP evolution loop
class CGPEngine {
  constructor(config = CGP_CONFIG) {
    this.config = config;
    this.population = Array.from(
      { length: config.populationSize },
      () => new CGPIndividual()
    );
  }

  async evolve(fitnessFunction) {
    for (let gen = 0; gen < this.config.generations; gen++) {
      // Evaluate
      for (const ind of this.population) {
        ind.fitness = await fitnessFunction(ind);
        ind.sti = Math.min(1.0, ind.fitness * 0.6 + ind.sti * 0.4);
      }

      // Sort by fitness + STI
      this.population.sort((a, b) => (b.fitness + b.sti) - (a.fitness + a.sti));

      // Elitism: keep top 10%
      const elite = this.population.slice(0, Math.floor(this.config.populationSize * 0.1));

      // Breed next generation
      const nextGen = [...elite];
      while (nextGen.length < this.config.populationSize) {
        const p1 = this.tournamentSelect(5);
        const p2 = this.tournamentSelect(5);

        let childGenome = this.crossover(p1.genome, p2.genome);
        childGenome = this.mutate(childGenome);
        nextGen.push(new CGPIndividual(childGenome));
      }

      this.population = nextGen;
    }

    const best = this.population[0];
    return {
      genome: best.genome,
      fitness: best.fitness.toFixed(4),
      sti: best.sti.toFixed(4),
      programLength: best.genome.length
    };
  }

  tournamentSelect(size = 5) {
    let best = this.population[0];
    for (let i = 1; i < size; i++) {
      const cand = this.population[Math.floor(Math.random() * this.population.length)];
      if (cand.fitness + cand.sti > best.fitness + best.sti) best = cand;
    }
    return best;
  }

  crossover(genome1, genome2) {
    const point = Math.floor(Math.random() * genome1.length);
    return [
      ...genome1.slice(0, point),
      ...genome2.slice(point)
    ];
  }

  mutate(genome) {
    return genome.map(gene => {
      if (Math.random() < this.config.mutationRate) {
        return Math.floor(Math.random() * 100); // simplistic – can be smarter
      }
      return gene;
    });
  }
}

// Example fitness – can be task-specific
async function exampleFitness(ind) {
  // Dummy: reward programs that output values close to 42 for input 1
  const result = ind.evaluate([1, 2, 3, 4]);
  const score = -Math.abs(result - 42);
  // Attention boost
  // const highAtt = await updateAttention(ind.toString());
  // score += highAtt.length * 2;
  return Math.max(0, Math.min(100, score + 100)) / 100;
}

// Export for index.html integration
export { CGPEngine, exampleFitness };
