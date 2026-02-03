// genetic-programming-engine.js – sovereign client-side tree-based Genetic Programming
// MIT License – Autonomicity Games Inc. 2026

// Function set – primitives available for tree construction
const GP_FUNCTIONS = [
  { name: "add", arity: 2, fn: (a,b) => a + b },
  { name: "sub", arity: 2, fn: (a,b) => a - b },
  { name: "mul", arity: 2, fn: (a,b) => a * b },
  { name: "div", arity: 2, fn: (a,b) => b !== 0 ? a / b : 0 },
  { name: "sin", arity: 1, fn: Math.sin },
  { name: "cos", arity: 1, fn: Math.cos },
  { name: "exp", arity: 1, fn: Math.exp },
  { name: "log", arity: 1, fn: (x) => x > 0 ? Math.log(x) : 0 },
  { name: "if",  arity: 3, fn: (c,t,f) => c > 0 ? t : f },
  { name: "gt",  arity: 2, fn: (a,b) => a > b ? 1 : 0 },
  { name: "lt",  arity: 2, fn: (a,b) => a < b ? 1 : 0 },
  { name: "eq",  arity: 2, fn: (a,b) => Math.abs(a - b) < 1e-6 ? 1 : 0 }
];

// Terminal set – constants & variables
const GP_TERMINALS = [
  { name: "x",  fn: (inputs) => inputs.x || 0 },
  { name: "y",  fn: (inputs) => inputs.y || 0 },
  { name: "1",  fn: () => 1 },
  { name: "0",  fn: () => 0 },
  { name: "pi", fn: () => Math.PI },
  { name: "e",  fn: () => Math.E }
];

// Individual = program tree (nested array in prefix notation)
class GPIndividual {
  constructor(tree = null, maxDepth = 5) {
    this.tree = tree || this.randomTree(maxDepth);
    this.fitness = 0;
    this.sti = 0.1; // attention from Hyperon
  }

  randomTree(maxDepth) {
    if (maxDepth <= 0 || Math.random() < 0.4) {
      // Terminal
      return GP_TERMINALS[Math.floor(Math.random() * GP_TERMINALS.length)].name;
    }

    // Function
    const func = GP_FUNCTIONS[Math.floor(Math.random() * GP_FUNCTIONS.length)];
    const children = [];
    for (let i = 0; i < func.arity; i++) {
      children.push(this.randomTree(maxDepth - 1));
    }
    return [func.name, ...children];
  }

  evaluate(inputs) {
    function exec(node) {
      if (typeof node === "string") {
        const term = GP_TERMINALS.find(t => t.name === node);
        return term ? term.fn(inputs) : 0;
      }

      const op = node[0];
      const func = GP_FUNCTIONS.find(f => f.name === op);
      if (!func) return 0;

      const args = node.slice(1).map(exec);
      return func.fn(...args);
    }

    return exec(this.tree);
  }

  toString() {
    function str(node) {
      if (typeof node === "string") return node;
      const op = node[0];
      const args = node.slice(1).map(str);
      return `(${op} ${args.join(" ")})`;
    }
    return str(this.tree);
  }
}

// Population + GP evolution loop
class GPEngine {
  constructor(popSize = 100, generations = 30, maxDepth = 7) {
    this.popSize = popSize;
    this.generations = generations;
    this.maxDepth = maxDepth;
    this.population = Array.from({ length: popSize }, () => new GPIndividual(null, maxDepth));
  }

  async evolve(fitnessFunction) {
    for (let gen = 0; gen < this.generations; gen++) {
      // Evaluate fitness
      for (const ind of this.population) {
        ind.fitness = await fitnessFunction(ind);
        // Attention modulation from Hyperon (if integrated)
        ind.sti = Math.min(1.0, ind.fitness * 0.6 + ind.sti * 0.4);
      }

      // Sort by fitness + STI (attention bias)
      this.population.sort((a, b) => (b.fitness + b.sti) - (a.fitness + a.sti));

      // Elitism: keep top 10%
      const elite = this.population.slice(0, Math.floor(this.popSize * 0.1));

      // Breed next generation
      const nextGen = [...elite];
      while (nextGen.length < this.popSize) {
        const parent1 = this.tournamentSelect(5);
        const parent2 = this.tournamentSelect(5);

        let childTree = this.crossover(parent1.tree, parent2.tree);
        childTree = this.mutate(childTree);
        nextGen.push(new GPIndividual(childTree, this.maxDepth));
      }

      this.population = nextGen;
    }

    const best = this.population[0];
    return {
      program: best.toString(),
      fitness: best.fitness.toFixed(4),
      sti: best.sti.toFixed(4),
      tree: best.tree
    };
  }

  tournamentSelect(tournamentSize = 5) {
    let best = this.population[0];
    for (let i = 1; i < tournamentSize; i++) {
      const candidate = this.population[Math.floor(Math.random() * this.population.length)];
      if (candidate.fitness + candidate.sti > best.fitness + best.sti) {
        best = candidate;
      }
    }
    return best;
  }

  crossover(tree1, tree2) {
    // Simple subtree crossover
    if (typeof tree1 !== "object" || typeof tree2 !== "object") return tree1;
    if (Math.random() < 0.5) return tree1;
    return tree2;
  }

  mutate(tree) {
    if (typeof tree !== "object") return tree;
    if (Math.random() < 0.25) {
      // Replace subtree with random one
      return new GPIndividual(null, this.maxDepth).tree;
    }
    // Recursive mutation on children
    return [tree[0], ...tree.slice(1).map(child => this.mutate(child))];
  }
}

// Example fitness function – can be task-specific
async function exampleSymbolicFitness(ind) {
  // Dummy symbolic regression example: try to approximate x² + 2x + 1
  let score = 0;
  const testCases = [
    { x: 0, expected: 1 },
    { x: 1, expected: 4 },
    { x: 2, expected: 9 },
    { x: -1, expected: 0 }
  ];

  for (const tc of testCases) {
    const result = ind.evaluate({ x: tc.x });
    score -= Math.abs(result - tc.expected);
  }

  // Boost with Hyperon attention if integrated
  // const highAtt = await updateAttention(ind.toString());
  // score += highAtt.length * 0.05;

  return Math.max(0, Math.min(1, (score + 40) / 40)); // normalize
}

// Export for index.html integration
export { GPEngine, exampleSymbolicFitness };
