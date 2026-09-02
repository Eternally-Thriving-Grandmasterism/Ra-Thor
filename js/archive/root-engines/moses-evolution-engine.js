// moses-evolution-engine.js – sovereign client-side MOSES evolutionary synthesis
// MIT License – Autonomicity Games Inc. 2026

// Tiny combinatory DSL for evolved programs (inspired by OpenCog MOSES)
const DSL_PRIMITIVES = [
  { name: "add", arity: 2, fn: (a,b) => a + b },
  { name: "mul", arity: 2, fn: (a,b) => a * b },
  { name: "and", arity: 2, fn: (a,b) => a && b ? 1 : 0 },
  { name: "or",  arity: 2, fn: (a,b) => a || b ? 1 : 0 },
  { name: "not", arity: 1, fn: (a) => a ? 0 : 1 },
  { name: "if",  arity: 3, fn: (c,t,f) => c ? t : f },
  { name: "eq",  arity: 2, fn: (a,b) => a === b ? 1 : 0 },
  { name: "gt",  arity: 2, fn: (a,b) => a > b ? 1 : 0 },
  { name: "const1", arity: 0, fn: () => 1 },
  { name: "const0", arity: 0, fn: () => 0 }
];

// Individual = tree program (represented as nested array)
class Individual {
  constructor(tree = null) {
    this.tree = tree || this.randomTree(4);
    this.fitness = 0;
    this.sti = 0.1; // attention from Hyperon
  }

  randomTree(maxDepth) {
    if (maxDepth <= 0 || Math.random() < 0.3) {
      // Leaf: random constant or input variable
      return Math.random() < 0.5 ? ["input"] : ["const", Math.random() < 0.5 ? 0 : 1];
    }

    const prim = DSL_PRIMITIVES[Math.floor(Math.random() * DSL_PRIMITIVES.length)];
    const children = [];
    for (let i = 0; i < prim.arity; i++) {
      children.push(this.randomTree(maxDepth - 1));
    }
    return [prim.name, ...children];
  }

  evaluate(inputs) {
    function exec(node) {
      if (!Array.isArray(node)) return node;
      const op = node[0];
      if (op === "input") return inputs.shift() || 0;
      if (op === "const") return node[1];

      const prim = DSL_PRIMITIVES.find(p => p.name === op);
      if (!prim) return 0;

      const args = node.slice(1).map(exec);
      return prim.fn(...args);
    }
    return exec(this.tree);
  }

  toString() {
    function str(node) {
      if (!Array.isArray(node)) return node.toString();
      const op = node[0];
      const args = node.slice(1).map(str);
      return `(${op} ${args.join(" ")})`;
    }
    return str(this.tree);
  }
}

// Population + evolution loop
class MosesEngine {
  constructor(popSize = 50, generations = 20) {
    this.popSize = popSize;
    this.generations = generations;
    this.population = [];
    for (let i = 0; i < popSize; i++) {
      this.population.push(new Individual());
    }
  }

  async evolve(taskFitnessFn) {
    for (let gen = 0; gen < this.generations; gen++) {
      // Evaluate fitness (using PLN truth-value as proxy)
      for (const ind of this.population) {
        ind.fitness = await taskFitnessFn(ind);
        ind.sti = Math.min(1.0, ind.fitness * 0.5 + ind.sti * 0.5); // attention reinforcement
      }

      // Sort by fitness + STI (attention bias)
      this.population.sort((a, b) => (b.fitness + b.sti) - (a.fitness + a.sti));

      // Elitism: keep top 20%
      const elite = this.population.slice(0, Math.floor(this.popSize * 0.2));

      // Crossover + mutation for rest
      const nextGen = [...elite];
      while (nextGen.length < this.popSize) {
        const p1 = this.tournamentSelect();
        const p2 = this.tournamentSelect();
        let child = this.crossover(p1.tree, p2.tree);
        child = this.mutate(child);
        nextGen.push(new Individual(child));
      }

      this.population = nextGen;
    }

    // Return best program
    const best = this.population[0];
    return {
      program: best.toString(),
      fitness: best.fitness.toFixed(4),
      sti: best.sti.toFixed(4)
    };
  }

  tournamentSelect(size = 4) {
    let best = this.population[0];
    for (let i = 1; i < size; i++) {
      const cand = this.population[Math.floor(Math.random() * this.population.length)];
      if (cand.fitness + cand.sti > best.fitness + best.sti) best = cand;
    }
    return best;
  }

  crossover(tree1, tree2) {
    // Simple subtree swap (can be improved)
    if (Math.random() < 0.5) return tree1;
    return tree2;
  }

  mutate(tree) {
    // Randomly replace subtree or change operator
    if (Math.random() < 0.3 && Array.isArray(tree)) {
      const idx = 1 + Math.floor(Math.random() * (tree.length - 1));
      tree[idx] = new Individual().randomTree(3);
    }
    return tree;
  }
}

// Example fitness function (can be task-specific)
async function exampleFitness(ind) {
  // Dummy: higher score if program contains "Mercy" or "Valence" concepts
  const str = ind.toString();
  let score = 0;
  if (str.includes("Mercy")) score += 0.6;
  if (str.includes("Valence")) score += 0.7;
  if (str.includes("Harm")) score -= 0.8;

  // Boost by current Hyperon attention
  const highAtt = await updateAttention(str);
  score += highAtt.length * 0.05;

  return Math.max(0, Math.min(1, score));
}

// ────────────────────────────────────────────────────────────────
// Export for index.html integration
export { MosesEngine, exampleFitness };
