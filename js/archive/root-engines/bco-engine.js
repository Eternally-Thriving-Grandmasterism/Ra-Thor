// bco-engine.js – sovereign client-side Bee Colony Optimization
// MIT License – Autonomicity Games Inc. 2026

class Bee {
  constructor(dimension, bounds = [-5, 5]) {
    this.dimension = dimension;
    this.position = new Float64Array(dimension);
    this.fitness = -Infinity;
    this.sti = 0.1; // attention from Hyperon
    this.trial = 0; // counter for abandonment

    for (let i = 0; i < dimension; i++) {
      this.position[i] = bounds[0] + Math.random() * (bounds[1] - bounds[0]);
    }
  }

  copy() {
    const copy = new Bee(this.dimension);
    copy.position.set(this.position);
    copy.fitness = this.fitness;
    copy.sti = this.sti;
    copy.trial = this.trial;
    return copy;
  }
}

class BCOEngine {
  constructor(dimension = 10, colonySize = 60, maxCycles = 100, limit = 100) {
    this.dimension = dimension;
    this.colonySize = colonySize;
    this.maxCycles = maxCycles;
    this.limit = limit; // abandonment limit

    this.bees = Array.from({ length: colonySize }, () => new Bee(dimension));
    this.bestBee = this.bees[0];
  }

  async optimize(fitnessFunction) {
    for (let cycle = 0; cycle < this.maxCycles; cycle++) {
      // Phase 1: Employed bees – local search around own food source
      for (const bee of this.bees) {
        const partner = this.bees[Math.floor(Math.random() * this.colonySize)];
        const j = Math.floor(Math.random() * this.dimension);
        const phi = Math.random() * 2 - 1;

        const newPos = bee.position.slice();
        newPos[j] = bee.position[j] + phi * (bee.position[j] - partner.position[j]);

        // Bound clamping
        newPos[j] = Math.max(-5, Math.min(5, newPos[j]));

        const newFitness = await fitnessFunction(newPos);
        if (newFitness > bee.fitness) {
          bee.position.set(newPos);
          bee.fitness = newFitness;
          bee.trial = 0;
          bee.sti = Math.min(1.0, newFitness * 0.6 + bee.sti * 0.4);
        } else {
          bee.trial++;
        }

        if (newFitness > this.bestBee.fitness) {
          this.bestBee = bee.copy();
        }
      }

      // Phase 2: Onlooker bees – probabilistic selection based on fitness
      const fitnessSum = this.bees.reduce((sum, bee) => sum + Math.max(0, bee.fitness), 0);
      for (let i = 0; i < this.colonySize; i++) {
        let r = Math.random() * fitnessSum;
        let selected = null;
        for (const bee of this.bees) {
          r -= Math.max(0, bee.fitness);
          if (r <= 0) {
            selected = bee;
            break;
          }
        }

        if (!selected) selected = this.bees[0];

        const partner = this.bees[Math.floor(Math.random() * this.colonySize)];
        const j = Math.floor(Math.random() * this.dimension);
        const phi = Math.random() * 2 - 1;

        const newPos = selected.position.slice();
        newPos[j] = selected.position[j] + phi * (selected.position[j] - partner.position[j]);
        newPos[j] = Math.max(-5, Math.min(5, newPos[j]));

        const newFitness = await fitnessFunction(newPos);
        if (newFitness > selected.fitness) {
          selected.position.set(newPos);
          selected.fitness = newFitness;
          selected.trial = 0;
          selected.sti = Math.min(1.0, newFitness * 0.6 + selected.sti * 0.4);
        } else {
          selected.trial++;
        }

        if (newFitness > this.bestBee.fitness) {
          this.bestBee = selected.copy();
        }
      }

      // Phase 3: Scout bees – abandon exhausted sources
      for (const bee of this.bees) {
        if (bee.trial >= this.limit) {
          bee.position = new Float64Array(this.dimension);
          for (let i = 0; i < this.dimension; i++) {
            bee.position[i] = -5 + Math.random() * 10;
          }
          bee.fitness = await fitnessFunction(bee.position);
          bee.trial = 0;
          bee.sti = 0.1;

          if (bee.fitness > this.bestBee.fitness) {
            this.bestBee = bee.copy();
          }
        }
      }
    }

    return {
      bestPosition: Array.from(this.bestBee.position),
      bestFitness: this.bestBee.fitness.toFixed(4),
      sti: this.bestBee.sti.toFixed(4),
      cycles: this.maxCycles
    };
  }
}

// Example fitness function (higher = better)
async function exampleFitness(position) {
  // Dummy: reward vector close to [1, 1, ..., 1]
  let sum = 0;
  for (const p of position) {
    sum -= Math.abs(p - 1);
  }
  // Attention boost
  // const highAtt = await updateAttention(position.join(","));
  // sum += highAtt.length * 0.1;
  return Math.max(-1000, Math.min(0, sum + position.length));
}

// Export for index.html integration
export { BCOEngine, exampleFitness };
