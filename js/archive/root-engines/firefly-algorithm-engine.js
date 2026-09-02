// firefly-algorithm-engine.js – sovereign client-side Firefly Algorithm
// MIT License – Autonomicity Games Inc. 2026

class Firefly {
  constructor(dimension, bounds = [-5, 5]) {
    this.dimension = dimension;
    this.position = new Float64Array(dimension);
    this.fitness = -Infinity;
    this.sti = 0.1; // attention from Hyperon

    for (let i = 0; i < dimension; i++) {
      this.position[i] = bounds[0] + Math.random() * (bounds[1] - bounds[0]);
    }
  }

  copy() {
    const copy = new Firefly(this.dimension);
    copy.position.set(this.position);
    copy.fitness = this.fitness;
    copy.sti = this.sti;
    return copy;
  }
}

class FireflyEngine {
  constructor(dimension = 10, nFireflies = 50, maxIter = 100, gamma = 1.0, beta0 = 1.0, alpha = 0.2) {
    this.dimension = dimension;
    this.nFireflies = nFireflies;
    this.maxIter = maxIter;
    this.gamma = gamma;     // light absorption coefficient
    this.beta0 = beta0;     // attractiveness at r=0
    this.alpha = alpha;     // randomization scale

    this.fireflies = Array.from({ length: nFireflies }, () => new Firefly(dimension));
    this.bestFirefly = this.fireflies[0];
  }

  async optimize(fitnessFunction) {
    // Initial evaluation
    for (const ff of this.fireflies) {
      ff.fitness = await fitnessFunction(ff.position);
      ff.sti = Math.min(1.0, ff.fitness * 0.6 + ff.sti * 0.4);
      if (ff.fitness > this.bestFirefly.fitness) {
        this.bestFirefly = ff.copy();
      }
    }

    for (let iter = 0; iter < this.maxIter; iter++) {
      // Sort fireflies by fitness descending
      this.fireflies.sort((a, b) => (b.fitness + b.sti) - (a.fitness + a.sti));

      // Move each firefly toward brighter ones
      for (let i = 0; i < this.nFireflies; i++) {
        for (let j = 0; j < i; j++) { // only move toward better fireflies
          const dist = this.euclideanDistance(this.fireflies[i].position, this.fireflies[j].position);
          const beta = this.beta0 * Math.exp(-this.gamma * dist * dist);

          if (this.fireflies[j].fitness > this.fireflies[i].fitness) {
            for (let k = 0; k < this.dimension; k++) {
              this.fireflies[i].position[k] += 
                beta * (this.fireflies[j].position[k] - this.fireflies[i].position[k]) +
                this.alpha * (Math.random() - 0.5);
            }
          }
        }

        // Bound clamping
        for (let k = 0; k < this.dimension; k++) {
          this.fireflies[i].position[k] = Math.max(-5, Math.min(5, this.fireflies[i].position[k]));
        }

        // Re-evaluate
        const newFitness = await fitnessFunction(this.fireflies[i].position);
        this.fireflies[i].fitness = newFitness;
        this.fireflies[i].sti = Math.min(1.0, newFitness * 0.6 + this.fireflies[i].sti * 0.4);

        if (newFitness > this.bestFirefly.fitness) {
          this.bestFirefly = this.fireflies[i].copy();
        }
      }

      // Optional: gradually reduce alpha (randomization)
      this.alpha *= 0.98;
    }

    return {
      bestPosition: Array.from(this.bestFirefly.position),
      bestFitness: this.bestFirefly.fitness.toFixed(4),
      sti: this.bestFirefly.sti.toFixed(4),
      iterations: this.maxIter
    };
  }

  euclideanDistance(a, b) {
    let sum = 0;
    for (let i = 0; i < this.dimension; i++) {
      sum += Math.pow(a[i] - b[i], 2);
    }
    return Math.sqrt(sum);
  }
}

// Example fitness function (higher = better)
async function exampleFitness(position) {
  // Dummy: reward vector close to [0.618, 0.618, ..., 0.618] (golden ratio)
  let sum = 0;
  const golden = (1 + Math.sqrt(5)) / 2 - 1;
  for (const p of position) {
    sum -= Math.abs(p - golden);
  }
  // Attention boost
  // const highAtt = await updateAttention(position.join(","));
  // sum += highAtt.length * 0.1;
  return Math.max(-1000, Math.min(0, sum + position.length * golden));
}

// Export for index.html integration
export { FireflyEngine, exampleFitness };
