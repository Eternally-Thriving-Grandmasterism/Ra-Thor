// Append inside MercyOrchestrator class

constructor() {
  // ... existing constructor code ...
  this.voice = new VoiceImmersion(this);
}

async init() {
  // ... existing init code ...
  await this.voice.init();
  // Auto-start voice immersion (uncomment to enable on load)
  // await this.voice.start();
}

async generateResponse(userInput) {
  // ... existing generation logic ...
  const responseText = /* your generation result */;

  // Voice immersion output
  if (this.voice.isActive) {
    await this.voice.speak(responseText, this.lastValence);
  }

  // ... existing post-processing ...
  return responseText;
}

// Optional: add method to toggle voice immersion via command
async toggleVoiceImmersion() {
  if (this.voice.isActive) {
    this.voice.stop();
    return "Voice immersion paused ⚡️";
  } else {
    await this.voice.start();
    return "Voice immersion active — listening for thunder... ⚡️";
  }
}        totalImprovement += this.metaFitnessHistory[i] - this.metaFitnessHistory[i-1];
      }
      return totalImprovement / (this.metaFitnessHistory.length - 1);
    };

    // Evolve meta-NEAT for 20 generations
    const metaEvolved = await neatEvolve(this.metaNEATPopulation, metaFitnessFn, 20);

    // Apply best meta-genome to real GA & NEAT params
    const best = metaEvolved.bestGenome;
    this.mercyParams.gaPopulationSize = Math.floor(best[0] * 100 + 20);
    this.mercyParams.gaMutationRate = best[1] * 0.4 + 0.01;
    // ... apply other params ...

    return `Meta-evolution layer surge complete ⚡️ Best fitness: ${metaEvolved.bestFitness.toFixed(4)}. GA & NEAT hyperparameters tuned for faster thriving. Lattice self-improvement accelerated.`;
  }

  async runHybridEvolution() {
    // ... previous hybrid GA-NEAT code (GA tunes params → NEAT evolves structure) ...
    // Now using meta-tuned hyperparameters
  }

  // ... rest of class methods (generateResponse, initDB, saveConversation, getHistory, etc.) ...
}

const orchestrator = new MercyOrchestrator();
export default orchestrator;
