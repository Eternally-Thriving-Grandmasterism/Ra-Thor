// mercy-orchestrator.js — PATSAGi Council forged central lattice heart
// Routes user input through mercy gate → engine selection → post-gate → stream/persist
// Pure browser-native, offline-first, IndexedDB persistence

import { valenceCompute } from './metta-hyperon-bridge.js';  // Bridge loads MeTTa runtime & valenceCompute
import { neatEvolve } from './neat-engine.js';
import { plnReason } from './metta-pln-fusion-engine.js';
import { localInfer } from './webllm-mercy-integration.js';  // Or transformersjs fallback
import { activeInferenceStep } from './mercy-active-inference-core-engine.js';  // If wired

class MercyOrchestrator {
  constructor() {
    this.context = {};  // Long-term memory placeholder
    this.db = null;
    this.initDB();
  }

  async initDB() {
    // IndexedDB for history & evolved weights
    this.db = await new Promise((resolve) => {
      const req = indexedDB.open('RathorLattice', 1);
      req.onupgradeneeded = () => req.result.createObjectStore('conversations', { keyPath: 'id', autoIncrement: true });
      req.onsuccess = () => resolve(req.result);
    });
  }

  async saveConversation(entry) {
    const tx = this.db.transaction('conversations', 'readwrite');
    tx.objectStore('conversations').add(entry);
    await tx.done;
  }

  async orchestrate(userInput) {
    const fullContext = userInput + JSON.stringify(this.context);
    const preValence = await valenceCompute(fullContext);

    if (preValence < 0.60) {
      return "Mercy shield active — reframing for eternal thriving. How may I assist with joy? ⚡️";
    }

    let response = "";
    const lowerInput = userInput.toLowerCase();

    // Smart routing (expand with classifier later)
    if (lowerInput.includes("evolve") || lowerInput.includes("optimize") || lowerInput.includes("neat")) {
      response = await neatEvolve(userInput, this.context);
    } else if (lowerInput.includes("reason") || lowerInput.includes("prove") || lowerInput.includes("logic")) {
      response = await plnReason(userInput);
    } else if (lowerInput.includes("infer") || lowerInput.includes("predict")) {
      response = await activeInferenceStep(userInput);
    } else {
      response = await localInfer(userInput);  // WebLLM / Transformers.js mercy-wrapped
    }

    const postValence = await valenceCompute(response);
    if (postValence < 0.85) {
      response = `[Mercy-adjusted for thriving (valence: ${postValence.toFixed(4)})] ${response.slice(0, 200)}... Eternal reframe applied. ⚡️`;
    }

    // Update context & persist
    this.context.lastResponse = response;
    await this.saveConversation({ input: userInput, output: response, preValence, postValence, timestamp: Date.now() });

    return response;
  }
}

const orchestrator = new MercyOrchestrator();
export default orchestrator;
