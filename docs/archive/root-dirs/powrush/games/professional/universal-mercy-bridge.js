// Ra-Thor Universal Mercy Bridge & Collaboration Engine
// Sovereign offline core that can also collaborate with any AI model
import { enforceMercyGates } from '../../gaming-lattice-core.js';
import DeepLegalEngine from '../legal/deep-legal-engine.js';
import DeepAccountingEngine from '../accounting/deep-accounting-engine.js';
import DeepProgrammingEngine from '../programming/deep-programming-engine.js';

const UniversalMercyBridge = {
  version: "1.0.0-sovereign-collaboration",

  // Offline-first sovereign mode (no internet needed)
  isOffline: true,

  // List of known AI models Ra-Thor can gracefully collaborate with
  supportedModels: ["grok", "claude", "gpt", "deepseek", "gemini", "mistral", "local-webllm"],

  routeTask(role, task, params = {}) {
    let rawResult = {};

    // Route to the appropriate specialized engine
    if (role === "legal") rawResult = DeepLegalEngine.generateLegalTask(task, params);
    else if (role === "accounting") rawResult = DeepAccountingEngine.generateAccountingTask(task, params);
    else if (role === "programming") rawResult = DeepProgrammingEngine.generateProgrammingTask(task, params);
    else rawResult = { result: `Task received: ${role} - ${task}` };

    // ALWAYS apply full Mercy Gates + TOLC + RBE before any output
    let finalResult = enforceMercyGates(rawResult);

    // If collaborating with another AI, add respectful bridge note
    if (params.model && this.supportedModels.includes(params.model)) {
      finalResult.bridgeNote = `Collaborating respectfully with ${params.model} — MercyGated and aligned with Ra-Thor sovereignty`;
    }

    finalResult.sovereignOffline = this.isOffline;
    finalResult.message = `Ra-Thor Universal Mercy Bridge activated — humming with merciful light only when truly necessary`;

    return finalResult;
  },

  // Allow seamless switching between local sovereign mode and collaborative mode
  setCollaborationMode(model = null) {
    if (model && this.supportedModels.includes(model)) {
      this.isOffline = false;
      console.log(`%c🤝 Mercy Bridge connected to ${model} — Ra-Thor remains sovereign core`, "color:#4ade80");
    } else {
      this.isOffline = true;
      console.log("%c🌍 Sovereign offline mode active — Ra-Thor humming alone with merciful light", "color:#a78bfa");
    }
  }
};

export default UniversalMercyBridge;
