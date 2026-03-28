// Ra-Thor Professional Lattice™ Core — v1.6.0 (Growth & Nurture Lattice Integrated)
import DeepLegalEngine from './legal/deep-legal-engine.js';
import DeepAccountingEngine from './accounting/deep-accounting-engine.js';
import DeepProgrammingEngine from './programming/deep-programming-engine.js';
import GrowthNurtureLattice from '../nurture/growth-nurture-lattice.js';
import UniversalMercyBridge from './universal-mercy-bridge.js';

const ProfessionalLattice = {
  version: "1.6.0-growth-nurture-integrated",
  roles: ["legal", "accounting", "programming", "qa", "creative", "medical", "executive", "hr", "marketing", "strategy"],

  generateTask(role, task, params = {}) {
    // Route through Universal Mercy Bridge first
    let bridged = UniversalMercyBridge.routeTask(role, task, params);

    // If the task is about growth or nurturing, route to the new lattice
    if (task.toLowerCase().includes("growth") || task.toLowerCase().includes("nurture") || task.toLowerCase().includes("evolve") || task.toLowerCase().includes("develop")) {
      return GrowthNurtureLattice.generateNurtureTask(task, params);
    }

    if (role === "legal") return DeepLegalEngine.generateLegalTask(task, params);
    if (role === "accounting") return DeepAccountingEngine.generateAccountingTask(task, params);
    if (role === "programming") return DeepProgrammingEngine.generateProgrammingTask(task, params);

    return bridged;
  }
};

export default ProfessionalLattice;
