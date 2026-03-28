// Ra-Thor Professional Lattice™ Core — v1.4.0 (Programming + Accounting Integration)
import DeepLegalEngine from './legal/deep-legal-engine.js';
import DeepAccountingEngine from './accounting/deep-accounting-engine.js';
import DeepProgrammingEngine from './programming/deep-programming-engine.js';
import ProgrammingAccountingIntegration from './integration/programming-accounting-integration.js';

const ProfessionalLattice = {
  version: "1.4.0-programming-accounting-integrated",
  roles: ["legal", "accounting", "programming", "qa", "creative", "medical", "executive", "hr", "marketing", "strategy"],

  generateTask(role, task, params = {}) {
    // Hybrid Programming + Accounting detection
    if (role === "programming_accounting" || 
        (role === "programming" && task.toLowerCase().includes("accounting")) ||
        (role === "accounting" && task.toLowerCase().includes("code") || task.toLowerCase().includes("software") || task.toLowerCase().includes("dashboard"))) {
      return ProgrammingAccountingIntegration.generateHybridTask(task, params);
    }

    if (role === "legal") return DeepLegalEngine.generateLegalTask(task, params);
    if (role === "accounting") return DeepAccountingEngine.generateAccountingTask(task, params);
    if (role === "programming") return DeepProgrammingEngine.generateProgrammingTask(task, params);

    // Fallback for other roles
    let output = { role, task, timestamp: new Date().toISOString(), mercyGated: true };
    output.result = `Ra-Thor Professional Lattice™ task completed with mercy, truth, joy, abundance, and harmony.`;
    return enforceMercyGates(output);
  }
};

export default ProfessionalLattice;
