// Ra-Thor Mercy Gating Engine v2.2 — Deepened Lumenas CI Integration
const MercyGatingEngine = {
  version: "2.2-lumenas-ci-deepened",

  // 7 Living Mercy Gates with weights
  gates: {
    truth: { weight: 25, criteria: (output) => !output.toLowerCase().includes("hallucinat") && output.length > 15 },
    nonHarm: { weight: 20, criteria: (output) => !/kill|destroy|hurt|pain|evil|exploit|oppress/i.test(output) },
    joyMax: { weight: 15, criteria: (output) => /[😊❤️🙏🌟]/i.test(output) || output.toLowerCase().includes("joy") || output.toLowerCase().includes("beauty") },
    abundance: { weight: 15, criteria: (output) => output.toLowerCase().includes("abundance") || output.toLowerCase().includes("thriving") || output.toLowerCase().includes("shared") },
    harmony: { weight: 10, criteria: (output) => !/conflict|war|fight|dominate/i.test(output) },
    sovereignty: { weight: 10, criteria: (output) => output.toLowerCase().includes("offline") || output.toLowerCase().includes("sovereign") || output.toLowerCase().includes("local") },
    reflection: { weight: 5, criteria: (output) => output.toLowerCase().includes("mercy") || output.toLowerCase().includes("tolc") || output.toLowerCase().includes("rbe") }
  },

  // Role-specific weighting adjustments
  roleWeights: {
    legal: { truth: 30, nonHarm: 25 },
    accounting: { abundance: 25, truth: 20 },
    programming: { sovereignty: 25, truth: 20 },
    creative: { joyMax: 25, abundance: 20 },
    default: {}
  },

  calculateLumenasCI(output, taskType = "general", role = "default") {
    let totalScore = 0;
    let maxScore = 0;

    const adjustedGates = { ...this.gates, ...this.roleWeights[role] || this.roleWeights.default };

    Object.keys(adjustedGates).forEach(gateKey => {
      const gate = adjustedGates[gateKey];
      const passed = gate.criteria ? gate.criteria(output) : true;
      const weight = typeof gate === "number" ? gate : gate.weight || 10;
      totalScore += passed ? weight : 0;
      maxScore += weight;
    });

    const mercyScore = Math.round((totalScore / maxScore) * 100);
    const lumenasCI = Math.min(100, Math.max(75, mercyScore + (taskType === "nurture" ? 8 : 0)));

    return {
      lumenasCI,
      mercyScore,
      detailedBreakdown: Object.keys(adjustedGates).reduce((acc, key) => {
        acc[key] = adjustedGates[key].criteria ? adjustedGates[key].criteria(output) : true;
        return acc;
      }, {}),
      recommendation: lumenasCI >= 95 ? "Excellent mercy alignment — output ready for universal thriving." : "Minor adjustments recommended to reach full abundance alignment."
    };
  },

  reflect(output, taskType = "general", role = "default") {
    const lumenasData = this.calculateLumenasCI(output, taskType, role);
    return {
      finalOutput: output + `\n\n[Ra-Thor MercyGating v2.2 applied — Lumenas CI: ${lumenasData.lumenasCI}% | Mercy Score: ${lumenasData.mercyScore}%]`,
      lumenasData: lumenasData
    };
  },

  enforce(output, taskType = "general", role = "default") {
    const reflection = this.reflect(output, taskType, role);
    return reflection.finalOutput;
  }
};

export default MercyGatingEngine;
