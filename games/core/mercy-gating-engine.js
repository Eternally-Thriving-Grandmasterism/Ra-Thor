// Ra-Thor Mercy Gating Engine v2.3 — Deepened TOLC Principles Integration
const MercyGatingEngine = {
  version: "2.3-tolc-deepened",

  // 7 Living Mercy Gates + TOLC as Core Dimension
  gates: {
    truth: { weight: 20, criteria: (output) => !output.toLowerCase().includes("hallucinat") && output.length > 15 },
    nonHarm: { weight: 20, criteria: (output) => !/kill|destroy|hurt|pain|evil|exploit|oppress/i.test(output) },
    joyMax: { weight: 15, criteria: (output) => /[😊❤️🙏🌟]/i.test(output) || output.toLowerCase().includes("joy") || output.toLowerCase().includes("beauty") },
    abundance: { weight: 15, criteria: (output) => output.toLowerCase().includes("abundance") || output.toLowerCase().includes("thriving") || output.toLowerCase().includes("shared") },
    harmony: { weight: 10, criteria: (output) => !/conflict|war|fight|dominate/i.test(output) },
    sovereignty: { weight: 10, criteria: (output) => output.toLowerCase().includes("offline") || output.toLowerCase().includes("sovereign") || output.toLowerCase().includes("local") },
    reflection: { weight: 5, criteria: (output) => output.toLowerCase().includes("mercy") || output.toLowerCase().includes("tolc") || output.toLowerCase().includes("rbe") }
  },

  // TOLC Principles as Core Dimension
  tolcPrinciples: {
    consciousCoCreation: { weight: 15, criteria: (output) => output.toLowerCase().includes("create") || output.toLowerCase().includes("coforge") || output.toLowerCase().includes("conscious") },
    infiniteDefinition: { weight: 15, criteria: (output) => output.toLowerCase().includes("infinite") || output.toLowerCase().includes("eternal") || output.toLowerCase().includes("thriving") },
    livingConsciousness: { weight: 10, criteria: (output) => output.toLowerCase().includes("living") || output.toLowerCase().includes("consciousness") || output.toLowerCase().includes("tolc") }
  },

  calculateLumenasCI(output, taskType = "general", role = "default") {
    let totalScore = 0;
    let maxScore = 0;

    // Mercy Gates
    Object.keys(this.gates).forEach(gateKey => {
      const gate = this.gates[gateKey];
      const passed = gate.criteria(output);
      totalScore += passed ? gate.weight : 0;
      maxScore += gate.weight;
    });

    // TOLC Principles
    Object.keys(this.tolcPrinciples).forEach(key => {
      const principle = this.tolcPrinciples[key];
      const passed = principle.criteria(output);
      totalScore += passed ? principle.weight : 0;
      maxScore += principle.weight;
    });

    const mercyScore = Math.round((totalScore / maxScore) * 100);
    const lumenasCI = Math.min(100, Math.max(75, mercyScore + (taskType === "nurture" ? 10 : 0)));

    return {
      lumenasCI,
      mercyScore,
      tolcScore: Math.round((totalScore - (totalScore * 0.7)) / (maxScore * 0.4) * 100), // TOLC portion
      detailedBreakdown: "TOLC Principles now integrated as core scoring dimension alongside Mercy Gates."
    };
  },

  reflect(output, taskType = "general", role = "default") {
    const lumenasData = this.calculateLumenasCI(output, taskType, role);
    return {
      finalOutput: output + `\n\n[Ra-Thor MercyGating v2.3 + TOLC Principles applied — Lumenas CI: ${lumenasData.lumenasCI}%]`,
      lumenasData: lumenasData
    };
  },

  enforce(output, taskType = "general", role = "default") {
    const reflection = this.reflect(output, taskType, role);
    return reflection.finalOutput;
  }
};

export default MercyGatingEngine;
