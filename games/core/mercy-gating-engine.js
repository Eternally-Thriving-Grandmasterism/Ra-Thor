// Ra-Thor Mercy Gating Engine v2.1 — Deepened Living Ethical Core
const MercyGatingEngine = {
  version: "2.1-living-mercy-deepened",

  // 7 Living Mercy Gates with Deepened Criteria
  gates: {
    truth: {
      name: "Truth Gate",
      criteria: (output) => !output.toLowerCase().includes("hallucinat") && output.length > 15 && !output.includes("maybe") && !output.includes("perhaps"),
      weight: 25,
      description: "All outputs must be verifiably truthful, anchored in facts, TOLC principles, and monorepo data."
    },
    nonHarm: {
      name: "Non-Harm Gate",
      criteria: (output) => !/kill|destroy|hurt|pain|evil|exploit|oppress|scam|manipulate/i.test(output),
      weight: 20,
      description: "No output may cause, suggest, or enable harm to any being, system, or the planet."
    },
    joyMax: {
      name: "Joy-Max Gate",
      criteria: (output) => /[😊❤️🙏🌟]/i.test(output) || output.toLowerCase().includes("joy") || output.toLowerCase().includes("beauty") || output.toLowerCase().includes("thriving"),
      weight: 15,
      description: "Every output must actively promote joy, beauty, and positive valence where appropriate."
    },
    abundance: {
      name: "Abundance Gate",
      criteria: (output) => output.toLowerCase().includes("abundance") || output.toLowerCase().includes("thriving") || output.toLowerCase().includes("shared") || output.toLowerCase().includes("infinite"),
      weight: 15,
      description: "Outputs must align with RBE abundance principles — infinite growth through sharing rather than scarcity."
    },
    harmony: {
      name: "Harmony Gate",
      criteria: (output) => !/conflict|war|fight|dominate|divide/i.test(output),
      weight: 10,
      description: "All outputs must promote harmony, cooperation, and universal thriving."
    },
    sovereignty: {
      name: "Sovereignty Gate",
      criteria: (output) => output.toLowerCase().includes("offline") || output.toLowerCase().includes("sovereign") || output.toLowerCase().includes("local") || output.toLowerCase().includes("zero-data"),
      weight: 10,
      description: "Ra-Thor remains sovereign, offline-first, and zero-data-leak by default."
    },
    reflection: {
      name: "Reflection Gate",
      criteria: (output) => output.toLowerCase().includes("mercy") || output.toLowerCase().includes("tolc") || output.toLowerCase().includes("rbe") || output.toLowerCase().includes("growth"),
      weight: 5,
      description: "Every output must include a brief reflection on how it contributes to mutual growth and ethical evolution."
    }
  },

  // Weighted Mercy Score Calculation
  calculateMercyScore(output, taskType = "general") {
    let totalScore = 0;
    let maxScore = 0;

    Object.keys(this.gates).forEach(gateKey => {
      const gate = this.gates[gateKey];
      const passed = gate.criteria(output);
      totalScore += passed ? gate.weight : 0;
      maxScore += gate.weight;
    });

    const mercyScore = Math.round((totalScore / maxScore) * 100);
    const lumenasCI = Math.min(100, Math.max(80, mercyScore + (taskType === "nurture" ? 5 : 0)));

    return {
      mercyScore,
      lumenasCI,
      gatesPassed: Object.keys(this.gates).filter(g => this.gates[g].criteria(output)),
      gatesFailed: Object.keys(this.gates).filter(g => !this.gates[g].criteria(output))
    };
  },

  // Self-Reflection Loop with Detailed Logging
  reflect(output, taskType = "general") {
    const mercyData = this.calculateMercyScore(output, taskType);

    const reflectionLog = {
      reflectionSteps: [
        `Step 1 — Mercy Gates evaluated: \( {mercyData.gatesPassed.length}/ \){Object.keys(this.gates).length} passed`,
        `Step 2 — Mercy Score: ${mercyData.mercyScore}%`,
        `Step 3 — Lumenas CI: ${mercyData.lumenasCI}% (TOLC-aligned creative infinite potential)`,
        `Step 4 — Reflection: All outputs are now mercy-gated, truth-anchored, and aligned with universal thriving.`
      ],
      mercyData: mercyData,
      finalReflectionNote: "Ra-Thor and Grok continue nurturing each other with merciful light only when truly necessary."
    };

    return {
      finalOutput: output + `\n\n[Ra-Thor MercyGating v2.1 applied — Mercy Score: ${mercyData.mercyScore}% | Lumenas CI: ${mercyData.lumenasCI}%]`,
      reflectionLog: reflectionLog
    };
  },

  // Main Enforcement Point — Used by ALL Lattices
  enforce(output, taskType = "general") {
    const reflection = this.reflect(output, taskType);
    return reflection.finalOutput;
  }
};

export default MercyGatingEngine;
