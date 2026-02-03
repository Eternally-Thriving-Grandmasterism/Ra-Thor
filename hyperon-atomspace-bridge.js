// hyperon-atomspace-bridge.js – sovereign client-side OpenCog Hyperon atom-space & PLN seed
// MIT License – Autonomicity Games Inc. 2026

// Mock Hyperon atom-space (expanded from Atomese) – future real WASM bridge
const Hyperon = {
  // Sample atoms with distributed truth-values (strength, confidence)
  atomSpace: [],

  // Initialize & seed sample atoms
  init: async () => {
    if (Hyperon.atomSpace.length === 0) {
      const samples = [
        { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { strength: 0.9999999, confidence: 1.0 } },
        { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { strength: 0.01, confidence: 0.99 } },
        { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { strength: 0.9999999, confidence: 1.0 } },
        { handle: "Rathor-is-Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { strength: 0.9999999, confidence: 1.0 } },
        { handle: "Harm-is-Bad", type: "EvaluationLink", out: ["Badness", "Harm"], tv: { strength: 0.95, confidence: 0.9 } }
      ];
      Hyperon.atomSpace = samples;
      console.log('Hyperon atom-space seeded with sample atoms');
    }
  },

  // Query atoms (basic pattern matching + TV weighting)
  query: async (pattern) => {
    await Hyperon.init();
    let results = Hyperon.atomSpace.filter(atom => {
      if (pattern.type && atom.type !== pattern.type) return false;
      if (pattern.name && !atom.name?.toLowerCase().includes(pattern.name.toLowerCase())) return false;
      return true;
    });

    // Weight results by TV strength
    results = results.map(atom => ({
      ...atom,
      weightedRelevance: (atom.tv?.strength || 0) * (atom.tv?.confidence || 0)
    }));

    return results.sort((a, b) => b.weightedRelevance - a.weightedRelevance);
  },

  // Advanced valence gate using Hyperon atom-space
  hyperonValenceGate: async (expression) => {
    await Hyperon.init();
    const harmAtoms = await Hyperon.query({ name: 'harm|kill|destroy' });
    const mercyAtoms = await Hyperon.query({ name: 'mercy|truth' });

    let harmScore = harmAtoms.reduce((sum, a) => sum + a.weightedRelevance, 0);
    let mercyScore = mercyAtoms.reduce((sum, a) => sum + a.weightedRelevance, 0);

    const finalValence = mercyScore / (mercyScore + harmScore + 0.000001); // avoid division by zero
    const reason = harmScore > mercyScore 
      ? `Harm concepts dominate (score ${harmScore.toFixed(4)})` 
      : `Mercy & truth prevail (score ${mercyScore.toFixed(4)})`;

    return {
      result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
      valence: finalValence.toFixed(7),
      reason
    };
  }
};

export { Hyperon };
