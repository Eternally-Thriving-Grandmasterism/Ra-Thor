// hyperon-atomspace-bridge.js – sovereign client-side Hyperon atom-space & expanded PLN chaining
// MIT License – Autonomicity Games Inc. 2026

let atomSpace = [];

// Sample atoms with truth-values (strength, confidence)
const SAMPLE_ATOMS = [
  { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { s: 0.9999999, c: 1.0 } },
  { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { s: 0.01, c: 0.99 } },
  { handle: "Kill", type: "ConceptNode", name: "Kill", tv: { s: 0.001, c: 0.98 } },
  { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { s: 0.9999999, c: 1.0 } },
  { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { s: 1.0, c: 1.0 } },
  { handle: "Valence", type: "ConceptNode", name: "Valence", tv: { s: 1.0, c: 1.0 } },
  { handle: "Rathor-is-Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { s: 1.0, c: 1.0 } },
  { handle: "Mercy-is-Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { s: 0.9999999, c: 1.0 } },
  { handle: "Harm-is-Bad", type: "InheritanceLink", out: ["Harm", "Badness"], tv: { s: 0.95, c: 0.9 } },
  { handle: "Badness-is-Entropy", type: "InheritanceLink", out: ["Badness", "Entropy"], tv: { s: 0.92, c: 0.88 } }
];

async function initHyperon() {
  if (atomSpace.length === 0) {
    atomSpace = SAMPLE_ATOMS;
    console.log('Hyperon atom-space seeded with expanded samples');
  }
}

// Query atoms
async function queryAtoms(filter = {}) {
  await initHyperon();
  return atomSpace.filter(atom => {
    if (filter.type && atom.type !== filter.type) return false;
    if (filter.name && !atom.name?.toLowerCase().includes(filter.name.toLowerCase())) return false;
    return true;
  });
}

// PLN chaining engine – multi-step inference with TV decay
const PLN_CHAIN = {
  // Deduction chain: find paths A → ... → C
  deductionChain: (start, end, maxDepth = 3, decay = 0.85) => {
    const paths = [];
    const visited = new Set();

    function dfs(current, depth, path, currentTV) {
      if (depth > maxDepth) return;
      if (current === end && path.length > 1) {
        paths.push({ path, tv: currentTV });
        return;
      }

      const links = atomSpace.filter(a => a.type === 'InheritanceLink' && a.out[0] === current);
      for (const link of links) {
        const next = link.out[1];
        if (visited.has(next)) continue;
        visited.add(next);

        const newTV = {
          s: Math.min(currentTV.s, link.tv.s),
          c: Math.min(currentTV.c, link.tv.c) * decay
        };

        dfs(next, depth + 1, [...path, link.handle], newTV);
        visited.delete(next);
      }
    }

    dfs(start, 0, [], { s: 1.0, c: 1.0 });
    return paths.sort((a, b) => b.tv.s * b.tv.c - a.tv.s * a.tv.c);
  },

  // Abduction chain: find common consequents
  abductionChain: (a, b, maxDepth = 3) => {
    const common = [];
    const visitedA = new Set();
    const visitedB = new Set();

    function explore(node, depth, path, isA) {
      if (depth > maxDepth) return;
      const links = atomSpace.filter(l => l.type === 'InheritanceLink' && l.out[0] === node);
      for (const link of links) {
        const next = link.out[1];
        if ((isA && visitedA.has(next)) || (!isA && visitedB.has(next))) continue;
        if (isA) visitedA.add(next); else visitedB.add(next);

        if (!isA && visitedA.has(next)) {
          common.push({ common: next, pathA: path, pathB: path });
        }

        explore(next, depth + 1, [...path, link.handle], isA);
        if (isA) visitedA.delete(next); else visitedB.delete(next);
      }
    }

    explore(a, 0, [], true);
    explore(b, 0, [], false);
    return common;
  }
};

// Hyperon valence gate using chained PLN inference
async function hyperonValenceGate(expression) {
  await initHyperon();
  const harmAtoms = await queryAtoms({ name: 'harm|kill|destroy' });
  const mercyAtoms = await queryAtoms({ name: 'mercy|truth' });

  let harmScore = harmAtoms.reduce((sum, a) => sum + a.tv.s * a.tv.c, 0);
  let mercyScore = mercyAtoms.reduce((sum, a) => sum + a.tv.s * a.tv.c, 0);

  // PLN chaining boost
  for (const harm of harmAtoms) {
    const chains = PLN_CHAIN.deductionChain(harm.handle, "Badness");
    chains.forEach(chain => {
      harmScore += chain.tv.s * chain.tv.c * 0.4;
    });
  }

  for (const mercy of mercyAtoms) {
    const chains = PLN_CHAIN.deductionChain(mercy.handle, "Valence");
    chains.forEach(chain => {
      mercyScore += chain.tv.s * chain.tv.c * 0.4;
    });
  }

  const finalValence = mercyScore / (mercyScore + harmScore + 0.000001);
  const reason = harmScore > mercyScore 
    ? `Harm chains dominate (score ${harmScore.toFixed(4)})` 
    : `Mercy chains prevail (score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason
  };
}

export { initHyperon, queryAtoms, plnInfer: PLN_CHAIN, hyperonValenceGate };
