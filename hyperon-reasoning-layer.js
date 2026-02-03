// hyperon-reasoning-layer.js – sovereign client-side Hyperon atom-space & PLN chaining engine
// MIT License – Autonomicity Games Inc. 2026

let hyperonDB;
const HYPERON_DB_NAME = "rathorHyperonDB";
const HYPERON_STORE = "hyperonAtoms";

// Sample atoms – core concepts & reasoning chains
const SAMPLE_HYPERON_ATOMS = [
  { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { s: 0.9999999, c: 1.0 } },
  { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { s: 0.01, c: 0.99 } },
  { handle: "Kill", type: "ConceptNode", name: "Kill", tv: { s: 0.001, c: 0.98 } },
  { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { s: 0.9999999, c: 1.0 } },
  { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { s: 1.0, c: 1.0 } },
  { handle: "Valence", type: "ConceptNode", name: "Valence", tv: { s: 1.0, c: 1.0 } },
  { handle: "Entropy", type: "ConceptNode", name: "Entropy", tv: { s: 0.05, c: 0.95 } },
  { handle: "Badness", type: "ConceptNode", name: "Badness", tv: { s: 0.9, c: 0.9 } },

  // Inheritance chains for chaining demo
  { handle: "Rathor-is-Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { s: 1.0, c: 1.0 } },
  { handle: "Mercy-is-Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { s: 0.9999999, c: 1.0 } },
  { handle: "Harm-is-Bad", type: "InheritanceLink", out: ["Harm", "Badness"], tv: { s: 0.95, c: 0.9 } },
  { handle: "Badness-is-Entropy", type: "InheritanceLink", out: ["Badness", "Entropy"], tv: { s: 0.92, c: 0.88 } },
  { handle: "Kill-is-Harm", type: "InheritanceLink", out: ["Kill", "Harm"], tv: { s: 0.98, c: 0.95 } }
];

async function initHyperonDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(HYPERON_DB_NAME, 1);
    req.onupgradeneeded = evt => {
      const db = evt.target.result;
      if (!db.objectStoreNames.contains(HYPERON_STORE)) {
        const store = db.createObjectStore(HYPERON_STORE, { keyPath: "handle" });
        store.createIndex("type", "type");
        store.createIndex("name", "name");
      }
    };
    req.onsuccess = async evt => {
      hyperonDB = evt.target.result;
      const tx = hyperonDB.transaction(HYPERON_STORE, "readwrite");
      const store = tx.objectStore(HYPERON_STORE);
      const countReq = store.count();
      countReq.onsuccess = async () => {
        if (countReq.result === 0) {
          SAMPLE_HYPERON_ATOMS.forEach(atom => store.add(atom));
        }
        resolve(hyperonDB);
      };
    };
    req.onerror = () => reject(req.error);
  });
}

async function addHyperonAtom(atom) {
  const db = await initHyperonDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HYPERON_STORE, "readwrite");
    const store = tx.objectStore(HYPERON_STORE);
    store.put(atom);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

async function queryHyperonAtoms(filter = {}) {
  const db = await initHyperonDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HYPERON_STORE, "readonly");
    const store = tx.objectStore(HYPERON_STORE);
    const req = store.getAll();
    req.onsuccess = () => {
      let results = req.result;
      if (filter.type) results = results.filter(a => a.type === filter.type);
      if (filter.name) results = results.filter(a => a.name?.toLowerCase().includes(filter.name.toLowerCase()));
      if (filter.minStrength) results = results.filter(a => (a.tv?.s || 0) >= filter.minStrength);
      resolve(results);
    };
    req.onerror = () => reject(req.error);
  });
}

// ────────────────────────────────────────────────────────────────
// PLN inference chaining – multi-step probabilistic deduction/abduction
async function plnChainInfer(startConcept, targetConcept = null, maxDepth = 4, decay = 0.88) {
  const inheritanceLinks = await queryHyperonAtoms({ type: "InheritanceLink" });
  const chains = [];
  const visited = new Set();

  async function dfs(current, depth, path, currentTV) {
    if (depth > maxDepth) return;
    if (targetConcept && current === targetConcept && path.length > 1) {
      chains.push({ path, tv: currentTV, length: path.length });
      return;
    }

    const outgoing = inheritanceLinks.filter(l => l.out[0] === current);
    for (const link of outgoing) {
      const next = link.out[1];
      if (visited.has(next)) continue;
      visited.add(next);

      const newTV = {
        s: Math.min(currentTV.s, link.tv.s),
        c: Math.min(currentTV.c, link.tv.c) * decay
      };

      await dfs(next, depth + 1, [...path, link.handle], newTV);
      visited.delete(next);
    }
  }

  // Start chaining from every known concept that matches startConcept pattern
  const startNodes = await queryHyperonAtoms({ name: startConcept });
  for (const start of startNodes) {
    await dfs(start.handle, 0, [], { s: 1.0, c: 1.0 });
  }

  // Sort by strength × confidence × inverse length (shorter stronger chains preferred)
  return chains.sort((a, b) => 
    (b.tv.s * b.tv.c / b.length) - (a.tv.s * a.tv.c / a.length)
  );
}

// ────────────────────────────────────────────────────────────────
// Hyperon valence gate – now with chaining boost
async function hyperonValenceGate(expression) {
  const atoms = await queryHyperonAtoms();
  let harmScore = 0;
  let mercyScore = 0;

  // Direct atom matching
  for (const atom of atoms) {
    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      const tv = atom.tv || { s: 0.5, c: 0.5 };
      if (/harm|kill|destroy|attack/i.test(atom.name)) {
        harmScore += tv.s * tv.c;
      }
      if (/mercy|truth|protect|love/i.test(atom.name)) {
        mercyScore += tv.s * tv.c;
      }
    }
  }

  // PLN chaining boost – long high-confidence chains amplify scores
  const harmChains = await plnChainInfer("Harm", "Entropy", 4);
  const mercyChains = await plnChainInfer("Mercy", "Valence", 4);

  harmChains.forEach(chain => {
    harmScore += chain.tv.s * chain.tv.c * 0.4 / chain.length;
  });

  mercyChains.forEach(chain => {
    mercyScore += chain.tv.s * chain.tv.c * 0.4 / chain.length;
  });

  const finalValence = mercyScore / (mercyScore + harmScore + 0.000001);
  const reason = harmScore > mercyScore 
    ? `Harm chaining dominates (${harmChains.length} chains, score ${harmScore.toFixed(4)})` 
    : `Mercy chaining prevails (${mercyChains.length} chains, score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason,
    harmChainsFound: harmChains.length,
    mercyChainsFound: mercyChains.length
  };
}

export { initHyperonDB, addHyperonAtom, queryHyperonAtoms, plnChainInfer, hyperonValenceGate };
