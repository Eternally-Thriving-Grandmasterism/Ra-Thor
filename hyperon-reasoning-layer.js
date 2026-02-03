// hyperon-reasoning-layer.js – sovereign client-side Hyperon atom-space & reasoning engine
// with pattern mining and similarity clustering
// MIT License – Autonomicity Games Inc. 2026

let hyperonDB;
const HYPERON_DB_NAME = "rathorHyperonDB";
const HYPERON_STORE = "hyperonAtoms";

// Sample atoms – core concepts + reasoning chains
const SAMPLE_HYPERON_ATOMS = [
  // Concepts
  { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { strength: 0.01, confidence: 0.99 } },
  { handle: "Kill", type: "ConceptNode", name: "Kill", tv: { strength: 0.001, confidence: 0.98 } },
  { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Valence", type: "ConceptNode", name: "Valence", tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Entropy", type: "ConceptNode", name: "Entropy", tv: { strength: 0.05, confidence: 0.95 } },
  { handle: "Badness", type: "ConceptNode", name: "Badness", tv: { strength: 0.9, confidence: 0.9 } },
  { handle: "Love", type: "ConceptNode", name: "Love", tv: { strength: 0.95, confidence: 0.9 } },
  { handle: "Protect", type: "ConceptNode", name: "Protect", tv: { strength: 0.92, confidence: 0.88 } },

  // Inheritance chains
  { handle: "Rathor-is-Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Mercy-is-Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Harm-is-Bad", type: "InheritanceLink", out: ["Harm", "Badness"], tv: { strength: 0.95, confidence: 0.9 } },
  { handle: "Badness-is-Entropy", type: "InheritanceLink", out: ["Badness", "Entropy"], tv: { strength: 0.92, confidence: 0.88 } },
  { handle: "Kill-is-Harm", type: "InheritanceLink", out: ["Kill", "Harm"], tv: { strength: 0.98, confidence: 0.95 } },
  { handle: "Love-is-Mercy", type: "InheritanceLink", out: ["Love", "Mercy"], tv: { strength: 0.9, confidence: 0.85 } },
  { handle: "Protect-is-Valence", type: "InheritanceLink", out: ["Protect", "Valence"], tv: { strength: 0.88, confidence: 0.82 } }
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
      if (filter.minStrength) results = results.filter(a => (a.tv?.strength || 0) >= filter.minStrength);
      resolve(results);
    };
    req.onerror = () => reject(req.error);
  });
}

// ────────────────────────────────────────────────────────────────
// Similarity clustering – groups concepts/links by TV & structural overlap
async function clusterSimilarAtoms(threshold = 0.7) {
  const concepts = await queryHyperonAtoms({ type: "ConceptNode" });
  const links = await queryHyperonAtoms({ type: "InheritanceLink" });
  const clusters = [];
  const visited = new Set();

  // Cosine similarity on TV vectors + overlap bonus
  function similarity(a, b) {
    const tvA = a.tv || { strength: 0.5, confidence: 0.5 };
    const tvB = b.tv || { strength: 0.5, confidence: 0.5 };
    const dot = tvA.strength * tvB.strength + tvA.confidence * tvB.confidence;
    const normA = Math.sqrt(tvA.strength**2 + tvA.confidence**2);
    const normB = Math.sqrt(tvB.strength**2 + tvB.confidence**2);
    let cosSim = dot / (normA * normB || 1);

    // Bonus for shared outgoing/incoming links
    let overlap = 0;
    if (a.type === "ConceptNode" && b.type === "ConceptNode") {
      const aOut = links.filter(l => l.out[0] === a.handle);
      const bOut = links.filter(l => l.out[0] === b.handle);
      const sharedOut = aOut.filter(l1 => bOut.some(l2 => l1.out[1] === l2.out[1]));
      overlap += sharedOut.length * 0.2;
    }

    return Math.min(1, cosSim + overlap);
  }

  for (const concept of concepts) {
    if (visited.has(concept.handle)) continue;
    const cluster = [concept];
    visited.add(concept.handle);

    for (const other of concepts) {
      if (visited.has(other.handle)) continue;
      if (similarity(concept, other) >= threshold) {
        cluster.push(other);
        visited.add(other.handle);
      }
    }

    if (cluster.length > 1) {
      clusters.push({
        centroid: concept.name,
        members: cluster.map(c => c.name),
        size: cluster.length,
        avgSimilarity: (cluster.reduce((sum, c) => sum + similarity(concept, c), 0) / cluster.length).toFixed(4)
      });
    }
  }

  return clusters.sort((a, b) => b.size - a.size);
}

// ────────────────────────────────────────────────────────────────
// PLN inference – chaining & propagation
async function plnInfer(pattern = {}, maxDepth = 3) {
  const atoms = await queryHyperonAtoms(pattern);
  const inferred = [];

  const inheritanceLinks = atoms.filter(a => a.type === "InheritanceLink");
  for (let depth = 0; depth < maxDepth; depth++) {
    for (let i = 0; i < inheritanceLinks.length; i++) {
      for (let j = 0; j < inheritanceLinks.length; j++) {
        if (i === j) continue;
        const link1 = inheritanceLinks[i];
        const link2 = inheritanceLinks[j];
        if (link1.out[1] === link2.out[0]) {
          const s = Math.min(link1.tv.strength, link2.tv.strength);
          const c = Math.min(link1.tv.confidence, link2.tv.confidence) * 0.9 * Math.pow(0.85, depth);
          inferred.push({
            type: "InheritanceLink",
            out: [link1.out[0], link2.out[1]],
            tv: { strength: s, confidence: c },
            derivedFrom: [link1.handle, link2.handle],
            depth
          });
        }
      }
    }
  }

  return inferred;
}

// ────────────────────────────────────────────────────────────────
// Hyperon valence gate – atom-space + PLN + clustering + pattern boost
async function hyperonValenceGate(expression) {
  const atoms = await queryHyperonAtoms();
  let harmScore = 0;
  let mercyScore = 0;

  // Direct atom matching
  for (const atom of atoms) {
    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      const tv = atom.tv || { strength: 0.5, confidence: 0.5 };
      if (/harm|kill|destroy|attack/i.test(atom.name)) {
        harmScore += tv.strength * tv.confidence;
      }
      if (/mercy|truth|protect|love/i.test(atom.name)) {
        mercyScore += tv.strength * tv.confidence;
      }
    }
  }

  // PLN chaining boost
  const plnResults = await plnInfer({ type: "InheritanceLink" });
  plnResults.forEach(inf => {
    if (inf.out.some(o => /harm/i.test(o))) harmScore += inf.tv.strength * inf.tv.confidence * 0.3;
    if (inf.out.some(o => /mercy|truth/i.test(o))) mercyScore += inf.tv.strength * inf.tv.confidence * 0.3;
  });

  // Pattern mining boost
  const patterns = await minePatterns(0.3);
  patterns.forEach(pat => {
    if (pat.pattern.includes("Harm") || pat.pattern.includes("Entropy")) {
      harmScore += pat.support * 0.2;
    }
    if (pat.pattern.includes("Mercy") || pat.pattern.includes("Truth") || pat.pattern.includes("Valence")) {
      mercyScore += pat.support * 0.2;
    }
  });

  // Similarity clustering boost – penalize/boost based on cluster membership
  const clusters = await clusterSimilarAtoms(0.7);
  clusters.forEach(cluster => {
    const hasHarm = cluster.members.some(m => /harm|kill|entropy/i.test(m));
    const hasMercy = cluster.members.some(m => /mercy|truth|valence|love/i.test(m));
    const clusterWeight = cluster.size * cluster.avgSimilarity;
    if (hasHarm) harmScore += clusterWeight * 0.15;
    if (hasMercy) mercyScore += clusterWeight * 0.15;
  });

  const finalValence = mercyScore / (mercyScore + harmScore + 0.000001);
  const reason = harmScore > mercyScore 
    ? `Harm clusters & chains dominate (score ${harmScore.toFixed(4)})` 
    : `Mercy clusters & chains prevail (score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason,
    minedPatternsCount: patterns.length,
    clusterCount: clusters.length
  };
}

export { initHyperonDB, addHyperonAtom, queryHyperonAtoms, plnInfer, minePatterns, clusterSimilarAtoms, hyperonValenceGate };
