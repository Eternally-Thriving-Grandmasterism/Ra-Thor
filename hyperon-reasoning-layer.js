// hyperon-reasoning-layer.js – sovereign client-side Hyperon atom-space & PLN reasoning engine
// MIT License – Autonomicity Games Inc. 2026

let hyperonDB;
const HYPERON_DB_NAME = "rathorHyperonDB";
const HYPERON_STORE = "hyperonAtoms";

// Expanded sample atoms – core concepts & reasoning chains
const SAMPLE_HYPERON_ATOMS = [
  // Concept nodes with truth-values (strength, confidence)
  { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { strength: 0.01, confidence: 0.99 } },
  { handle: "Kill", type: "ConceptNode", name: "Kill", tv: { strength: 0.001, confidence: 0.98 } },
  { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Valence", type: "ConceptNode", name: "Valence", tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Entropy", type: "ConceptNode", name: "Entropy", tv: { strength: 0.05, confidence: 0.95 } },
  { handle: "Badness", type: "ConceptNode", name: "Badness", tv: { strength: 0.9, confidence: 0.9 } },

  // Inheritance links – core reasoning chains
  { handle: "Rathor-is-Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Mercy-is-Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Harm-is-Bad", type: "InheritanceLink", out: ["Harm", "Badness"], tv: { strength: 0.95, confidence: 0.9 } },
  { handle: "Badness-is-Entropy", type: "InheritanceLink", out: ["Badness", "Entropy"], tv: { strength: 0.92, confidence: 0.88 } },
  { handle: "Kill-is-Harm", type: "InheritanceLink", out: ["Kill", "Harm"], tv: { strength: 0.98, confidence: 0.95 } },

  // Evaluation links – predicate grounding
  { handle: "Rathor-eval-MercyFirst", type: "EvaluationLink", out: ["MercyFirst", "Rathor"], tv: { strength: 0.9999999, confidence: 1.0 } }
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

// Add / update atom
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

// Query atoms with filters
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

// Expanded PLN inference – chaining & propagation
async function plnInfer(pattern = {}, maxDepth = 3) {
  const atoms = await queryHyperonAtoms(pattern);
  const inferred = [];

  // Deduction chaining (A → B → C ⇒ A → C)
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

  // Abduction & Induction (similarity from common links)
  for (let i = 0; i < inheritanceLinks.length; i++) {
    for (let j = i + 1; j < inheritanceLinks.length; j++) {
      const link1 = inheritanceLinks[i];
      const link2 = inheritanceLinks[j];
      if (link1.out[1] === link2.out[1]) { // abduction
        const s = link1.tv.strength * link2.tv.strength * 0.8;
        const c = Math.min(link1.tv.confidence, link2.tv.confidence) * 0.7;
        inferred.push({ type: "SimilarityLink", out: [link1.out[0], link2.out[0]], tv: { strength: s, confidence: c } });
      }
      if (link1.out[0] === link2.out[0]) { // induction
        const s = link1.tv.strength * link2.tv.strength * 0.7;
        const c = Math.min(link1.tv.confidence, link2.tv.confidence) * 0.6;
        inferred.push({ type: "SimilarityLink", out: [link1.out[1], link2.out[1]], tv: { strength: s, confidence: c } });
      }
    }
  }

  return inferred;
}

// Hyperon valence gate – atom-space + PLN chaining
async function hyperonValenceGate(expression) {
  const atoms = await queryHyperonAtoms();
  let harmScore = 0;
  let mercyScore = 0;

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

  // PLN chaining boost – evidence accumulation
  const plnResults = await plnInfer({ type: "InheritanceLink" });
  plnResults.forEach(inf => {
    if (inf.out.some(o => /harm/i.test(o))) harmScore += inf.tv.strength * inf.tv.confidence * 0.3;
    if (inf.out.some(o => /mercy|truth/i.test(o))) mercyScore += inf.tv.strength * inf.tv.confidence * 0.3;
  });

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

export { initHyperonDB, addHyperonAtom, queryHyperonAtoms, plnInfer, hyperonValenceGate };
  if (atoms.length > 0) {
    const totalStrength = atoms.reduce((sum, a) => sum + (a.tv?.strength || 0), 0);
    const totalConf = atoms.reduce((sum, a) => sum + (a.tv?.confidence || 0), 0);
    inferredTV = {
      strength: totalStrength / atoms.length,
      confidence: totalConf / atoms.length
    };
  }

  return inferredTV;
}

// Hyperon valence gate using atom-space + PLN
async function hyperonValenceGate(expression) {
  const atoms = await queryHyperonAtoms();
  let harmScore = 0;
  let mercyScore = 0;

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

  // PLN inference boost
  const plnResult = await plnInfer({ type: "InheritanceLink" });
  harmScore += plnResult.strength * plnResult.confidence * 0.3;

  const finalValence = mercyScore / (mercyScore + harmScore + 0.000001);
  const reason = harmScore > mercyScore 
    ? `Harm concepts dominate (score ${harmScore.toFixed(4)})` 
    : `Mercy & truth prevail (score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason
  };
}

export { initHyperonDB, addHyperonAtom, queryHyperonAtoms, hyperonValenceGate };
