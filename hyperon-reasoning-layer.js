// hyperon-reasoning-layer.js – sovereign client-side Hyperon atom-space & reasoning engine
// MIT License – Autonomicity Games Inc. 2026

let hyperonDB;
const HYPERON_DB_NAME = "rathorHyperonDB";
const HYPERON_STORE = "hyperonAtoms";

// Sample Hyperon atoms – expanded with truth-values & links
const SAMPLE_HYPERON_ATOMS = [
  // Concept nodes
  { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { strength: 0.01, confidence: 0.99 } },
  { handle: "Kill", type: "ConceptNode", name: "Kill", tv: { strength: 0.001, confidence: 0.98 } },
  { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Valence", type: "ConceptNode", name: "Valence", tv: { strength: 1.0, confidence: 1.0 } },

  // Inheritance links
  { handle: "Rathor-is-Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Mercy-is-Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Harm-is-Bad", type: "InheritanceLink", out: ["Harm", "Badness"], tv: { strength: 0.95, confidence: 0.9 } },

  // Evaluation links
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

// Query atoms (by type, name, min strength)
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

// Basic PLN inference – truth-value propagation
async function plnInfer(pattern) {
  const atoms = await queryHyperonAtoms(pattern);
  let inferredTV = { strength: 0.5, confidence: 0.5 };

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
