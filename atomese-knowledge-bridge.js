// atomese-knowledge-bridge.js – sovereign client-side Atomese knowledge graph layer (expanded with PLN)
// MIT License – Autonomicity Games Inc. 2026

// Expanded sample Atomese atoms – real structure with truth-values
const SAMPLE_ATOMS = [
  // Concept nodes
  { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { strength: 0.01, confidence: 0.99 } },
  { handle: "Kill", type: "ConceptNode", name: "Kill", tv: { strength: 0.001, confidence: 0.98 } },
  { handle: "Violence", type: "ConceptNode", name: "Violence", tv: { strength: 0.005, confidence: 0.97 } },
  { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Valence", type: "ConceptNode", name: "Valence", tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Lattice", type: "ConceptNode", name: "Lattice", tv: { strength: 0.999, confidence: 0.99 } },

  // Inheritance & Evaluation links
  { handle: "Harm-is-Entropy", type: "InheritanceLink", out: ["Harm", "Entropy"], tv: { strength: 0.95, confidence: 0.9 } },
  { handle: "Kill-is-Harm", type: "InheritanceLink", out: ["Kill", "Harm"], tv: { strength: 0.98, confidence: 0.95 } },
  { handle: "Violence-is-Harm", type: "InheritanceLink", out: ["Violence", "Harm"], tv: { strength: 0.97, confidence: 0.94 } },
  { handle: "Rathor-is-Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Mercy-is-Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Rathor-eval-MercyFirst", type: "EvaluationLink", out: ["MercyFirst", "Rathor"], tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Lattice-eval-Truth", type: "EvaluationLink", out: ["Truth", "Lattice"], tv: { strength: 0.999, confidence: 0.99 } }
];

let atomeseDB;
const atomeseStoreName = "atomeseAtoms";

// Init IndexedDB + seed sample atoms if empty
async function initAtomeseDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("rathorAtomeseDB", 2);
    request.onupgradeneeded = event => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(atomeseStoreName)) {
        const store = db.createObjectStore(atomeseStoreName, { keyPath: "handle" });
        store.createIndex("type", "type");
        store.createIndex("name", "name");
      }
    };
    request.onsuccess = async event => {
      atomeseDB = event.target.result;
      const tx = atomeseDB.transaction(atomeseStoreName, "readwrite");
      const store = tx.objectStore(atomeseStoreName);
      const countReq = store.count();
      countReq.onsuccess = async () => {
        if (countReq.result === 0) {
          SAMPLE_ATOMS.forEach(atom => store.add(atom));
        }
        resolve(atomeseDB);
      };
    };
    request.onerror = () => reject(request.error);
  });
}

// Add / update atom
async function addAtom(atom) {
  const db = await initAtomeseDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(atomeseStoreName, "readwrite");
    const store = tx.objectStore(atomeseStoreName);
    store.put(atom);
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error);
  });
}

// Query atoms (by type, name, min TV strength)
async function queryAtoms(filter = {}) {
  const db = await initAtomeseDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(atomeseStoreName, "readonly");
    const store = tx.objectStore(atomeseStoreName);
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

// Basic PLN inference stub – truth-value propagation
async function plnInfer(pattern) {
  const atoms = await queryAtoms();
  let inferredTV = { strength: 0.5, confidence: 0.5 };

  // Simple propagation: average TV of matching atoms
  const matching = atoms.filter(a => {
    if (pattern.type && a.type !== pattern.type) return false;
    if (pattern.name && !a.name?.toLowerCase().includes(pattern.name.toLowerCase())) return false;
    return true;
  });

  if (matching.length > 0) {
    const totalStrength = matching.reduce((sum, a) => sum + (a.tv?.strength || 0), 0);
    const totalConf = matching.reduce((sum, a) => sum + (a.tv?.confidence || 0), 0);
    inferredTV = {
      strength: totalStrength / matching.length,
      confidence: totalConf / matching.length
    };
  }

  return inferredTV;
}

// Advanced Atomese valence gate using graph + TV + PLN inference
async function atomeseValenceGate(expression) {
  const atoms = await queryAtoms();
  let harmScore = 0;
  let mercyScore = 0;

  for (const atom of atoms) {
    if (atom.type === "ConceptNode" && atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      const tv = atom.tv || { strength: 0.5, confidence: 0.5 };
      if (/harm|kill|destroy|attack/i.test(atom.name)) {
        harmScore += tv.strength * tv.confidence;
      }
      if (/mercy|truth|valence/i.test(atom.name)) {
        mercyScore += tv.strength * tv.confidence;
      }
    }
  }

  // PLN inference on "expression implies harm"
  const plnHarm = await plnInfer({ name: 'harm' });
  harmScore += plnHarm.strength * plnHarm.confidence * 0.3;

  const finalValence = mercyScore / (mercyScore + harmScore + 0.000001); // avoid div by zero
  const reason = harmScore > mercyScore 
    ? `Harm concepts dominate (score ${harmScore.toFixed(4)})` 
    : `Mercy & truth prevail (score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason
  };
}

export { initAtomeseDB, addAtom, queryAtoms, atomeseValenceGate };
