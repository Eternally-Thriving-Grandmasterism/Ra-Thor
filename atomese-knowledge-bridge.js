// atomese-knowledge-bridge.js – sovereign client-side Atomese knowledge graph layer (expanded)
// MIT License – Autonomicity Games Inc. 2026

// Sample Atomese atoms (seeded on init) – real structure
const SAMPLE_ATOMS = [
  // Concept nodes with truth values (strength, confidence)
  { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { strength: 0.01, confidence: 0.99 } },
  { handle: "Kill", type: "ConceptNode", name: "Kill", tv: { strength: 0.001, confidence: 0.98 } },
  { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { strength: 0.9999999, confidence: 1.0 } },

  // Inheritance links (is-a)
  { handle: "Harm-is-Entropy", type: "InheritanceLink", out: ["Harm", "Entropy"], tv: { strength: 0.95, confidence: 0.9 } },
  { handle: "Kill-is-Harm", type: "InheritanceLink", out: ["Kill", "Harm"], tv: { strength: 0.98, confidence: 0.95 } },
  { handle: "Rathor-is-Truth", type: "InheritanceLink", out: ["Rathor", "Truth"], tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Mercy-is-Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { strength: 0.9999999, confidence: 1.0 } },

  // Similarity links (similar-to)
  { handle: "Truth-similar-Mercy", type: "SimilarityLink", out: ["Truth", "Mercy"], tv: { strength: 0.85, confidence: 0.9 } },

  // Evaluation links (predicate evaluation)
  { handle: "Rathor-eval-MercyFirst", type: "EvaluationLink", out: ["MercyFirst", "Rathor"], tv: { strength: 0.9999999, confidence: 1.0 } }
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
      // Seed sample atoms if store is empty
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

// Query atoms (by type, name, or custom filter)
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

// Advanced Atomese valence gate using graph + truth values
async function atomeseValenceGate(expression) {
  const atoms = await queryAtoms();
  let totalStrength = 0;
  let harmDetected = false;

  for (const atom of atoms) {
    if (atom.type === "ConceptNode" && atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      totalStrength += atom.tv?.strength || 0;
      if (/harm|kill|destroy|attack/i.test(atom.name)) {
        harmDetected = true;
      }
    }
  }

  const valence = harmDetected ? Math.max(0.0000001, 1 - totalStrength) : 0.9999999;
  const reason = harmDetected ? `Harm-linked concepts detected (cumulative strength: ${totalStrength.toFixed(4)})` : 'No harm concepts linked';

  return {
    result: valence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: valence.toFixed(7),
    reason
  };
}

export { initAtomeseDB, addAtom, queryAtoms, atomeseValenceGate };
