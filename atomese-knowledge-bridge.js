// atomese-knowledge-bridge.js – sovereign client-side Atomese knowledge graph layer
// MIT License – Autonomicity Games Inc. 2026

let atomeseDB;
const ATOMESE_DB_NAME = "rathorAtomeseDB";
const ATOMESE_STORE = "atoms";

// Sample Atomese atoms – real structure with truth-values (strength, confidence)
const SAMPLE_ATOMS = [
  { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { strength: 0.01, confidence: 0.99 } },
  { handle: "Kill", type: "ConceptNode", name: "Kill", tv: { strength: 0.001, confidence: 0.98 } },
  { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { strength: 0.9999999, confidence: 1.0 } },
  { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Rathor-is-Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { strength: 1.0, confidence: 1.0 } },
  { handle: "Harm-is-Bad", type: "EvaluationLink", out: ["Badness", "Harm"], tv: { strength: 0.95, confidence: 0.9 } },
  { handle: "Mercy-is-Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { strength: 0.9999999, confidence: 1.0 } }
];

// Init IndexedDB + seed samples if empty
async function initAtomeseDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(ATOMESE_DB_NAME, 1);
    req.onupgradeneeded = evt => {
      const db = evt.target.result;
      if (!db.objectStoreNames.contains(ATOMESE_STORE)) {
        const store = db.createObjectStore(ATOMESE_STORE, { keyPath: "handle" });
        store.createIndex("type", "type");
        store.createIndex("name", "name");
      }
    };
    req.onsuccess = async evt => {
      atomeseDB = evt.target.result;
      const tx = atomeseDB.transaction(ATOMESE_STORE, "readwrite");
      const store = tx.objectStore(ATOMESE_STORE);
      const countReq = store.count();
      countReq.onsuccess = async () => {
        if (countReq.result === 0) {
          SAMPLE_ATOMS.forEach(atom => store.add(atom));
        }
        resolve(atomeseDB);
      };
    };
    req.onerror = () => reject(req.error);
  });
}

// Add / update atom
async function addAtom(atom) {
  const db = await initAtomeseDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(ATOMESE_STORE, "readwrite");
    const store = tx.objectStore(ATOMESE_STORE);
    store.put(atom);
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

// Query atoms (by type, name, min strength)
async function queryAtoms(filter = {}) {
  const db = await initAtomeseDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(ATOMESE_STORE, "readonly");
    const store = tx.objectStore(ATOMESE_STORE);
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

// Atomese valence gate – TV-weighted harm/mercy detection
async function atomeseValenceGate(expression) {
  const atoms = await queryAtoms();
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
