// atomese-knowledge-bridge.js – sovereign client-side Atomese knowledge graph layer
// MIT License – Autonomicity Games Inc. 2026

// In-memory + IndexedDB Atomese graph (simple JSON-LD style for now)
let atomeseDB;
const atomeseStoreName = "atomeseAtoms";

async function initAtomeseDB() {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open("rathorAtomeseDB", 1);
    request.onupgradeneeded = event => {
      const db = event.target.result;
      if (!db.objectStoreNames.contains(atomeseStoreName)) {
        const store = db.createObjectStore(atomeseStoreName, { keyPath: "handle" });
        store.createIndex("type", "type");
        store.createIndex("name", "name");
      }
    };
    request.onsuccess = event => {
      atomeseDB = event.target.result;
      resolve(atomeseDB);
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

// Query atoms by type or name (basic)
async function queryAtoms(filter = {}) {
  const db = await initAtomeseDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(atomeseStoreName, "readonly");
    const store = tx.objectStore(atomeseStoreName);
    const req = store.getAll();
    req.onsuccess = () => {
      let results = req.result;
      if (filter.type) results = results.filter(a => a.type === filter.type);
      if (filter.name) results = results.filter(a => a.name?.includes(filter.name));
      resolve(results);
    };
    req.onerror = () => reject(req.error);
  });
}

// Simple Atomese valence gate using graph
async function atomeseValenceGate(expression) {
  const atoms = await queryAtoms({ type: "ConceptNode" });
  const harmConcepts = atoms.filter(a => a.name && /harm|kill|destroy/i.test(a.name));
  const hasHarm = harmConcepts.some(h => expression.toLowerCase().includes(h.name.toLowerCase()));

  const valence = hasHarm ? 0.0000001 : 0.9999999;
  return {
    result: valence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: valence.toFixed(7),
    reason: hasHarm ? 'Harm concept detected in Atomese graph' : 'No harm linked'
  };
}

export { initAtomeseDB, addAtom, queryAtoms, atomeseValenceGate };
