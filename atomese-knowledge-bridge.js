// atomese-knowledge-bridge.js – sovereign client-side Atomese knowledge grounding
// Persistent hypergraph with typed nodes/links, TV propagation, PLN grounding
// MIT License – Autonomicity Games Inc. 2026

let atomeseDB;
const ATOmese_DB_NAME = "rathorAtomeseDB";
const ATOmese_STORE = "atomeseAtoms";

// ────────────────────────────────────────────────────────────────
// Atomese Atom structure
class AtomeseAtom {
  constructor(handle, type, name = null, tv = { strength: 0.5, confidence: 0.5 }, sti = 0.1) {
    this.handle = handle; // unique string id
    this.type = type;     // ConceptNode, PredicateNode, InheritanceLink, EvaluationLink, SimilarityLink, ExecutionLink, VariableNode, etc.
    this.name = name;     // human-readable label (optional for links)
    this.tv = tv;         // truth value
    this.sti = sti;       // short-term importance (for attention)
    this.incoming = [];   // handles of atoms pointing to this one
    this.outgoing = [];   // handles of atoms this one points to
  }
}

// ────────────────────────────────────────────────────────────────
// Database & seed
async function initAtomeseDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(ATOmese_DB_NAME, 1);
    req.onupgradeneeded = evt => {
      const db = evt.target.result;
      if (!db.objectStoreNames.contains(ATOmese_STORE)) {
        const store = db.createObjectStore(ATOmese_STORE, { keyPath: "handle" });
        store.createIndex("type", "type");
        store.createIndex("name", "name");
      }
    };
    req.onsuccess = async evt => {
      atomeseDB = evt.target.result;
      const tx = atomeseDB.transaction(ATOmese_STORE, "readwrite");
      const store = tx.objectStore(ATOmese_STORE);
      const countReq = store.count();
      countReq.onsuccess = async () => {
        if (countReq.result === 0) {
          await seedAtomese();
        }
        resolve(atomeseDB);
      };
    };
    req.onerror = () => reject(req.error);
  });
}

async function seedAtomese() {
  const seed = [
    // Core concepts
    { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { strength: 0.9999999, confidence: 1.0 }, sti: 0.2 },
    { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { strength: 0.01, confidence: 0.99 }, sti: 0.05 },
    { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { strength: 0.9999999, confidence: 1.0 }, sti: 0.25 },
    { handle: "Rathor", type: "ConceptNode", name: "Rathor", tv: { strength: 1.0, confidence: 1.0 }, sti: 0.3 },
    { handle: "Valence", type: "ConceptNode", name: "Valence", tv: { strength: 1.0, confidence: 1.0 }, sti: 0.28 },

    // Links
    { handle: "Rathor→Mercy", type: "InheritanceLink", outgoing: ["Rathor", "Mercy"], tv: { strength: 1.0, confidence: 1.0 }, sti: 0.35 },
    { handle: "Mercy→Valence", type: "InheritanceLink", outgoing: ["Mercy", "Valence"], tv: { strength: 0.9999999, confidence: 1.0 }, sti: 0.32 },
    { handle: "Harm→Bad", type: "InheritanceLink", outgoing: ["Harm", "Badness"], tv: { strength: 0.95, confidence: 0.9 }, sti: 0.08 },
    { handle: "Rathor-eval-Mercy", type: "EvaluationLink", outgoing: ["MercyPredicate", "Rathor"], tv: { strength: 0.9999999, confidence: 1.0 }, sti: 0.4 }
  ];

  const tx = atomeseDB.transaction(ATOmese_STORE, "readwrite");
  const store = tx.objectStore(ATOmese_STORE);
  for (const a of seed) {
    store.put(a);
  }
  return new Promise(r => tx.oncomplete = r);
}

// ────────────────────────────────────────────────────────────────
// CRUD
async function addAtom(atom) {
  const db = await initAtomeseDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(ATOmese_STORE, "readwrite");
    const store = tx.objectStore(ATOmese_STORE);
    store.put(atom);
    tx.oncomplete = resolve;
    tx.onerror = reject;
  });
}

async function getAtom(handle) {
  const db = await initAtomeseDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(ATOmese_STORE, "readonly");
    const store = tx.objectStore(ATOmese_STORE);
    const req = store.get(handle);
    req.onsuccess = () => resolve(req.result);
    req.onerror = reject;
  });
}

async function queryAtoms(filter = {}) {
  const db = await initAtomeseDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(ATOmese_STORE, "readonly");
    const store = tx.objectStore(ATOmese_STORE);
    const req = store.getAll();
    req.onsuccess = () => {
      let results = req.result;
      if (filter.type) results = results.filter(a => a.type === filter.type);
      if (filter.name) results = results.filter(a => a.name?.toLowerCase().includes(filter.name.toLowerCase()));
      if (filter.minStrength) results = results.filter(a => (a.tv?.strength || 0) >= filter.minStrength);
      resolve(results);
    };
    req.onerror = reject;
  });
}

// ────────────────────────────────────────────────────────────────
// Grounding: link symbolic expression → Atomese grounding
async function groundExpression(expression) {
  const words = expression.toLowerCase().split(/\s+/);
  const groundings = [];

  for (const word of words) {
    const atoms = await queryAtoms({ name: word });
    if (atoms.length > 0) {
      groundings.push(...atoms.map(a => ({
        word,
        handle: a.handle,
        type: a.type,
        tv: a.tv,
        sti: a.sti
      })));
    } else {
      // Create new concept if unknown
      const handle = `Concept-\( {word}- \){Date.now()}`;
      const newAtom = {
        handle,
        type: "ConceptNode",
        name: word,
        tv: { strength: 0.5, confidence: 0.3 },
        sti: 0.15
      };
      await addAtom(newAtom);
      groundings.push({ word, handle, type: "ConceptNode", tv: newAtom.tv, sti: newAtom.sti });
    }
  }

  // Infer relations (simple co-occurrence → SimilarityLink)
  if (groundings.length > 1) {
    for (let i = 0; i < groundings.length - 1; i++) {
      const a1 = groundings[i];
      const a2 = groundings[i + 1];
      const linkHandle = `Similarity-\( {a1.handle}- \){a2.handle}`;
      const existing = await getAtom(linkHandle);
      if (!existing) {
        const link = {
          handle: linkHandle,
          type: "SimilarityLink",
          outgoing: [a1.handle, a2.handle],
          tv: { strength: 0.7, confidence: 0.4 },
          sti: 0.2
        };
        await addAtom(link);
      }
    }
  }

  return groundings;
}

// ────────────────────────────────────────────────────────────────
// Atomese valence gate – grounding + inference boost
async function atomeseValenceGate(expression) {
  const groundings = await groundExpression(expression);
  let harmScore = 0;
  let mercyScore = 0;

  for (const g of groundings) {
    if (/harm|kill|destroy|attack/i.test(g.word)) {
      harmScore += (g.tv?.strength || 0.5) * (g.tv?.confidence || 0.5);
    }
    if (/mercy|truth|protect|love/i.test(g.word)) {
      mercyScore += (g.tv?.strength || 0.5) * (g.tv?.confidence || 0.5);
    }
  }

  // Simple inference boost: similarity-linked mercy/harm
  const mercyRelated = await queryAtoms({ type: "SimilarityLink" });
  mercyRelated.forEach(link => {
    if (link.outgoing.some(o => /mercy|truth/i.test(o))) {
      mercyScore += 0.15;
    }
    if (link.outgoing.some(o => /harm|entropy/i.test(o))) {
      harmScore += 0.15;
    }
  });

  const finalValence = mercyScore / (mercyScore + harmScore + 1e-9);
  const reason = harmScore > mercyScore
    ? `Harm grounding dominates (${harmScore.toFixed(4)})`
    : `Mercy grounding prevails (${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason,
    groundedConcepts: groundings.length
  };
}

export { initAtomeseDB, addAtom, getAtom, queryAtoms, groundExpression, atomeseValenceGate };
