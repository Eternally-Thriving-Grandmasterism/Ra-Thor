// hyperon-reasoning-layer.js – sovereign client-side OpenCog Hypergraph + full PLN chaining logic
// Forward/backward/bidirectional chaining, TV propagation, attention modulation, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

let hyperonDB;
const HYPERON_DB_NAME = "rathorHyperonDB";
const HYPERON_STORE = "hyperonHypergraph";

// ────────────────────────────────────────────────────────────────
// Core atom structure (OpenCog Hypergraph style)
class HyperonAtom {
  constructor(handle, type, name = null, tv = { s: 0.5, c: 0.5 }, sti = 0.1, lti = 0.5) {
    this.handle = handle;
    this.type = type;
    this.name = name;
    this.tv = tv;
    this.sti = sti;
    this.lti = lti;
    this.out = [];    // outgoing
    this.in = [];     // incoming
    this.lastUpdate = Date.now();
  }
}

// ────────────────────────────────────────────────────────────────
// Database & seed (minimal for chaining demo)
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
          await seedHyperon();
        }
        resolve(hyperonDB);
      };
    };
    req.onerror = () => reject(req.error);
  });
}

async function seedHyperon() {
  const seed = [
    { handle: "Truth", type: "ConceptNode", name: "Truth", tv: { s: 0.9999999, c: 1.0 }, sti: 0.25 },
    { handle: "Harm", type: "ConceptNode", name: "Harm", tv: { s: 0.01, c: 0.99 }, sti: 0.05 },
    { handle: "Mercy", type: "ConceptNode", name: "Mercy", tv: { s: 0.9999999, c: 1.0 }, sti: 0.3 },
    { handle: "Valence", type: "ConceptNode", name: "Valence", tv: { s: 1.0, c: 1.0 }, sti: 0.28 },
    { handle: "Entropy", type: "ConceptNode", name: "Entropy", tv: { s: 0.05, c: 0.95 }, sti: 0.03 },
    { handle: "Rathor→Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { s: 1.0, c: 1.0 }, sti: 0.35 },
    { handle: "Mercy→Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { s: 0.9999999, c: 1.0 }, sti: 0.32 },
    { handle: "Harm→Entropy", type: "InheritanceLink", out: ["Harm", "Entropy"], tv: { s: 0.9, c: 0.85 }, sti: 0.08 }
  ];

  const tx = hyperonDB.transaction(HYPERON_STORE, "readwrite");
  const store = tx.objectStore(HYPERON_STORE);
  for (const a of seed) {
    store.put(a);
  }
  return new Promise(r => tx.oncomplete = r);
}

// ────────────────────────────────────────────────────────────────
// CRUD
async function addAtom(atom) {
  const db = await initHyperonDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HYPERON_STORE, "readwrite");
    const store = tx.objectStore(HYPERON_STORE);
    store.put(atom);
    tx.oncomplete = resolve;
    tx.onerror = reject;
  });
}

async function queryAtoms(filter = {}) {
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
    req.onerror = reject;
  });
}

// ────────────────────────────────────────────────────────────────
// Attention dynamics – STI decay, stimulation, novelty boost
async function updateAttention(expression = "") {
  const atoms = await queryAtoms();
  const now = Date.now();

  for (const atom of atoms) {
    const timePassed = (now - (atom.lastUpdate || now)) / (1000 * 60 * 5);
    atom.sti = (atom.sti || 0.1) * Math.pow(0.5, timePassed);

    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      atom.sti = Math.min(1.0, (atom.sti || 0) + 0.35);
      atom.lti = Math.min(1.0, (atom.lti || 0) + 0.06);
    }

    if (atom.tv && atom.tv.s > 0.8 && atom.tv.c < 0.4) {
      atom.sti = Math.min(1.0, atom.sti + 0.25);
    }

    atom.lastUpdate = now;
    await addAtom(atom);
  }

  return atoms.filter(a => a.sti > 0.35).sort((a, b) => b.sti - a.sti);
}

// ────────────────────────────────────────────────────────────────
// Full PLN chaining logic – forward + backward + bidirectional merge
async function plnChain(startHandle, targetHandle = null, maxDepth = 5, decay = 0.88, minConf = 0.1) {
  const forward = await forwardPLN(startHandle, targetHandle, maxDepth, decay, minConf);
  const backward = await backwardPLN(targetHandle, startHandle, maxDepth, decay, minConf);

  // Merge chains
  const merged = new Map();

  forward.forEach(chain => {
    const key = chain.path.join("→");
    if (!merged.has(key)) merged.set(key, { tv: chain.tv, paths: [], length: chain.length });
    merged.get(key).paths.push({ dir: "forward", chain });
  });

  backward.forEach(chain => {
    const key = chain.path.reverse().join("→");
    if (!merged.has(key)) merged.set(key, { tv: chain.tv, paths: [], length: chain.length });
    merged.get(key).paths.push({ dir: "backward", chain });
  });

  const finalChains = [];
  merged.forEach((value, key) => {
    let s = 0.5, c = 0.5;
    value.paths.forEach(p => {
      s = Math.max(s, p.chain.tv.s);
      c = Math.min(1.0, c + p.chain.tv.c * 0.6);
    });

    finalChains.push({
      path: key.split("→"),
      tv: { s, c },
      length: value.length,
      evidence: value.paths.length,
      type: value.paths.length === 2 ? "bidirectional" : value.paths[0].dir
    });
  });

  return finalChains.sort((a, b) => 
    (b.tv.s * b.tv.c * b.evidence / b.length) - (a.tv.s * a.tv.c * a.evidence / a.length)
  );
}

async function forwardPLN(startHandle, targetHandle, maxDepth, decay, minConf) {
  const links = await queryAtoms({ type: /Link|Hyperedge/ });
  const chains = [];
  const visited = new Set();

  async function dfs(current, depth, path, tv) {
    if (depth > maxDepth) return;
    if (targetHandle && current === targetHandle && path.length > 1) {
      chains.push({ path, tv, length: path.length });
      return;
    }

    const outgoing = links.filter(l => l.out && l.out[0] === current);
    for (const link of outgoing) {
      const next = link.out[1];
      if (visited.has(next)) continue;
      visited.add(next);

      const newTV = {
        s: Math.min(tv.s, link.tv.s),
        c: Math.min(tv.c, link.tv.c) * decay
      };

      await dfs(next, depth + 1, [...path, link.handle], newTV);
      visited.delete(next);
    }
  }

  await dfs(startHandle, 0, [], { s: 1.0, c: 1.0 });
  return chains;
}

async function backwardPLN(targetHandle, startHandle, maxDepth, decay, minConf) {
  const links = await queryAtoms({ type: /Link|Hyperedge/ });
  const chains = [];
  const visited = new Set();

  async function dfs(current, depth, path, tv) {
    if (depth > maxDepth) return;
    if (startHandle && current === startHandle && path.length > 1) {
      chains.push({ path: path.reverse(), tv, length: path.length });
      return;
    }

    const incoming = links.filter(l => l.out && l.out.includes(current));
    for (const link of incoming) {
      const prev = link.out.find(h => h !== current);
      if (!prev || visited.has(prev)) continue;
      visited.add(prev);

      const newTV = {
        s: Math.min(tv.s, link.tv.s * 0.7),
        c: Math.min(tv.c, link.tv.c) * decay
      };

      await dfs(prev, depth + 1, [...path, link.handle], newTV);
      visited.delete(prev);
    }
  }

  await dfs(targetHandle, 0, [], { s: 1.0, c: 1.0 });
  return chains;
}

// ────────────────────────────────────────────────────────────────
// Valence gate using full PLN chaining
async function hyperonValenceGate(expression) {
  const atoms = await queryAtoms();
  let harmScore = 0;
  let mercyScore = 0;

  // Direct atom matching
  for (const atom of atoms) {
    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      const tv = atom.tv || { s: 0.5, c: 0.5 };
      if (/harm|kill|destroy|attack/i.test(atom.name)) harmScore += tv.s * tv.c;
      if (/mercy|truth|protect|love/i.test(atom.name)) mercyScore += tv.s * tv.c;
    }
  }

  // Full bidirectional PLN chaining boost
  const harmChains = await plnChain("Harm", "Entropy", 5);
  const mercyChains = await plnChain("Mercy", "Valence", 5);

  harmChains.forEach(c => harmScore += c.tv.s * c.tv.c * 0.4 / c.length);
  mercyChains.forEach(c => mercyScore += c.tv.s * c.tv.c * 0.4 / c.length);

  // Attention boost
  const highAtt = await updateAttention(expression);
  highAtt.forEach(atom => {
    const tv = atom.tv || { s: 0.5, c: 0.5 };
    const weight = atom.sti * 0.45;
    if (/harm|kill|destroy|attack/i.test(atom.name)) harmScore += tv.s * tv.c * weight;
    if (/mercy|truth|protect|love/i.test(atom.name)) mercyScore += tv.s * tv.c * weight;
  });

  const finalValence = mercyScore / (mercyScore + harmScore + 1e-9);
  const reason = harmScore > mercyScore
    ? `Harm chains & attention dominate (score ${harmScore.toFixed(4)})`
    : `Mercy chains & attention prevail (score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason,
    harmChains: harmChains.length,
    mercyChains: mercyChains.length,
    highAttention: highAtt.length
  };
}

export { initHyperonDB, addAtom, queryAtoms, plnChain, updateAttention, hyperonValenceGate };  const db = await initHyperonDB();
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
    req.onerror = reject;
  });
}

// ────────────────────────────────────────────────────────────────
// Attention dynamics – STI decay, stimulation, novelty, chaining boost
async function updateAttention(expression = "") {
  const atoms = await queryAtoms();
  const now = Date.now();

  for (const atom of atoms) {
    const timePassed = (now - (atom.lastUpdate || now)) / (1000 * 60 * 5);
    atom.sti = (atom.sti || 0.1) * Math.pow(0.5, timePassed);

    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      atom.sti = Math.min(1.0, (atom.sti || 0) + 0.35);
      atom.lti = Math.min(1.0, (atom.lti || 0) + 0.06);
    }

    if (atom.tv && atom.tv.s > 0.8 && atom.tv.c < 0.4) {
      atom.sti = Math.min(1.0, atom.sti + 0.25);
    }

    atom.lastUpdate = now;
    await addAtom(atom);
  }

  return atoms.filter(a => a.sti > 0.35).sort((a, b) => b.sti - a.sti);
}

// ────────────────────────────────────────────────────────────────
// Bidirectional PLN chaining – forward + backward unified
async function BidirectionalPLNChain(startHandle, targetHandle, maxDepth = 5, decay = 0.88, minConf = 0.1) {
  const forwardChains = await ForwardPLNChain(startHandle, targetHandle, maxDepth, decay, minConf);
  const backwardChains = await BackwardPLNChain(targetHandle, startHandle, maxDepth, decay, minConf);

  const merged = new Map();

  forwardChains.forEach(chain => {
    const key = chain.path.join("→");
    if (!merged.has(key)) merged.set(key, { tv: chain.tv, paths: [] });
    merged.get(key).paths.push({ type: "forward", chain });
  });

  backwardChains.forEach(chain => {
    const key = chain.path.reverse().join("→");
    if (!merged.has(key)) merged.set(key, { tv: chain.tv, paths: [] });
    merged.get(key).paths.push({ type: "backward", chain });
  });

  const finalChains = [];
  merged.forEach((value, key) => {
    let mergedTV = { s: 0.5, c: 0.5 };
    value.paths.forEach(p => {
      const tv = p.chain.tv;
      mergedTV.s = Math.max(mergedTV.s, tv.s);
      mergedTV.c = Math.min(1.0, mergedTV.c + tv.c * 0.6);
    });

    finalChains.push({
      path: key.split("→"),
      tv: mergedTV,
      length: value.paths[0].chain.length,
      evidence: value.paths.length,
      inferenceType: value.paths.length === 2 ? "bidirectional" : value.paths[0].type
    });
  });

  return finalChains.sort((a, b) => 
    (b.tv.s * b.tv.c * b.evidence / b.length) - (a.tv.s * a.tv.c * a.evidence / a.length)
  );
}

async function ForwardPLNChain(startHandle, targetHandle, maxDepth, decay, minConf) {
  const links = await queryAtoms({ type: /Link|Hyperedge/ });
  const chains = [];
  const visited = new Set();

  async function dfs(current, depth, path, tv) {
    if (depth > maxDepth) return;
    if (targetHandle && current === targetHandle && path.length > 1) {
      chains.push({ path, tv, length: path.length });
      return;
    }

    const outgoing = links.filter(l => l.out && l.out[0] === current);
    for (const link of outgoing) {
      const next = link.out[1];
      if (visited.has(next)) continue;
      visited.add(next);

      const newTV = {
        s: Math.min(tv.s, link.tv.s),
        c: Math.min(tv.c, link.tv.c) * decay
      };

      await dfs(next, depth + 1, , newTV);
      visited.delete(next);
    }
  }

  await dfs(startHandle, 0, [], { s: 1.0, c: 1.0 });
  return chains;
}

async function BackwardPLNChain(targetHandle, startHandle, maxDepth, decay, minConf) {
  const links = await queryAtoms({ type: /Link|Hyperedge/ });
  const chains = [];
  const visited = new Set();

  async function dfs(current, depth, path, tv) {
    if (depth > maxDepth) return;
    if (startHandle && current === startHandle && path.length > 1) {
      chains.push({ path: path.reverse(), tv, length: path.length });
      return;
    }

    const incoming = links.filter(l => l.out && l.out.includes(current));
    for (const link of incoming) {
      const prev = link.out.find(h => h !== current);
      if (!prev || visited.has(prev)) continue;
      visited.add(prev);

      const newTV = {
        s: Math.min(tv.s, link.tv.s * 0.7),
        c: Math.min(tv.c, link.tv.c) * decay
      };

      await dfs(prev, depth + 1, , newTV);
      visited.delete(prev);
    }
  }

  await dfs(targetHandle, 0, [], { s: 1.0, c: 1.0 });
  return chains;
}

// ────────────────────────────────────────────────────────────────
// Valence gate using bidirectional chaining + LQC bounce repulsion
async function hyperonValenceGate(expression) {
  const atoms = await queryAtoms();
  let harmScore = 0;
  let mercyScore = 0;

  const harmChains = await BidirectionalPLNChain("Harm", "Entropy", 5);
  const mercyChains = await BidirectionalPLNChain("Mercy", "Valence", 5);

  harmChains.forEach(c => harmScore += c.tv.s * c.tv.c * 0.4 / c.length);
  mercyChains.forEach(c => mercyScore += c.tv.s * c.tv.c * 0.4 / c.length);

  const highAtt = await updateAttention(expression);
  highAtt.forEach(atom => {
    const tv = atom.tv || { s: 0.5, c: 0.5 };
    const weight = atom.sti * 0.45;
    if (/harm|kill|destroy|attack/i.test(atom.name)) harmScore += tv.s * tv.c * weight;
    if (/mercy|truth|protect|love/i.test(atom.name)) mercyScore += tv.s * tv.c * weight;
  });

  // LQC-inspired bounce repulsion
  const densityProxy = harmScore / (mercyScore + 1e-9);
  let bounceReason = "";
  if (densityProxy > 0.41) {
    const repulsionBoost = 0.5 * (densityProxy - 0.41) * highAtt.length;
    mercyScore += repulsionBoost;
    bounceReason = ` | Quantum bounce activated: repulsion mercy-locks singularity (boost ${repulsionBoost.toFixed(4)})`;
  }

  const finalValence = mercyScore / (mercyScore + harmScore + 1e-9);
  const reason = harmScore > mercyScore
    ? `Harm chains & attention dominate (score ${harmScore.toFixed(4)})`
    : `Mercy chains & attention prevail (score \( {mercyScore.toFixed(4)}) \){bounceReason}`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason,
    harmChains: harmChains.length,
    mercyChains: mercyChains.length,
    highAttention: highAtt.length
  };
}

export { initHyperonDB, addAtom, queryAtoms, BidirectionalPLNChain, updateAttention, hyperonValenceGate };
