// hyperon-reasoning-layer.js – sovereign client-side OpenCog Hypergraph reasoning engine
// Full PLN chaining (bi-directional), attention dynamics, pattern mining, clustering, LQC bounce
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
// Database & seed
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
    { handle: "Badness", type: "ConceptNode", name: "Badness", tv: { s: 0.9, c: 0.9 }, sti: 0.08 },
    { handle: "Rathor→Mercy", type: "InheritanceLink", out: ["Rathor", "Mercy"], tv: { s: 1.0, c: 1.0 }, sti: 0.35 },
    { handle: "Mercy→Valence", type: "InheritanceLink", out: ["Mercy", "Valence"], tv: { s: 0.9999999, c: 1.0 }, sti: 0.32 },
    { handle: "Harm→Badness", type: "InheritanceLink", out: ["Harm", "Badness"], tv: { s: 0.95, c: 0.9 }, sti: 0.08 },
    { handle: "Badness→Entropy", type: "InheritanceLink", out: ["Badness", "Entropy"], tv: { s: 0.92, c: 0.88 }, sti: 0.06 }
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
// Pattern mining – discovers frequent link patterns & subgraphs
async function minePatterns(minSupport = 0.3, maxPatternSize = 5) {
  const allLinks = await queryAtoms({ type: /Link|Hyperedge/ });
  const patterns = [];
  const linkFreq = new Map();

  // Count individual link frequencies
  allLinks.forEach(link => {
    if (!link.out) return;
    const key = link.out.join(" → ");
    linkFreq.set(key, (linkFreq.get(key) || 0) + 1);
  });

  const totalLinks = allLinks.length;
  linkFreq.forEach((count, key) => {
    const support = count / totalLinks;
    if (support >= minSupport) {
      patterns.push({
        pattern: key,
        support: support.toFixed(4),
        count,
        type: "frequent-link"
      });
    }
  });

  // Chain pattern detection (A → B → C)
  for (let i = 0; i < allLinks.length; i++) {
    for (let j = 0; j < allLinks.length; j++) {
      if (i === j) continue;
      const link1 = allLinks[i];
      const link2 = allLinks[j];
      if (link1.out && link1.out[1] === link2.out[0]) {
        const chain = `${link1.out[0]} → ${link1.out[1]} → ${link2.out[1]}`;
        const support = Math.min(link1.tv.s, link2.tv.s);
        if (support >= minSupport) {
          patterns.push({
            pattern: chain,
            support: support.toFixed(4),
            type: "inference-chain"
          });
        }
      }
    }
  }

  return patterns.sort((a, b) => b.support - a.support);
}

// ────────────────────────────────────────────────────────────────
// Similarity clustering – groups concepts/links by TV & structural overlap
async function clusterSimilarAtoms(threshold = 0.7) {
  const concepts = await queryAtoms({ type: "ConceptNode" });
  const links = await queryAtoms({ type: /Link|Hyperedge/ });
  const clusters = [];
  const visited = new Set();

  function similarity(a, b) {
    const tvA = a.tv || { s: 0.5, c: 0.5 };
    const tvB = b.tv || { s: 0.5, c: 0.5 };
    const dot = tvA.s * tvB.s + tvA.c * tvB.c;
    const normA = Math.sqrt(tvA.s**2 + tvA.c**2);
    const normB = Math.sqrt(tvB.s**2 + tvB.c**2);
    let cosSim = dot / (normA * normB || 1);

    let overlap = 0;
    if (a.type === "ConceptNode" && b.type === "ConceptNode") {
      const aOut = links.filter(l => l.out.includes(a.handle));
      const bOut = links.filter(l => l.out.includes(b.handle));
      const shared = new Set(aOut.map(l => l.handle)).size && new Set(bOut.map(l => l.handle)).size;
      overlap += shared * 0.25;
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
// Valence gate using full PLN chaining + pattern mining + clustering + LQC bounce
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
  const harmChains = await BidirectionalPLNChainInfer("Harm", "Entropy", 5);
  const mercyChains = await BidirectionalPLNChainInfer("Mercy", "Valence", 5);

  harmChains.forEach(c => harmScore += c.tv.s * c.tv.c * 0.4 / c.length);
  mercyChains.forEach(c => mercyScore += c.tv.s * c.tv.c * 0.4 / c.length);

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

  // Similarity clustering boost
  const clusters = await clusterSimilarAtoms(0.7);
  clusters.forEach(cluster => {
    const hasHarm = cluster.members.some(m => /harm|kill|entropy/i.test(m));
    const hasMercy = cluster.members.some(m => /mercy|truth|valence|love/i.test(m));
    const clusterWeight = cluster.size * cluster.avgSimilarity;
    if (hasHarm) harmScore += clusterWeight * 0.15;
    if (hasMercy) mercyScore += clusterWeight * 0.15;
  });

  // Attention dynamics boost
  const highAttention = await updateAttention(expression);
  highAttention.forEach(atom => {
    const tv = atom.tv || { s: 0.5, c: 0.5 };
    const weight = atom.sti * 0.4;
    if (/harm|kill|destroy|attack/i.test(atom.name)) harmScore += tv.s * tv.c * weight;
    if (/mercy|truth|protect|love/i.test(atom.name)) mercyScore += tv.s * tv.c * weight;
  });

  // LQC-inspired bounce repulsion
  const densityProxy = harmScore / (mercyScore + 1e-9);
  let bounceReason = "";
  if (densityProxy > 0.41) {
    const repulsionBoost = 0.5 * (densityProxy - 0.41) * highAttention.length;
    mercyScore += repulsionBoost;
    bounceReason = ` | Quantum bounce activated: repulsion mercy-locks singularity (boost ${repulsionBoost.toFixed(4)})`;
  }

  const finalValence = mercyScore / (mercyScore + harmScore + 1e-9);
  const reason = harmScore > mercyScore
    ? `Harm chains, patterns, clusters & attention dominate (score ${harmScore.toFixed(4)})`
    : `Mercy chains, patterns, clusters & attention prevail (score \( {mercyScore.toFixed(4)}) \){bounceReason}`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason,
    harmChains: harmChains.length,
    mercyChains: mercyChains.length,
    patternsFound: patterns.length,
    clustersFound: clusters.length,
    highAttention: highAttention.length
  };
}

export { initHyperonDB, addAtom, queryAtoms, updateAttention, minePatterns, clusterSimilarAtoms, hyperonValenceGate };
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

async function getAtom(handle) {
  const db = await initHyperonDB();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(HYPERON_STORE, "readonly");
    const store = tx.objectStore(HYPERON_STORE);
    const req = store.get(handle);
    req.onsuccess = () => resolve(req.result);
    req.onerror = reject;
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
// Attention dynamics
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
// PLN inference chaining over hypergraph
async function plnChainInfer(start, target = null, maxDepth = 5, decay = 0.88) {
  const links = await queryAtoms({ type: /Link|Hyperedge/ });
  const chains = [];
  const visited = new Set();

  async function dfs(currentHandle, depth, path, currentTV) {
    if (depth > maxDepth) return;
    if (target && currentHandle === target && path.length > 1) {
      chains.push({ path, tv: currentTV, length: path.length });
      return;
    }

    const outgoing = links.filter(l => l.out && l.out[0] === currentHandle);
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

  await dfs(start, 0, [], { s: 1.0, c: 1.0 });
  return chains.sort((a, b) => (b.tv.s * b.tv.c / b.length) - (a.tv.s * a.tv.c / a.length));
}

// ────────────────────────────────────────────────────────────────
// Hyperon valence gate – full hypergraph reasoning stack
async function hyperonValenceGate(expression) {
  const atoms = await queryAtoms();
  let harmScore = 0;
  let mercyScore = 0;

  for (const atom of atoms) {
    if (atom.name && expression.toLowerCase().includes(atom.name.toLowerCase())) {
      const tv = atom.tv || { s: 0.5, c: 0.5 };
      if (/harm|kill|destroy|attack/i.test(atom.name)) harmScore += tv.s * tv.c;
      if (/mercy|truth|protect|love/i.test(atom.name)) mercyScore += tv.s * tv.c;
    }
  }

  // PLN chaining boost
  const plnResults = await plnChainInfer("Harm", "Entropy");
  plnResults.forEach(inf => {
    if (inf.out.some(o => /harm/i.test(o))) harmScore += inf.tv.s * inf.tv.c * 0.3;
    if (inf.out.some(o => /mercy|truth/i.test(o))) mercyScore += inf.tv.s * inf.tv.c * 0.3;
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

  // Similarity clustering boost
  const clusters = await clusterSimilarAtoms(0.7);
  clusters.forEach(cluster => {
    const hasHarm = cluster.members.some(m => /harm|kill|entropy/i.test(m));
    const hasMercy = cluster.members.some(m => /mercy|truth|valence|love/i.test(m));
    const clusterWeight = cluster.size * cluster.avgSimilarity;
    if (hasHarm) harmScore += clusterWeight * 0.15;
    if (hasMercy) mercyScore += clusterWeight * 0.15;
  });

  // Attention dynamics boost
  const highAttention = await updateAttention(expression);
  highAttention.forEach(atom => {
    const tv = atom.tv || { s: 0.5, c: 0.5 };
    const weight = atom.sti * 0.4;
    if (/harm|kill|destroy|attack/i.test(atom.name)) harmScore += tv.s * tv.c * weight;
    if (/mercy|truth|protect|love/i.test(atom.name)) mercyScore += tv.s * tv.c * weight;
  });

  const finalValence = mercyScore / (mercyScore + harmScore + 0.000001);
  const reason = harmScore > mercyScore 
    ? `Harm hyperedges, clusters & attention dominate (score ${harmScore.toFixed(4)})` 
    : `Mercy hyperedges, clusters & attention prevail (score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason,
    minedPatternsCount: patterns.length,
    clusterCount: clusters.length,
    highAttentionCount: highAttention.length
  };
}

export { initHyperonDB, addAtom, queryAtoms, plnChainInfer, updateAttention, hyperonValenceGate };
