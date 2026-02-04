// hyperon-reasoning-layer.js – sovereign client-side OpenCog Hypergraph + bidirectional PLN chaining
// Forward/backward unification, TV propagation, attention boost, mercy-gated
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
// Database & seed (unchanged from previous – abbreviated)
async function initHyperonDB() {
  // ... (same as before: open DB, seed if empty)
}

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
// Bidirectional PLN chaining – forward + backward unified
async function BidirectionalPLNChain(startHandle, targetHandle, maxDepth = 5, decay = 0.88, minConf = 0.1) {
  const forwardChains = await ForwardPLNChain(startHandle, targetHandle, maxDepth, decay, minConf);
  const backwardChains = await BackwardPLNChain(targetHandle, startHandle, maxDepth, decay, minConf);

  // Merge & reconcile bidirectional evidence
  const merged = new Map(); // key = path string, value = {tv, paths: [forward, backward]}

  forwardChains.forEach(chain => {
    const key = chain.path.join("→");
    if (!merged.has(key)) merged.set(key, { tv: chain.tv, paths: [] });
    merged.get(key).paths.push({ type: "forward", chain });
  });

  backwardChains.forEach(chain => {
    const key = chain.path.reverse().join("→"); // reverse to match forward direction
    if (!merged.has(key)) merged.set(key, { tv: chain.tv, paths: [] });
    merged.get(key).paths.push({ type: "backward", chain });
  });

  // Combine TV from dual paths (evidence accumulation)
  const finalChains = [];
  merged.forEach((value, key) => {
    let mergedTV = { s: 0.5, c: 0.5 };
    value.paths.forEach(p => {
      const tv = p.chain.tv;
      mergedTV.s = Math.max(mergedTV.s, tv.s);
      mergedTV.c = Math.min(1.0, mergedTV.c + tv.c * 0.6); // accumulate confidence
    });

    finalChains.push({
      path: key.split("→"),
      tv: mergedTV,
      length: value.paths[0].chain.length,
      evidence: value.paths.length, // number of directions supporting
      inferenceType: value.paths.length === 2 ? "bidirectional" : value.paths[0].type
    });
  });

  // Sort by combined strength × confidence / length
  return finalChains.sort((a, b) => 
    (b.tv.s * b.tv.c * b.evidence / b.length) - (a.tv.s * a.tv.c * a.evidence / a.length)
  );
}

// Forward chaining (deduction) – unchanged from previous
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

      await dfs(next, depth + 1, [...path, link.handle], newTV);
      visited.delete(next);
    }
  }

  await dfs(startHandle, 0, [], { s: 1.0, c: 1.0 });
  return chains;
}

// Backward chaining (abduction) – symmetric with TV inversion
async function BackwardPLNChain(targetHandle, startHandle, maxDepth, decay, minConf) {
  const links = await queryAtoms({ type: /Link|Hyperedge/ });
  const chains = [];
  const visited = new Set();

  async function dfs(current, depth, path, tv) {
    if (depth > maxDepth) return;
    if (startHandle && current === startHandle && path.length > 1) {
      chains.push({ path: path.reverse(), tv, length: path.length }); // reverse to forward direction
      return;
    }

    const incoming = links.filter(l => l.out && l.out.includes(current));
    for (const link of incoming) {
      const prev = link.out.find(h => h !== current); // assume binary for simplicity
      if (!prev || visited.has(prev)) continue;
      visited.add(prev);

      // Abduction TV inversion: discount strength, confidence decay
      const newTV = {
        s: Math.min(tv.s, link.tv.s * 0.7), // abduction discount
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
// Valence gate using bidirectional chaining
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

  // Bidirectional chaining boost
  const harmChains = await BidirectionalPLNChainInfer("Harm", "Entropy", 5);
  const mercyChains = await BidirectionalPLNChainInfer("Mercy", "Valence", 5);

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
    ? `Harm bidirectional chains & attention dominate (${harmChains.length} chains, score ${harmScore.toFixed(4)})`
    : `Mercy bidirectional chains & attention prevail (${mercyChains.length} chains, score ${mercyScore.toFixed(4)})`;

  return {
    result: finalValence >= 0.9999999 ? 'ACCEPTED' : 'REJECTED',
    valence: finalValence.toFixed(7),
    reason,
    harmChains: harmChains.length,
    mercyChains: mercyChains.length,
    highAttention: highAtt.length
  };
}

export { initHyperonDB, addAtom, queryAtoms, BidirectionalPLNChainInfer, updateAttention, hyperonValenceGate };
