// hyperon-runtime-optimizer.js – sovereign PLN/Atomspace optimizer v1
// Memoization, depth limit, derivation cache, mercy-weighted pruning
// MIT License – Autonomicity Games Inc. 2026

import { hyperon } from './hyperon-runtime.js';
import { fuzzyMercy } from './fuzzy-mercy-logic.js';

class HyperonOptimizer {
  constructor() {
    this.derivationCache = new Map(); // queryHash → {atoms, tv}
    this.maxDepth = 6; // Reduced from 8 to prevent explosion
    this.mercyPruneThreshold = 0.9999999 * 0.95;
  }

  hash(key) {
    let h = 0;
    for (let i = 0; i < key.length; i++) {
      h = ((h << 5) - h + key.charCodeAt(i)) | 0;
    }
    return h.toString(36);
  }

  // Memoized forward chain
  async optimizedForwardChain(query, depth = this.maxDepth) {
    const key = this.hash(query + depth);
    if (this.derivationCache.has(key)) {
      return this.derivationCache.get(key);
    }

    const derived = await hyperon.forwardChain(depth);
    const filtered = derived.filter(d => {
      const tv = d.atom.tv;
      return tv.strength * tv.confidence >= this.mercyPruneThreshold;
    });

    const result = { derived: filtered, count: filtered.length };
    this.derivationCache.set(key, result);
    return result;
  }

  // Prune low-mercy atoms from atomspace
  pruneLowValence() {
    for (const [handle, atom] of hyperon.atomSpace) {
      const tv = atom.tv;
      if (tv.strength * tv.confidence < this.mercyPruneThreshold) {
        hyperon.atomSpace.delete(handle);
      }
    }
  }

  // Clear cache periodically
  clearCache() {
    this.derivationCache.clear();
  }
}

const optimizer = new HyperonOptimizer();
export { optimizer };
