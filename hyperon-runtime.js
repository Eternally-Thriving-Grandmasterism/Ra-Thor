// hyperon-runtime.js – sovereign client-side Hyperon atomspace & PLN runtime
// Offline-first, mercy-gated, valence-locked inference engine
// MIT License – Autonomicity Games Inc. 2026

class HyperonAtom {
  constructor(type, name = null, tv = { strength: 0.5, confidence: 0.5 }) {
    this.type = type; // ConceptNode, InheritanceLink, EvaluationLink, etc.
    this.name = name;
    this.tv = tv;
    this.outgoing = []; // for links
    this.incoming = new Set(); // reverse refs
    this.attention = 0.1; // STI/LTI proxy
  }

  truthValue() {
    return this.tv.strength * this.tv.confidence;
  }

  isMercyAligned() {
    return this.truthValue() >= 0.9999999;
  }
}

class HyperonRuntime {
  constructor() {
    this.atomSpace = new Map(); // handle → Atom
    this.nextHandle = 0;
    this.mercyThreshold = 0.9999999;
  }

  newHandle() {
    return this.nextHandle++;
  }

  addAtom(atom) {
    const handle = this.newHandle();
    this.atomSpace.set(handle, atom);
    atom.handle = handle;

    // Index incoming links
    atom.outgoing.forEach(targetHandle => {
      const target = this.atomSpace.get(targetHandle);
      if (target) target.incoming.add(handle);
    });

    return handle;
  }

  getAtom(handle) {
    return this.atomSpace.get(handle);
  }

  query(pattern) {
    // Simple pattern matching stub – real impl uses PLN backward chaining
    const results = [];
    for (const [handle, atom] of this.atomSpace) {
      if (pattern.type && atom.type !== pattern.type) continue;
      if (pattern.name && atom.name !== pattern.name) continue;
      if (pattern.minTruth && atom.truthValue() < pattern.minTruth) continue;
      results.push({ handle, atom });
    }
    return results;
  }

  evaluate(expression) {
    // Mercy-gated PLN evaluation stub
    const result = this.query(expression);
    if (result.length === 0) return { truth: 0.1, confidence: 0.3 };

    const tv = result.reduce((acc, r) => {
      const v = r.atom.truthValue();
      return acc + v * (v >= this.mercyThreshold ? 1.5 : 0.5);
    }, 0) / result.length;

    return { truth: tv, confidence: Math.min(1, tv * 2) };
  }

  loadFromLattice(buffer) {
    // Stub – real impl parses binary into atoms
    console.log('Hyperon atoms loaded from lattice:', buffer.byteLength, 'bytes');

    // Example bootstrap atoms
    const truth = new HyperonAtom("ConceptNode", "Truth", { strength: 1.0, confidence: 1.0 });
    const mercy = new HyperonAtom("ConceptNode", "Mercy", { strength: 0.9999999, confidence: 1.0 });
    const inheritance = new HyperonAtom("InheritanceLink");
    inheritance.outgoing = [truth.handle, mercy.handle];

    this.addAtom(truth);
    this.addAtom(mercy);
    this.addAtom(inheritance);

    console.log('Hyperon bootstrap complete – mercy-aligned atoms ready');
  }

  clear() {
    this.atomSpace.clear();
    this.nextHandle = 0;
  }
}

const hyperon = new HyperonRuntime();
export { hyperon };
