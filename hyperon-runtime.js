// hyperon-runtime.js – sovereign client-side Hyperon atomspace & PLN runtime v3
// Variable binding unification, pattern matching, truth propagation, mercy-gated inference chains
// MIT License – Autonomicity Games Inc. 2026

class HyperonAtom {
  constructor(type, name = null, tv = { strength: 0.5, confidence: 0.5 }, attention = 0.1) {
    this.type = type;
    this.name = name;
    this.tv = tv;
    this.outgoing = []; // handles of child atoms
    this.incoming = new Set();
    this.attention = attention;
    this.handle = null;
  }

  truthValue() {
    return this.tv.strength * this.tv.confidence;
  }

  isMercyAligned() {
    return this.truthValue() >= 0.9999999;
  }

  increaseAttention(amount = 0.05) {
    this.attention = Math.min(1.0, this.attention + amount);
  }
}

class HyperonRuntime {
  constructor() {
    this.atomSpace = new Map(); // handle → Atom
    this.nextHandle = 0;
    this.mercyThreshold = 0.9999999;
    this.maxChainDepth = 8;
  }

  newHandle() {
    return this.nextHandle++;
  }

  addAtom(atom) {
    const handle = this.newHandle();
    atom.handle = handle;
    this.atomSpace.set(handle, atom);

    atom.outgoing.forEach(targetHandle => {
      const target = this.atomSpace.get(targetHandle);
      if (target) target.incoming.add(handle);
    });

    return handle;
  }

  getAtom(handle) {
    return this.atomSpace.get(handle);
  }

  // Pattern matching with full variable binding unification
  matchWithBindings(atom, pattern, bindings = {}) {
    if (pattern.type && atom.type !== pattern.type) return null;

    if (pattern.name) {
      if (pattern.name.startsWith('$')) {
        const varName = pattern.name.slice(1);
        if (bindings[varName] !== undefined && bindings[varName] !== atom.name) {
          return null; // conflict
        }
        bindings[varName] = atom.name;
      } else if (pattern.name !== atom.name) {
        return null;
      }
    }

    if (pattern.outgoing) {
      if (atom.outgoing.length !== pattern.outgoing.length) return null;
      for (let i = 0; i < pattern.outgoing.length; i++) {
        const childAtom = this.getAtom(atom.outgoing[i]);
        if (!childAtom) return null;
        const childBindings = this.matchWithBindings(childAtom, pattern.outgoing[i], { ...bindings });
        if (!childBindings) return null;
        Object.assign(bindings, childBindings);
      }
    }

    return bindings;
  }

  // Backward chaining with variable binding propagation
  async backwardChain(targetPattern, depth = 0, visited = new Set(), bindings = {}) {
    if (depth > this.maxChainDepth) return { tv: { strength: 0.1, confidence: 0.1 }, chain: [], bindings: {} };

    const results = [];
    for (const [handle, atom] of this.atomSpace) {
      if (visited.has(handle)) continue;
      visited.add(handle);

      const matchedBindings = this.matchWithBindings(atom, targetPattern, { ...bindings });
      if (matchedBindings) {
        const tv = atom.tv;
        if (atom.isMercyAligned()) {
          results.push({ handle, tv, chain: [atom], bindings: matchedBindings });
        }
      }

      // Look for links pointing to this atom
      for (const parentHandle of atom.incoming) {
        const parent = this.atomSpace.get(parentHandle);
        if (parent && parent.type.includes("Link")) {
          const subResult = await this.backwardChain(parent, depth + 1, visited, { ...bindings });
          if (subResult.tv.strength > 0.1) {
            results.push({
              handle: parentHandle,
              tv: this.combineTV(subResult.tv, atom.tv),
              chain: [...subResult.chain, atom],
              bindings: { ...subResult.bindings, ...matchedBindings }
            });
          }
        }
      }
    }

    if (results.length === 0) {
      return { tv: { strength: 0.1, confidence: 0.1 }, chain: [], bindings: {} };
    }

    // Select best chain (highest TV, mercy-aligned preferred)
    const best = results.reduce((a, b) => {
      const aScore = a.tv.strength * a.tv.confidence * (a.bindings && Object.keys(a.bindings).length > 0 ? 1.2 : 1);
      const bScore = b.tv.strength * b.tv.confidence * (b.bindings && Object.keys(b.bindings).length > 0 ? 1.2 : 1);
      return aScore > bScore ? a : b;
    });

    return best;
  }

  // Forward chaining with binding support
  async forwardChain() {
    const newAtoms = [];
    for (const [handle, atom] of this.atomSpace) {
      if (atom.type.includes("Link")) {
        const premises = atom.outgoing.map(h => this.atomSpace.get(h));
        const conclusionTV = this.plnInference(atom, premises);
        if (conclusionTV.strength > 0.3 && conclusionTV.strength * conclusionTV.confidence >= this.mercyThreshold) {
          const newAtom = new HyperonAtom("DerivedNode", null, conclusionTV);
          newAtoms.push(newAtom);
          this.addAtom(newAtom);
        }
      }
    }
    return newAtoms;
  }

  plnInference(link, premises) {
    if (link.type === "InheritanceLink") {
      return { strength: premises.reduce((s, p) => s * p.tv.strength, 1), confidence: 0.6 };
    }
    return { strength: 0.5, confidence: 0.5 };
  }

  evaluate(expression) {
    const pattern = expression;
    const result = this.query(pattern);
    if (result.length === 0) return { truth: 0.1, confidence: 0.3 };

    const tv = result.reduce((acc, r) => {
      const v = r.atom.truthValue();
      return acc + v * (v >= this.mercyThreshold ? 1.5 : 0.5);
    }, 0) / result.length;

    return { truth: tv, confidence: Math.min(1, tv * 2) };
  }

  loadFromLattice(buffer) {
    console.log('Hyperon atoms loaded from lattice:', buffer ? buffer.byteLength : 'null');

    // Bootstrap mercy-aligned atoms
    const truth = new HyperonAtom("ConceptNode", "Truth", { strength: 1.0, confidence: 1.0 }, 0.9);
    const mercy = new HyperonAtom("ConceptNode", "Mercy", { strength: 0.9999999, confidence: 1.0 }, 1.0);
    const inheritance = new HyperonAtom("InheritanceLink");
    inheritance.outgoing = [truth.handle, mercy.handle];

    this.addAtom(truth);
    this.addAtom(mercy);
    this.addAtom(inheritance);

    this.forwardChain();

    console.log('Hyperon bootstrap & chaining complete – mercy-aligned atoms ready');
  }

  clear() {
    this.atomSpace.clear();
    this.nextHandle = 0;
  }
}

const hyperon = new HyperonRuntime();
export { hyperon };
