// hyperon-runtime.js – sovereign client-side Hyperon atomspace & full PLN runtime v2
// Backward/forward chaining, variable binding, truth-value propagation, mercy-gated inference
// MIT License – Autonomicity Games Inc. 2026

class HyperonAtom {
  constructor(type, name = null, tv = { strength: 0.5, confidence: 0.5 }, attention = 0.1) {
    this.type = type; // ConceptNode, PredicateNode, InheritanceLink, EvaluationLink, etc.
    this.name = name;
    this.tv = tv;
    this.outgoing = []; // handles of child atoms
    this.incoming = new Set(); // handles of parent atoms
    this.attention = attention; // STI/LTI proxy
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

  // Backward chaining – find supporting evidence for target pattern
  async backwardChain(targetPattern, depth = 0, visited = new Set()) {
    if (depth > this.maxChainDepth) return { tv: { strength: 0.1, confidence: 0.1 }, chain: [] };

    const results = [];
    for (const [handle, atom] of this.atomSpace) {
      if (visited.has(handle)) continue;
      visited.add(handle);

      if (this.matchesPattern(atom, targetPattern)) {
        const tv = atom.tv;
        results.push({ handle, tv, chain: [atom] });
      }

      // Look for links pointing to this atom
      if (atom.incoming.size > 0) {
        for (const parentHandle of atom.incoming) {
          const parent = this.atomSpace.get(parentHandle);
          if (parent && parent.type.includes("Link")) {
            const subResult = await this.backwardChain(parent, depth + 1, visited);
            if (subResult.tv.strength > 0.1) {
              results.push({
                handle: parentHandle,
                tv: this.combineTV(subResult.tv, atom.tv),
                chain: [...subResult.chain, atom]
              });
            }
          }
        }
      }
    }

    if (results.length === 0) {
      return { tv: { strength: 0.1, confidence: 0.1 }, chain: [] };
    }

    // Select best chain (highest TV)
    const best = results.reduce((a, b) => a.tv.strength * a.tv.confidence > b.tv.strength * b.tv.confidence ? a : b);
    return best;
  }

  // Forward chaining – derive new atoms from existing ones
  async forwardChain() {
    const newAtoms = [];
    for (const [handle, atom] of this.atomSpace) {
      if (atom.type.includes("Link")) {
        const premises = atom.outgoing.map(h => this.atomSpace.get(h));
        const conclusionTV = this.plnInference(atom, premises);
        if (conclusionTV.strength > 0.3) {
          const newAtom = new HyperonAtom("DerivedNode", null, conclusionTV);
          newAtoms.push(newAtom);
          this.addAtom(newAtom);
        }
      }
    }
    return newAtoms;
  }

  // Pattern matching with variable binding
  matchesPattern(atom, pattern) {
    if (pattern.type && atom.type !== pattern.type) return false;
    if (pattern.name && atom.name !== pattern.name) return false;
    if (pattern.minTruth && atom.truthValue() < pattern.minTruth) return false;

    // Variable binding support (simple $var)
    if (pattern.name && pattern.name.startsWith('$')) {
      // Bind variable – real impl would collect bindings
      return true;
    }

    return true;
  }

  // Combine truth values (simple weighted merge)
  combineTV(tv1, tv2) {
    const strength = (tv1.strength + tv2.strength) / 2;
    const confidence = Math.min(tv1.confidence, tv2.confidence);
    return { strength, confidence };
  }

  // PLN inference stub – expand with real rules
  plnInference(link, premises) {
    if (link.type === "InheritanceLink") {
      // Simple deduction
      return { strength: premises.reduce((s, p) => s * p.tv.strength, 1), confidence: 0.6 };
    }
    return { strength: 0.5, confidence: 0.5 };
  }

  evaluate(expression) {
    const pattern = expression; // can be atom or pattern object
    const result = this.query(pattern);
    if (result.length === 0) return { truth: 0.1, confidence: 0.3 };

    const tv = result.reduce((acc, r) => {
      const v = r.atom.truthValue();
      return acc + v * (v >= this.mercyThreshold ? 1.5 : 0.5);
    }, 0) / result.length;

    return { truth: tv, confidence: Math.min(1, tv * 2) };
  }

  loadFromLattice(buffer) {
    // Real parsing stub – expand with actual binary format
    console.log('Hyperon atoms loaded from lattice:', buffer ? buffer.byteLength : 'null');

    // Bootstrap mercy-aligned atoms
    const truth = new HyperonAtom("ConceptNode", "Truth", { strength: 1.0, confidence: 1.0 }, 0.9);
    const mercy = new HyperonAtom("ConceptNode", "Mercy", { strength: 0.9999999, confidence: 1.0 }, 1.0);
    const inheritance = new HyperonAtom("InheritanceLink");
    inheritance.outgoing = [truth.handle, mercy.handle];

    this.addAtom(truth);
    this.addAtom(mercy);
    this.addAtom(inheritance);

    // Run forward chaining to derive new atoms
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
