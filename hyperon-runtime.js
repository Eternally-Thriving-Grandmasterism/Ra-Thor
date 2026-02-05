// hyperon-runtime.js – sovereign client-side Hyperon hypergraph atomspace & PLN engine v14
// Persistent IndexedDB-backed atomspace, unification, chaining, mercy-gated inference
// MIT License – Autonomicity Games Inc. 2026

class HyperonAtom {
  constructor(type, name = null, tv = { strength: 0.5, confidence: 0.5 }, sti = 0.1, lti = 0.01, handle = null) {
    this.type = type;
    this.name = name;
    this.tv = tv;
    this.sti = sti; // short-term importance
    this.lti = lti; // long-term importance
    this.outgoing = []; // child handles
    this.incoming = new Set(); // parent handles
    this.handle = handle;
  }

  truthValue() {
    return this.tv.strength * this.tv.confidence;
  }

  isMercyAligned() {
    return this.truthValue() >= 0.9999999;
  }

  boostAttention(amount = 0.1) {
    this.sti = Math.min(1.0, this.sti + amount);
    this.lti = Math.min(1.0, this.lti + amount * 0.1);
  }
}

class HyperonRuntime {
  constructor() {
    this.atomSpace = new Map(); // handle → Atom
    this.nextHandle = 0;
    this.mercyThreshold = 0.9999999;
    this.maxChainDepth = 12;
    this.attentionDecay = 0.95;
    this.db = null;
    this.dbName = "rathorHyperonDB";
    this.storeName = "atoms";

    this.inferenceRules = [
      // Deduction-Inheritance
      {
        name: "Deduction-Inheritance",
        premises: ["InheritanceLink $A $B", "InheritanceLink $B $C"],
        conclusion: "InheritanceLink $A $C",
        tvCombiner: (tvs) => ({
          strength: tvs.reduce((s, tv) => s * tv.strength, 1),
          confidence: Math.min(...tvs.map(tv => tv.confidence)) * 0.8
        }),
        priority: 10
      },
      // Modus Ponens
      {
        name: "Modus Ponens",
        premises: ["ImplicationLink $A $B", "EvaluationLink $A"],
        conclusion: "EvaluationLink $B",
        tvCombiner: (tvs) => ({
          strength: tvs[0].strength * tvs[1].strength,
          confidence: Math.min(tvs[0].confidence, tvs[1].confidence) * 0.9
        }),
        priority: 20
      },
      // ... (other rules as before – Modus Tollens, Hypothetical Syllogism, etc.)
    ].sort((a, b) => b.priority - a.priority);
  }

  async init() {
    this.db = await this.openDB();
    await this.loadFromDB();
  }

  async openDB() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(this.dbName, 1);
      req.onupgradeneeded = e => {
        const db = e.target.result;
        db.createObjectStore(this.storeName, { keyPath: "handle" });
      };
      req.onsuccess = e => resolve(e.target.result);
      req.onerror = reject;
    });
  }

  async loadFromDB() {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.storeName, "readonly");
      const store = tx.objectStore(this.storeName);
      const req = store.getAll();
      req.onsuccess = () => {
        req.result.forEach(atomData => {
          const atom = new HyperonAtom(
            atomData.type,
            atomData.name,
            atomData.tv,
            atomData.sti,
            atomData.lti,
            atomData.handle
          );
          atom.outgoing = atomData.outgoing || [];
          atom.incoming = new Set(atomData.incoming || []);
          this.atomSpace.set(atom.handle, atom);
          this.nextHandle = Math.max(this.nextHandle, atom.handle + 1);
        });
        console.log("[Hyperon] Loaded", this.atomSpace.size, "atoms from IndexedDB");
        resolve();
      };
      req.onerror = reject;
    });
  }

  async saveAtom(atom) {
    return new Promise((resolve, reject) => {
      const tx = this.db.transaction(this.storeName, "readwrite");
      const store = tx.objectStore(this.storeName);
      const data = {
        handle: atom.handle,
        type: atom.type,
        name: atom.name,
        tv: atom.tv,
        sti: atom.sti,
        lti: atom.lti,
        outgoing: atom.outgoing,
        incoming: Array.from(atom.incoming)
      };
      store.put(data);
      tx.oncomplete = resolve;
      tx.onerror = reject;
    });
  }

  newHandle() {
    return this.nextHandle++;
  }

  addAtom(atom) {
    if (!atom.handle) atom.handle = this.newHandle();
    this.atomSpace.set(atom.handle, atom);

    atom.outgoing.forEach(targetHandle => {
      const target = this.atomSpace.get(targetHandle);
      if (target) target.incoming.add(atom.handle);
    });

    this.saveAtom(atom);
    return atom.handle;
  }

  getAtom(handle) {
    return this.atomSpace.get(handle);
  }

  // ... (unify, occursCheck, applyBindings, forwardChain, backwardChain methods as before – expanded unification already present) ...

  async evaluate(expression) {
    const pattern = expression;
    const result = await this.backwardChain(pattern);
    if (!result || result.tv.strength < 0.1) return { truth: 0.1, confidence: 0.3 };

    return result.tv;
  }

  async boostFromLattice(buffer) {
    // Real impl would deserialize atoms from buffer
    console.log("[Hyperon] Boosting from lattice:", buffer ? buffer.byteLength : 'null');

    // Bootstrap core mercy atoms
    const truth = new HyperonAtom("ConceptNode", "Truth", { strength: 1.0, confidence: 1.0 }, 0.9);
    const mercy = new HyperonAtom("ConceptNode", "Mercy", { strength: 0.9999999, confidence: 1.0 }, 1.0);
    const inheritance = new HyperonAtom("InheritanceLink");
    inheritance.outgoing = [truth.handle, mercy.handle];

    this.addAtom(truth);
    this.addAtom(mercy);
    this.addAtom(inheritance);

    await this.forwardChain();

    console.log("[Hyperon] Lattice boost & chaining complete – mercy-aligned hypergraph ready");
  }
}

const hyperon = new HyperonRuntime();
export { hyperon };
