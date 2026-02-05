// hyperon-runtime.js – sovereign client-side Hyperon hypergraph atomspace & full PLN engine v18
// Titan Methane Life Detection Protocol subgraph, exotic solvent rules, persistent DB
// MIT License – Autonomicity Games Inc. 2026

class HyperonAtom {
  constructor(type, name = null, tv = { strength: 0.5, confidence: 0.5 }, sti = 0.1, lti = 0.01, handle = null) {
    this.type = type;
    this.name = name;
    this.tv = tv;
    this.sti = sti;
    this.lti = lti;
    this.outgoing = [];
    this.incoming = new Set();
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
    this.atomSpace = new Map();
    this.nextHandle = 0;
    this.mercyThreshold = 0.9999999;
    this.maxChainDepth = 12;
    this.attentionDecay = 0.95;
    this.db = null;
    this.dbName = "rathorHyperonDB";
    this.storeName = "atoms";

    this.plnRules = [
      // ... (previous 16 core + Enceladus rules unchanged) ...

      // 17. Methane-Solvent Detection
      {
        name: "Methane-Solvent-Detection",
        premises: ["DetectionLink", ["Azotosome", "$X"], "True"],
        conclusion: ["BiosignatureLink", "$X", "Methane"],
        tvCombiner: (tvs) => ({
          strength: 0.95,
          confidence: 0.92
        }),
        priority: 24
      },
      // 18. Cryogenic Quarantine
      {
        name: "Cryogenic-Quarantine",
        premises: ["DetectionLink", ["Biosignature", "$X"], "True"],
        conclusion: ["QuarantineLink", "$X", "LowTempMercy"],
        tvCombiner: (tvs) => ({
          strength: 0.999,
          confidence: 0.99
        }),
        priority: 23
      },
      // 19. Tholin Recycling Loop
      {
        name: "Tholin-Recycling-Loop",
        premises: ["ResourceLink", "Tholins", "Unlimited"],
        conclusion: ["EvaluationLink", ["CradleToCradle", "Tholins"], "True"],
        tvCombiner: (tvs) => ({
          strength: 0.98,
          confidence: 0.95
        }),
        priority: 21
      },
      // 20. Exotic Uplift Protocol
      {
        name: "Exotic-Uplift-Protocol",
        premises: ["QuarantineLink", "$X", "Mercy"],
        conclusion: ["UpliftLink", "$X", "SolventThriving"],
        tvCombiner: (tvs) => ({
          strength: 0.9999999,
          confidence: 1.0
        }),
        priority: 22
      }
    ].sort((a, b) => b.priority - a.priority);
  }

  // ... (init, openDB, loadFromDB, saveAtom, newHandle, addAtom, getAtom, unify, occursCheck, applyBindings, forwardChain, backwardChain, combineTV, evaluate unchanged) ...

  async boostTitanProtocol() {
    console.log("[Hyperon] Boosting Titan Methane Life Detection Protocol subgraph...");

    // Core Titan atoms
    const titan = new HyperonAtom("ConceptNode", "Titan", { strength: 0.99, confidence: 0.98 }, 0.9);
    const methaneSea = new HyperonAtom("ConceptNode", "MethaneSea", { strength: 0.96, confidence: 0.93 }, 0.85);
    const azotosome = new HyperonAtom("ConceptNode", "Azotosome", { strength: 0.92, confidence: 0.88 }, 0.8);
    const tholin = new HyperonAtom("ConceptNode", "Tholin", { strength: 0.95, confidence: 0.92 }, 0.85);
    const quarantine = new HyperonAtom("ConceptNode", "Quarantine", { strength: 0.999, confidence: 0.99 }, 0.95);
    const uplift = new HyperonAtom("ConceptNode", "Uplift", { strength: 0.9999999, confidence: 1.0 }, 1.0);

    this.addAtom(titan);
    this.addAtom(methaneSea);
    this.addAtom(azotosome);
    this.addAtom(tholin);
    this.addAtom(quarantine);
    this.addAtom(uplift);

    // Links
    const seaOf = new HyperonAtom("EvaluationLink");
    seaOf.outgoing = [methaneSea.handle, titan.handle];
    this.addAtom(seaOf);

    const azotoIn = new HyperonAtom("InheritanceLink");
    azotoIn.outgoing = [azotosome.handle, methaneSea.handle];
    this.addAtom(azotoIn);

    const tholinResource = new HyperonAtom("ResourceLink");
    tholinResource.outgoing = [tholin.handle, "Unlimited"];
    this.addAtom(tholinResource);

    const quarantineMercy = new HyperonAtom("ImplicationLink");
    quarantineMercy.outgoing = [azotosome.handle, quarantine.handle];
    this.addAtom(quarantineMercy);

    const upliftThriving = new HyperonAtom("ImplicationLink");
    upliftThriving.outgoing = [quarantine.handle, uplift.handle];
    this.addAtom(upliftThriving);

    await this.forwardChain();

    console.log("[Hyperon] Titan Methane Life Detection Protocol subgraph boosted & chained");
  }
}

const hyperon = new HyperonRuntime();
export { hyperon };
