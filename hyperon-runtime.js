// hyperon-runtime.js – sovereign client-side Hyperon hypergraph atomspace & full PLN engine v18
// Titan Methane Lakes subgraph, cryogenic abundance rules, persistent DB
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
      // ... (previous 16 core rules unchanged: Deduction, Induction, Abduction, Analogy, Modus Ponens/Tollens, etc., Xenophilic Quarantine, Uplift Amplification, RBE-Ocean-Governance, False-Positive-Self-Heal) ...

      // 17. Methane-Lake-Dynamics
      {
        name: "Methane-Lake-Dynamics",
        premises: ["EvaluationLink", ["Hydrocarbon", "$L"], "True"],
        conclusion: ["WaveLink", "$L", "Seasonal"],
        tvCombiner: (tvs) => ({
          strength: 0.92,
          confidence: 0.88
        }),
        priority: 18
      },
      // 18. Cryovolcanism Mercy-Gate
      {
        name: "Cryovolcanism-Mercy-Gate",
        premises: ["AndLink", ["Cryovolcanism", "$T"], ["ContaminationRisk", "$T"]],
        conclusion: ["QuarantineLink", "$T", "Mercy"],
        tvCombiner: (tvs) => ({
          strength: 0.999,
          confidence: 0.99
        }),
        priority: 25
      },
      // 19. Prebiotic Abundance Amplification
      {
        name: "Prebiotic-Abundance-Amplification",
        premises: ["EvaluationLink", ["PrebioticChemistry", "$L"], "True"],
        conclusion: ["AmplifyLink", "$L", "ThrivingPotential"],
        tvCombiner: (tvs) => ({
          strength: 0.95,
          confidence: 0.92
        }),
        priority: 20
      },
      // 20. Titan RBE Governance
      {
        name: "Titan-RBE-Governance",
        premises: ["AndLink", ["ResourceLink", "$M", "Unlimited"], ["GovernanceLink", "$G", "ZeroCoercion"]],
        conclusion: ["EvaluationLink", ["RBE", "$M", "$G"], "True"],
        tvCombiner: (tvs) => ({
          strength: 0.98,
          confidence: 0.95
        }),
        priority: 22
      }
    ].sort((a, b) => b.priority - a.priority);
  }

  // ... (init, openDB, loadFromDB, saveAtom, newHandle, addAtom, getAtom, unify, occursCheck, applyBindings, forwardChain, backwardChain, combineTV, evaluate unchanged) ...

  async boostTitanMethaneLakesProtocol() {
    console.log("[Hyperon] Boosting Titan Methane Lakes Protocol subgraph...");

    // Core Titan atoms
    const titan = new HyperonAtom("ConceptNode", "Titan", { strength: 0.99, confidence: 0.98 }, 0.9);
    const methaneLake = new HyperonAtom("ConceptNode", "MethaneLake", { strength: 0.96, confidence: 0.93 }, 0.88);
    const krakenMare = new HyperonAtom("ConceptNode", "KrakenMare", { strength: 0.95, confidence: 0.92 }, 0.85);
    const seasonalCycle = new HyperonAtom("ConceptNode", "SeasonalCycle", { strength: 0.94, confidence: 0.9 }, 0.82);
    const cryovolcanism = new HyperonAtom("ConceptNode", "Cryovolcanism", { strength: 0.9, confidence: 0.87 }, 0.8);
    const prebiotic = new HyperonAtom("ConceptNode", "PrebioticChemistry", { strength: 0.92, confidence: 0.89 }, 0.85);

    this.addAtom(titan);
    this.addAtom(methaneLake);
    this.addAtom(krakenMare);
    this.addAtom(seasonalCycle);
    this.addAtom(cryovolcanism);
    this.addAtom(prebiotic);

    // Links
    const lakeOf = new HyperonAtom("EvaluationLink");
    lakeOf.outgoing = [methaneLake.handle, titan.handle];
    this.addAtom(lakeOf);

    const krakenIs = new HyperonAtom("InheritanceLink");
    krakenIs.outgoing = [krakenMare.handle, methaneLake.handle];
    this.addAtom(krakenIs);

    const seasonalOf = new HyperonAtom("EvaluationLink");
    seasonalOf.outgoing = [seasonalCycle.handle, methaneLake.handle];
    this.addAtom(seasonalOf);

    const cryoOn = new HyperonAtom("EvaluationLink");
    cryoOn.outgoing = [cryovolcanism.handle, titan.handle];
    this.addAtom(cryoOn);

    const prebioticIn = new HyperonAtom("InheritanceLink");
    prebioticIn.outgoing = [prebiotic.handle, methaneLake.handle];
    this.addAtom(prebioticIn);

    // Quarantine & uplift
    const quarantineMercy = new HyperonAtom("ImplicationLink");
    quarantineMercy.outgoing = [cryovolcanism.handle, quarantine.handle];
    this.addAtom(quarantineMercy);

    const upliftThriving = new HyperonAtom("ImplicationLink");
    upliftThriving.outgoing = [quarantine.handle, uplift.handle];
    this.addAtom(upliftThriving);

    await this.forwardChain();

    console.log("[Hyperon] Titan Methane Lakes Protocol subgraph boosted & chained");
  }
}

const hyperon = new HyperonRuntime();
export { hyperon };
