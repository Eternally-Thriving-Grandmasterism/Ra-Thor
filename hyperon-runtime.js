// hyperon-runtime.js – sovereign client-side Hyperon hypergraph atomspace & full PLN engine v20
// JWST data ingestion, Kuiper Belt + Titan + Enceladus protocols, persistent DB
// MIT License – Autonomicity Games Inc. 2026

// ... (HyperonAtom class unchanged) ...

class HyperonRuntime {
  constructor() {
    // ... (previous constructor unchanged) ...

    this.plnRules = [
      // ... (previous 24 rules unchanged) ...

      // 25. JWST Spectral Detection
      {
        name: "JWST-Spectral-Detection",
        premises: ["DetectionLink", ["JWST_Spectra", "$X"], "True"],
        conclusion: ["CompositionLink", "$X", "IceVolatiles"],
        tvCombiner: (tvs) => ({
          strength: 0.97,
          confidence: 0.94
        }),
        priority: 26
      },
      // 26. JWST Valence Amplification
      {
        name: "JWST-Valence-Amplification",
        premises: ["CompositionLink", "$X", "IceVolatiles"],
        conclusion: ["HighValenceResource", "$X"],
        tvCombiner: (tvs) => ({
          strength: 0.999,
          confidence: 0.98
        }),
        priority: 24
      }
    ].sort((a, b) => b.priority - a.priority);
  }

  // ... (init, openDB, loadFromDB, saveAtom, etc. unchanged) ...

  async ingestJWSTData(targetName) {
    console.log("[Hyperon] Ingesting JWST data for:", targetName);

    const jwstData = await jwstFetcher.fetchObservation(targetName);
    if (jwstData.error) {
      console.warn("[Hyperon] JWST data rejected:", jwstData.error);
      return;
    }

    const target = new HyperonAtom("ConceptNode", targetName, { strength: 0.99, confidence: 0.98 }, 0.9);
    const spectra = new HyperonAtom("ConceptNode", "JWST_Spectra", { strength: jwstData.valenceScore, confidence: 0.95 }, 0.85);

    this.addAtom(target);
    this.addAtom(spectra);

    const detection = new HyperonAtom("DetectionLink");
    detection.outgoing = [spectra.handle, target.handle];
    this.addAtom(detection);

    const composition = new HyperonAtom("CompositionLink");
    composition.outgoing = [target.handle, "IceVolatiles"];
    this.addAtom(composition);

    await this.forwardChain();

    console.log("[Hyperon] JWST data ingested & chained into Kuiper lattice");
  }
}

const hyperon = new HyperonRuntime();
export { hyperon };
