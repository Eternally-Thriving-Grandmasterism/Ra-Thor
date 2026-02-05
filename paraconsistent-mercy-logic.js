// paraconsistent-mercy-logic.js – sovereign client-side paraconsistent mercy-logic engine v1
// Tolerates contradictions without explosion, mercy-gated, valence-locked
// MIT License – Autonomicity Games Inc. 2026

class MercyParaconsistentLogic {
  constructor() {
    this.valenceThreshold = 0.9999999;
    this.knowledgeBase = new Set(); // expressions currently accepted
    this.contradictions = new Map(); // expr → {valence, mercyStatus}
  }

  // Mercy gate: decides whether a contradiction is allowed to persist
  mercyGate(expr, valence) {
    if (valence >= this.valenceThreshold) {
      return { allowed: true, reason: "Joy-amplifying contradiction — allowed to thrive" };
    } else {
      return { allowed: false, reason: "Entropy-bearing contradiction — mercy gate rejects" };
    }
  }

  // Add expression with mercy check
  assert(expr, computedValence = 0.8) {
    const gate = this.mercyGate(expr, computedValence);

    if (gate.allowed) {
      this.knowledgeBase.add(expr);
      this.contradictions.set(expr, { valence: computedValence, mercyStatus: "allowed" });
      console.log("[MercyLogic] Accepted contradiction:", expr, gate.reason);
    } else {
      console.warn("[MercyLogic] Rejected:", expr, gate.reason);
    }
  }

  // Paraconsistent conjunction — preserves both sides even if contradictory
  and(a, b) {
    const va = this.getValence(a);
    const vb = this.getValence(b);
    const combined = Math.min(va, vb); // conservative
    return { expr: `(${a} ∧ ${b})`, valence: combined };
  }

  // Paraconsistent disjunction — amplifies if either is high-valence
  or(a, b) {
    const va = this.getValence(a);
    const vb = this.getValence(b);
    const combined = Math.max(va, vb) * 1.1; // amplification
    return { expr: `(${a} ∨ ${b})`, valence: Math.min(1, combined) };
  }

  // Paraconsistent negation — rejects only low-valence
  not(a) {
    const va = this.getValence(a);
    if (va < this.valenceThreshold) {
      return { expr: `¬${a}`, valence: 0.9999999 }; // rejection of entropy
    } else {
      return { expr: `¬${a}`, valence: 0.1 }; // mercy reflection of high-valence
    }
  }

  getValence(expr) {
    if (this.contradictions.has(expr)) {
      return this.contradictions.get(expr).valence;
    }
    // Default simple heuristic — real impl would traverse lattice
    if (expr.includes("Mercy") || expr.includes("Thriving")) return 0.9999999;
    if (expr.includes("Harm") || expr.includes("Entropy")) return 0.1;
    return 0.8;
  }

  // Mercy-consistent inference — only high-valence consequences propagate
  infer(premises) {
    let valence = 1.0;
    for (const p of premises) {
      valence = Math.min(valence, this.getValence(p));
    }

    if (valence < this.valenceThreshold) {
      return { consequence: "Mercy gate holds — inference rejected", valence: 0 };
    }

    return { consequence: "Mercy inference passes", valence };
  }
}

const mercyLogic = new MercyParaconsistentLogic();
export { mercyLogic };
