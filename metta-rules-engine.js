// metta-rules-engine.js – sovereign client-side MeTTa symbolic reasoning engine v2
// Real unification-based pattern matcher, recursive matching, mercy-gated binding
// MIT License – Autonomicity Games Inc. 2026

class MeTTaEngine {
  constructor() {
    this.rules = [];
    this.mercyThreshold = 0.9999999;
  }

  loadRules() {
    // Real rules can be loaded from lattice .bin later
    this.rules = [
      {
        name: "mercy-purity",
        pattern: ["EvaluationLink", ["MercyGate", "$X"], "True"],
        rewrite: ["EvaluationLink", ["HighValence", "$X"], "True"],
        tv: { strength: 0.9999999, confidence: 1.0 }
      },
      {
        name: "harm-rejection",
        pattern: ["EvaluationLink", ["ContainsHarm", "$X"], "True"],
        rewrite: ["EvaluationLink", ["Reject", "$X"], "True"],
        tv: { strength: 1.0, confidence: 1.0 }
      },
      {
        name: "truth-reflection",
        pattern: ["ImplicationLink", "$A", "$B"],
        rewrite: ["EvaluationLink", ["Reflects", "$A", "$B"], "True"],
        tv: { strength: 0.95, confidence: 0.9 }
      },
      {
        name: "transitive-inheritance",
        pattern: ["AndLink", ["InheritanceLink", "$X", "$Y"], ["InheritanceLink", "$Y", "$Z"]],
        rewrite: ["InheritanceLink", "$X", "$Z"],
        tv: { strength: 0.9, confidence: 0.85 }
      }
    ];

    console.log("[MeTTa] Loaded", this.rules.length, "symbolic rules with unification");
  }

  // ────────────────────────────────────────────────────────────────
  // Core unification & matching logic
  // ────────────────────────────────────────────────────────────────

  unify(pattern, expr, bindings = {}) {
    // Base cases
    if (pattern === expr) return bindings;

    // Variable in pattern
    if (this.isVariable(pattern)) {
      const varName = pattern.slice(1);
      if (varName in bindings) {
        return this.unify(bindings[varName], expr, bindings);
      }
      // Occurs check
      if (this.occursCheck(varName, expr)) {
        return null; // occurs-check failure
      }
      return { ...bindings, [varName]: expr };
    }

    // Variable in expression (swap)
    if (this.isVariable(expr)) {
      return this.unify(expr, pattern, bindings);
    }

    // Both are lists (expressions)
    if (Array.isArray(pattern) && Array.isArray(expr)) {
      if (pattern.length !== expr.length) return null;

      let currentBindings = { ...bindings };
      for (let i = 0; i < pattern.length; i++) {
        const newBindings = this.unify(pattern[i], expr[i], currentBindings);
        if (!newBindings) return null;
        currentBindings = newBindings;
      }
      return currentBindings;
    }

    // No unification possible
    return null;
  }

  occursCheck(varName, expr) {
    if (this.isVariable(expr)) {
      return expr.slice(1) === varName;
    }
    if (Array.isArray(expr)) {
      return expr.some(e => this.occursCheck(varName, e));
    }
    return false;
  }

  isVariable(x) {
    return typeof x === 'string' && x.startsWith('$') && x.length > 1;
  }

  applyBindings(expr, bindings) {
    if (this.isVariable(expr)) {
      const varName = expr.slice(1);
      return varName in bindings ? bindings[varName] : expr;
    }
    if (Array.isArray(expr)) {
      return expr.map(e => this.applyBindings(e, bindings));
    }
    return expr;
  }

  // ────────────────────────────────────────────────────────────────
  // Rewrite using unification
  // ────────────────────────────────────────────────────────────────

  async rewrite(input) {
    // Input can be string or expression tree
    let expr = typeof input === 'string' ? this.parse(input) : input;

    let changed = true;
    let iterations = 0;
    const maxIterations = 12;

    while (changed && iterations < maxIterations) {
      changed = false;
      iterations++;

      for (const rule of this.rules) {
        const bindings = this.unify(rule.pattern, expr);
        if (bindings) {
          const newExpr = this.applyBindings(rule.rewrite, bindings);
          // Mercy gate on new expression
          const valence = this.estimateValence(newExpr);
          if (valence >= this.mercyThreshold) {
            expr = newExpr;
            changed = true;
            console.log("[MeTTa] Unified & rewritten:", rule.name);
            break;
          } else {
            console.warn("[MeTTa] Rule rejected by mercy gate:", rule.name, valence);
          }
        }
      }
    }

    return typeof input === 'string' ? this.stringify(expr) : expr;
  }

  // Very basic parser – real impl would use proper MeTTa parser
  parse(str) {
    // Naive: split on spaces, treat $ as variables
    const tokens = str.trim().split(/\s+/);
    if (tokens.length === 1) return tokens[0];
    return tokens;
  }

  stringify(expr) {
    if (Array.isArray(expr)) {
      return expr.join(' ');
    }
    return expr;
  }

  estimateValence(expr) {
    // Dummy – real impl would traverse lattice
    if (Array.isArray(expr)) {
      if (expr.includes("Reject") || expr.includes("harm")) return 0.1;
      if (expr.includes("HighValence") || expr.includes("Mercy")) return 0.9999999;
    }
    return 0.8;
  }
}

const mettaEngine = new MeTTaEngine();
export { mettaEngine };
