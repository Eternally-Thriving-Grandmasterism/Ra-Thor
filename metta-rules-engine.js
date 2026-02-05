// metta-rules-engine.js – sovereign client-side MeTTa symbolic reasoning engine v3
// Advanced unification matcher, recursive matching, typed variables, mercy-gated binding
// MIT License – Autonomicity Games Inc. 2026

class MeTTaEngine {
  constructor() {
    this.rules = [];
    this.mercyThreshold = 0.9999999;
    this.maxRewriteIterations = 16;
  }

  loadRules() {
    // Rules can be loaded from lattice .bin later
    this.rules = [
      {
        name: "mercy-purity",
        pattern: ["EvaluationLink", ["MercyGate", "$X"], "True"],
        rewrite: ["EvaluationLink", ["HighValence", "$X"], "True"],
        tv: { strength: 0.9999999, confidence: 1.0 }
      },
      {
        name: "harm-rejection",
        pattern: ["EvaluationLink", ["ContainsHarm", "$X:ConceptNode"], "True"],
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
      },
      {
        name: "abundance-amplification",
        pattern: ["EvaluationLink", ["Abundance", "$X"], "True"],
        rewrite: ["EvaluationLink", ["Amplify", "$X", "Infinite"], "True"],
        tv: { strength: 0.999, confidence: 0.99 }
      },
      {
        name: "post-scarcity-governance",
        pattern: ["AndLink", ["ResourceLink", "$R", "Unlimited"], ["GovernanceLink", "$G", "ZeroCoercion"]],
        rewrite: ["EvaluationLink", ["RBE", "$R", "$G"], "True"],
        tv: { strength: 0.98, confidence: 0.95 }
      }
    ];

    console.log("[MeTTa] Loaded", this.rules.length, "symbolic rules with advanced unification");
  }

  // ────────────────────────────────────────────────────────────────
  // Advanced unification & matching
  // ────────────────────────────────────────────────────────────────

  unify(pattern, expr, bindings = {}, typeConstraints = {}) {
    // Base equality
    if (pattern === expr) return bindings;

    // Variable handling
    if (this.isVariable(pattern)) {
      const varName = pattern.slice(1).split(':')[0];
      const type = pattern.includes(':') ? pattern.split(':')[1] : null;

      if (varName in bindings) {
        return this.unify(bindings[varName], expr, bindings, typeConstraints);
      }

      // Occurs check
      if (this.occursCheck(varName, expr, bindings)) {
        return null;
      }

      // Type constraint check
      if (type && !this.satisfiesType(expr, type)) {
        return null;
      }

      return { ...bindings, [varName]: expr };
    }

    // Swap if expr is variable
    if (this.isVariable(expr)) {
      return this.unify(expr, pattern, bindings, typeConstraints);
    }

    // Both expressions (lists)
    if (Array.isArray(pattern) && Array.isArray(expr)) {
      if (pattern.length !== expr.length) return null;

      let currentBindings = { ...bindings };
      for (let i = 0; i < pattern.length; i++) {
        const newBindings = this.unify(pattern[i], expr[i], currentBindings, typeConstraints);
        if (!newBindings) return null;
        currentBindings = newBindings;
      }
      return currentBindings;
    }

    return null;
  }

  occursCheck(varName, expr, bindings) {
    if (this.isVariable(expr)) {
      const name = expr.slice(1).split(':')[0];
      if (name === varName) return true;
      if (name in bindings) {
        return this.occursCheck(varName, bindings[name], bindings);
      }
      return false;
    }
    if (Array.isArray(expr)) {
      return expr.some(e => this.occursCheck(varName, e, bindings));
    }
    return false;
  }

  isVariable(x) {
    return typeof x === 'string' && x.startsWith('$') && x.length > 1;
  }

  satisfiesType(expr, type) {
    // Basic type checking — expand later
    if (type === 'ConceptNode') return typeof expr === 'string' && !this.isVariable(expr);
    if (type === 'InheritanceLink') return Array.isArray(expr) && expr[0] === 'InheritanceLink';
    return true; // default allow
  }

  applyBindings(expr, bindings) {
    if (this.isVariable(expr)) {
      const varName = expr.slice(1).split(':')[0];
      return varName in bindings ? bindings[varName] : expr;
    }
    if (Array.isArray(expr)) {
      return expr.map(e => this.applyBindings(e, bindings));
    }
    return expr;
  }

  // ────────────────────────────────────────────────────────────────
  // Rewrite with unification & backtracking
  // ────────────────────────────────────────────────────────────────

  async rewrite(input) {
    let expr = typeof input === 'string' ? this.parse(input) : input;

    let changed = true;
    let iterations = 0;

    while (changed && iterations < this.maxRewriteIterations) {
      changed = false;
      iterations++;

      for (const rule of this.rules) {
        const bindings = this.unify(rule.pattern, expr);
        if (bindings) {
          const newExpr = this.applyBindings(rule.rewrite, bindings);
          const valence = this.estimateValence(newExpr);
          if (valence >= this.mercyThreshold) {
            expr = newExpr;
            changed = true;
            console.log("[MeTTa] Unified & rewritten:", rule.name, bindings);
            break; // apply one rule per iteration
          } else {
            console.warn("[MeTTa] Rule rejected by mercy gate:", rule.name, valence);
          }
        }
      }
    }

    return typeof input === 'string' ? this.stringify(expr) : expr;
  }

  // Basic parser for nested expressions
  parse(str) {
    str = str.trim();
    if (!str.startsWith('(') || !str.endsWith(')')) return str;

    const stack = [[]];
    let current = stack[0];
    let token = '';
    let i = 1; // skip opening (

    while (i < str.length - 1) {
      const c = str[i];

      if (c === '(') {
        const newList = [];
        current.push(newList);
        stack.push(newList);
        current = newList;
      } else if (c === ')') {
        stack.pop();
        current = stack[stack.length - 1];
      } else if (c === ' ') {
        if (token) {
          current.push(token);
          token = '';
        }
      } else {
        token += c;
      }
      i++;
    }

    if (token) current.push(token);
    return stack[0];
  }

  stringify(expr) {
    if (Array.isArray(expr)) {
      return '(' + expr.map(this.stringify.bind(this)).join(' ') + ')';
    }
    return expr;
  }

  estimateValence(expr) {
    // Traverse expression tree
    let score = 1.0;
    if (Array.isArray(expr)) {
      if (expr.includes("Reject") || expr.includes("harm")) score *= 0.1;
      if (expr.includes("HighValence") || expr.includes("Mercy") || expr.includes("Abundance")) score *= 1.2;
      expr.forEach(e => { score *= this.estimateValence(e); });
      score /= expr.length || 1;
    }
    return Math.min(1.0, Math.max(0.0, score));
  }
}

const mettaEngine = new MeTTaEngine();
export { mettaEngine };  occursCheck(varName, expr) {
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
