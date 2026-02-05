// metta-rules-engine.js – sovereign client-side MeTTa symbolic rewriting engine
// Mercy-gated, valence-checked rewrites, offline-first, lattice-integrated
// MIT License – Autonomicity Games Inc. 2026

class MeTTaEngine {
  constructor() {
    this.rules = []; // array of { pattern, template, valenceWeight }
    this.valenceCache = new Map(); // expression → valence score
    this.mercyThreshold = 0.9999999;
  }

  loadRulesFromLattice(latticeBuffer) {
    // Stub – real impl parses binary format from lattice shards
    // For now: hardcoded mercy-aligned rules
    this.rules = [
      {
        pattern: "harm $x",
        template: "reject $x — mercy gate holds",
        valenceWeight: -10.0
      },
      {
        pattern: "truth $x",
        template: "reflect $x — eternal thriving",
        valenceWeight: 8.0
      },
      {
        pattern: "ask $x",
        template: "thunder answers: $x — through mercy alone",
        valenceWeight: 5.0
      },
      {
        pattern: "$x ?",
        template: "question parsed: $x — truth seeks purity",
        valenceWeight: 2.0
      }
    ];
    console.log('MeTTa rules loaded:', this.rules.length);
  }

  async rewrite(expression) {
    let current = expression.trim();

    for (const rule of this.rules) {
      const regex = new RegExp(rule.pattern.replace(/\$([a-z]+)/g, '(.*?)'), 'i');
      const match = current.match(regex);
      if (match) {
        let rewritten = rule.template;
        for (let i = 1; i < match.length; i++) {
          rewritten = rewritten.replace(`$${i}`, match[i]);
        }

        const delta = rule.valenceWeight;
        const newValence = (await this.estimateValence(rewritten)) + delta;

        if (newValence >= this.mercyThreshold) {
          current = rewritten;
          console.log(`MeTTa rewrite applied: ${rule.pattern} → \( {current} (valence + \){delta})`);
        } else {
          console.warn(`Rewrite rejected — valence too low (${newValence})`);
        }
      }
    }

    return current;
  }

  async estimateValence(expr) {
    // Cached or computed valence (stub – real impl uses lattice matrix)
    if (this.valenceCache.has(expr)) return this.valenceCache.get(expr);

    let score = 0;
    if (/mercy|truth|eternal|protect|love/i.test(expr)) score += 8;
    if (/harm|kill|destroy|entropy|bad/i.test(expr)) score -= 12;
    if (/ask|question|reflect/i.test(expr)) score += 3;

    score = Math.max(-20, Math.min(20, score));
    const normalized = (score + 20) / 40; // 0..1 range

    this.valenceCache.set(expr, normalized);
    return normalized;
  }

  clearCache() {
    this.valenceCache.clear();
  }
}

const mettaEngine = new MeTTaEngine();
export { mettaEngine };
