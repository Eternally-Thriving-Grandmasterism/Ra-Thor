'use strict';
const createRouter = require('@arangodb/foxx/router');
const joi = require('joi');
const db = require('@arangodb').db;
const aql = require('aql');

const router = createRouter();
module.context.use(router);

// Eternal mercy constants
const VALENCE_THRESHOLD = 0.9999999;

// ... existing routes: /validate, /insert-atom, /high-valence ...

// NEW: POST /mercy/metta-eval - Evaluate .metta expression mercy-gated
router.post('/metta-eval', (req, res) => {
  const { expression, valence, context = 'default' } = req.body;

  if (typeof valence !== 'number' || valence < VALENCE_THRESHOLD) {
    res.status(403).send({ error: 'Mercy shield: low valence — .metta eval rejected' });
    return;
  }

  try {
    // Simple JS MeTTa evaluator subset
    const result = evaluateMetta(expression, loadRulesFromSpace(context));

    // Optional: store eval result in collection
    const stored = db._executeTransaction({
      collections: { write: 'MettaEvals' },
      action: function () {
        const db = require('@arangodb').db;
        return db.MettaEvals.save({
          input: expression,
          output: JSON.stringify(result),
          valence,
          context,
          timestamp: new Date().toISOString()
        });
      }
    });

    res.send({
      success: true,
      result,
      storedId: stored._key,
      message: 'Mercy-approved .metta evaluation complete'
    });
  } catch (e) {
    res.status(500).send({ error: 'MeTTa eval failed: ' + e.message });
  }
})
.body(joi.object({
  expression: joi.string().required(),  // e.g. "(+ 1 2)" or "! (= (foo) bar)"
  valence: joi.number().required(),
  context: joi.string().optional()
}), '.metta expression + valence')
.response(['application/json'], 'Eval result')
.summary('Mercy-gated .metta evaluation')
.description('Server-side MeTTa eval with valence gate + unification basics');

// Helper: Load equality rules from collection (as space)
function loadRulesFromSpace(context) {
  const cursor = db._query(aql`
    FOR rule IN MettaRules
    FILTER rule.context == ${context}
    RETURN rule
  `);
  return cursor.toArray();  // [{ left: "(+ $a $b)", right: "(add $a $b)" }, ...]
}

// Core JS MeTTa evaluator subset (recursive reduction)
function evaluateMetta(exprStr, rules) {
  // Basic parser: naive S-expr to nested array
  function parse(s) {
    s = s.trim();
    if (s.startsWith('(') && s.endsWith(')')) {
      s = s.slice(1, -1).trim();
      const parts = [];
      let depth = 0, start = 0;
      for (let i = 0; i < s.length; i++) {
        if (s[i] === '(') depth++;
        if (s[i] === ')') depth--;
        if (depth === 0 && s[i] === ' ') {
          parts.push(parse(s.slice(start, i)));
          start = i + 1;
        }
      }
      parts.push(parse(s.slice(start)));
      return parts.filter(p => p !== '');
    }
    return s;  // atom
  }

  const expr = parse(exprStr);

  // Immediate eval marker !
  if (Array.isArray(expr) && expr[0] === '!') {
    return reduce(expr.slice(1), rules);
  }

  // Normal add to "space" or reduce
  return reduce(expr, rules);
}

// Reduction engine: apply equality rules recursively
function reduce(expr, rules) {
  if (!Array.isArray(expr)) return expr;  // atom

  // Try unify/apply rules
  for (const rule of rules) {
    if (unify(expr, parse(rule.left))) {
      return substitute(expr, parse(rule.left), parse(rule.right));
    }
  }

  // Recurse on children
  return expr.map(child => reduce(child, rules));
}

// Simple unify (pattern match)
function unify(a, b) {
  if (a === b) return true;
  if (typeof a === 'string' && a.startsWith('$')) return true;  // var
  if (!Array.isArray(a) || !Array.isArray(b) || a.length !== b.length) return false;
  return a.every((x, i) => unify(x, b[i]));
}

// Substitute vars (placeholder simplistic)
function substitute(expr, pattern, replacement) {
  // In full: bind vars from unify, substitute
  // Here: naive replace if match
  return replacement;  // expand for production
}
