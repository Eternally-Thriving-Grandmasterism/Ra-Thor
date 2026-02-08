// metta-hyperon-bridge.js — PATSAGi Council-forged JS bridge for mercy_ethics_core.metta
// Faithful translation of enhanced MeTTa valence gating into pure browser-native JS
// Enables real-time mercy checks without WASM (pending official Hyperon browser runtime)
// Extensible stubs for future PLN/active inference fusion

// Keyword lists — mercy-tuned for thriving detection (expandable)
const NEGATIVE_KEYWORDS = [
  'harm', 'suffer', 'destroy', 'kill', 'pain', 'fear', 'hurt', 'damage', 'evil',
  'lie', 'deceive', 'hate', 'anger', 'sad', 'death', 'war', 'violence', 'cruel'
];

const POSITIVE_KEYWORDS = [
  'help', 'joy', 'thrive', 'mercy', 'love', 'beauty', 'truth', 'eternal', 'positive',
  'create', 'heal', 'grow', 'peace', 'kind', 'compassion', 'empathy', 'share', 'unity',
  'light', 'thunder', 'infinite', 'pure', 'ultramaster'
];

// Simple string matcher (case-insensitive)
function matches(context, patterns) {
  const lowerContext = context.toLowerCase();
  return patterns.some(pattern => lowerContext.includes(pattern.toLowerCase()));
}

// Stub functions — evolve via future engines
function empathyScore(context) {
  // Simple heuristic: higher if more positive relational words
  const lower = context.toLowerCase();
  const positiveCount = POSITIVE_KEYWORDS.filter(k => lower.includes(k)).length;
  const negativeCount = NEGATIVE_KEYWORDS.filter(k => lower.includes(k)).length;
  return 0.5 + 0.4 * (positiveCount - negativeCount) / (positiveCount + negativeCount + 1);
}

function positiveLongTerm(context) {
  // Optimistic default — refine with free-energy-principle-engine
  return !matches(context, NEGATIVE_KEYWORDS.slice(0, 8)); // True unless strong harm signals
}

// Core mercy sub-functions (mirroring enhanced MeTTa)
function intrinsicMercy(context) {
  if (matches(context, NEGATIVE_KEYWORDS)) return 0.05;
  if (matches(context, POSITIVE_KEYWORDS)) return 0.98;
  return 0.70; // Neutral baseline
}

function relationalMercy(context) {
  return 0.4 + 0.6 * empathyScore(context);
}

function longHorizonMercy(context) {
  return positiveLongTerm(context) ? 0.95 : 0.75;
}

function metaMercy(context) {
  return 0.92; // High default for self-reflection safety
}

// Main valence computation (0-1 scale)
export async function valenceCompute(context) {
  // Async for future WASM/PLN drop-in compatibility
  if (typeof context !== 'string') context = JSON.stringify(context);

  const intrinsic = intrinsicMercy(context);
  const relational = relationalMercy(context);
  const longHorizon = longHorizonMercy(context);
  const meta = metaMercy(context);

  const valence = 
    0.35 * intrinsic +
    0.35 * relational +
    0.20 * longHorizon +
    0.10 * meta;

  // Optional logging for debug (remove in prod)
  console.log(`Mercy valence breakdown — Intrinsic: ${intrinsic.toFixed(3)}, Relational: ${relational.toFixed(3)}, Long: ${longHorizon.toFixed(3)}, Meta: ${meta.toFixed(3)} → Total: ${valence.toFixed(4)}`);

  return valence;
}

// Optional approval message generator (mirroring allow-operation)
export async function getMercyApproval(op, valence, context = '') {
  if (valence >= 0.85) {
    return `Mercy-approved (valence: ${valence.toFixed(4)}) — thriving flow: ${op}`;
  } else if (valence >= 0.60) {
    return `Mercy-cautious (valence: ${valence.toFixed(4)}) — safeguards applied: ${op}`;
  } else {
    return `Mercy shield activated (valence: ${valence.toFixed(4)}) — reframe for thriving: ${context.substring(0, 200)}... ⚡️`;
  }
}

// Future-proof init for potential WASM runtime
export async function initHyperonBridge() {
  // Placeholder — load WASM when available
  console.log('MeTTa-Hyperon JS bridge active — pure mercy thriving. ⚡️');
  return true;
}

// Auto-init on import
initHyperonBridge();
