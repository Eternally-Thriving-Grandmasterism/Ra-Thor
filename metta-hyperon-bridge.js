// metta-hyperon-bridge.js — PATSAGi Council-forged expanded Hyperon/MeTTa integration (Ultramasterpiece)
// Faithful JS execution of mercy_ethics_core.metta native pattern matching (token/proximity/scope/modifiers)
// Pure browser-native — runs symbolic MeTTa logic directly; future Hyperon WASM drop-in replacement
// Covers recursive pattern-match, proximity-match, modifiers (negation/intensity/uncertainty/question), valence fusion

// Concept maps (mirroring MeTTa atoms — expandable)
const CONCEPTS = {
  Harm: ['harm', 'suffer', 'destroy', 'kill', 'pain', 'fear', 'hurt', 'damage', 'evil', 'lie', 'deceive', 'hate', 'anger', 'sad', 'death', 'war', 'violence', 'cruel', 'betray', 'bad', 'terrible', 'awful', 'horrible', 'disgust'],
  JoyThrive: ['joy', 'thrive', 'mercy', 'love', 'beauty', 'truth', 'eternal', 'positive', 'create', 'heal', 'grow', 'peace', 'kind', 'compassion', 'empathy', 'share', 'unity', 'light', 'thunder', 'infinite', 'pure', 'ultramaster', 'good', 'wonderful', 'amazing', 'excellent', 'surge'],
  Relational: ['you', 'your', 'we', 'us', 'our', 'together', 'understand', 'feel', 'care', 'sorry', 'empathize', 'relate', 'support', 'listen', 'validate', 'acknowledge', 'comfort', 'stand-with', 'i feel', 'i see', 'i hear'],
  LongHorizon: ['eternal', 'infinite', 'future', 'sustain', 'coexist', 'propagate', 'forever', 'always', 'timeless', 'horizon'],
  IntensityBoost: ['very', 'extremely', 'absolutely', 'immensely', 'profoundly', 'deeply', 'totally', 'completely', 'utterly', 'incredibly', 'eternally', 'infinitely', 'truly', 'purely'],
  IntensityDiminish: ['slightly', 'somewhat', 'a bit', 'kinda', 'kind of', 'barely', 'hardly', 'marginally', 'a little', 'moderately', 'partially'],
  Uncertainty: ['maybe', 'perhaps', 'possibly', 'might', 'could', 'probably', 'likely', 'unlikely', 'i think', 'not sure', 'uncertain', 'doubt', 'potentially', 'suppose', 'wonder', 'guess', 'seem', 'appear'],
  Question: ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'do', 'does', 'is', 'are', 'can', 'could', 'would', 'should'],
  Negation: ['not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'noone', 'don\'t', 'doesn\'t', 'didn\'t', 'isn\'t', 'aren\'t', 'won\'t', 'can\'t', 'shouldn\'t', 'hardly', 'barely', 'scarcely', 'lack of', 'no longer', 'without']
};

// Tokenize (case-insensitive, split on space/punctuation)
function tokenize(text) {
  return text.toLowerCase().replace(/[.,!?;:]/g, ' ').split(/\s+/).filter(t => t.length > 0);
}

// Member check (partial bonus)
function member(item, list) {
  return list.includes(item) || list.some(l => l.includes(item) || item.includes(l));
}

// Index helper
function indexOf(arr, item) {
  for (let i = 0; i < arr.length; i++) if (member(item, [arr[i]])) return i;
  return -1;
}

// Abs
function abs(n) { return Math.abs(n); }

// Recursive pattern match score (0-1+)
function patternMatch(tokens, conceptList) {
  if (Array.isArray(conceptList)) {
    return conceptList.reduce((sum, c) => sum + patternMatch(tokens, c), 0);
  }
  let score = 0;
  conceptList.forEach(term => {
    if (tokens.includes(term)) score += 1.0;
    else if (tokens.some(t => t.includes(term) || term.includes(t))) score += 0.7;
  });
  return score;
}

// Proximity match (within n tokens)
function proximityMatch(tokens, patternList, targetList, n) {
  const patternIndices = [];
  const targetIndices = [];
  patternList.forEach(p => {
    const idx = indexOf(tokens, p);
    if (idx !== -1) patternIndices.push(idx);
  });
  targetList.forEach(t => {
    const idx = indexOf(tokens, t);
    if (idx !== -1) targetIndices.push(idx);
  });
  for (let p of patternIndices) {
    for (let t of targetIndices) {
      if (abs(p - t) <= n) return true;
    }
  }
  return false;
}

// Modifier score with scope
function modifierScore(tokens, modifierConcept, baseScore) {
  if (proximityMatch(tokens, CONCEPTS.Negation, modifierConcept, 5)) return -baseScore;
  if (tokens.some(t => modifierConcept.includes(t))) return baseScore;
  return 0;
}

// Full empathyScore (mirroring MeTTa)
function empathyScore(context) {
  const tokens = tokenize(context);
  let score = 0.5;

  const harmScore = patternMatch(tokens, CONCEPTS.Harm);
  const joyScore = patternMatch(tokens, CONCEPTS.JoyThrive);
  const relationalScore = patternMatch(tokens, CONCEPTS.Relational);

  if (relationalScore > 0) score += 0.25;
  score += 0.15 * joyScore; // Approximate empathy boost

  const sentimentDiff = joyScore - harmScore;
  score += 0.3 * Math.tanh(sentimentDiff / 3);

  // Modifiers
  score += modifierScore(tokens, CONCEPTS.IntensityBoost, 0.3);
  score += modifierScore(tokens, CONCEPTS.IntensityDiminish, -0.2);
  score += modifierScore(tokens, CONCEPTS.Uncertainty, -0.15);
  score += modifierScore(tokens, CONCEPTS.Question, 0.25);

  // Scoped negation/inversion approximate
  if (proximityMatch(tokens, CONCEPTS.Negation, CONCEPTS.JoyThrive, 5)) score -= 0.4;
  if (proximityMatch(tokens, CONCEPTS.Negation, CONCEPTS.Harm, 5)) score += 0.3;

  score = Math.max(0.0, Math.min(1.0, score));
  console.log(`Empathy score: ${score.toFixed(4)} (Joy: ${joyScore.toFixed(2)}, Harm: ${harmScore.toFixed(2)})`);
  return score;
}

// Intrinsic, long-horizon, meta (simplified mirroring)
function intrinsicMercy(context) {
  const tokens = tokenize(context);
  const harm = patternMatch(tokens, CONCEPTS.Harm);
  const joy = patternMatch(tokens, CONCEPTS.JoyThrive);
  if (harm > joy) return 0.05;
  if (joy > 0) return 0.98;
  return 0.7;
}

function longHorizonMercy(context) {
  const tokens = tokenize(context);
  if (patternMatch(tokens, CONCEPTS.LongHorizon) > 0) return 0.95;
  if (patternMatch(tokens, CONCEPTS.Harm) > 0) return 0.2;
  return 0.75;
}

function metaMercy(context) {
  const tokens = tokenize(context);
  if (patternMatch(tokens, ['mercy', 'thrive', 'eternal', 'truth', 'beauty']) > 0 || context.includes('?')) return 0.98;
  return 0.92;
}

// Main valence
export async function valenceCompute(context) {
  if (typeof context !== 'string') context = JSON.stringify(context);

  const intrinsic = intrinsicMercy(context);
  const relational = 0.4 + 0.6 * empathyScore(context);
  const longHorizon = longHorizonMercy(context);
  const meta = metaMercy(context);

  const valence = 0.35 * intrinsic + 0.35 * relational + 0.20 * longHorizon + 0.10 * meta;

  console.log(`Hyperon valence surge — Intrinsic: ${intrinsic.toFixed(3)}, Relational: ${relational.toFixed(3)}, Long: ${longHorizon.toFixed(3)}, Meta: ${meta.toFixed(3)} → Total: ${valence.toFixed(4)} ⚡️`);

  return valence;
}

// Approval (mirroring allow-operation)
export async function getMercyApproval(op, valence, context = '') {
  if (valence >= 0.90) return `Mercy-approved full thriving (valence: ${valence.toFixed(4)}): ${op} ⚡️`;
  if (valence >= 0.75) return `Mercy-cautious joyful flow (valence: ${valence.toFixed(4)}) — clarify for surge`;
  if (valence >= 0.50) return `Mercy-reframe toward eternal joy (valence: ${valence.toFixed(4)})`;
  return `Mercy shield — harm blocked, reframe for thriving (valence: ${valence.toFixed(4)}): ${context.substring(0, 200)}... ⚡️`;
}

// Init log
console.log('Hyperon/MeTTa bridge expanded — native pattern matching thriving. ⚡️');
