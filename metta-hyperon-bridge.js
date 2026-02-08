// metta-hyperon-bridge.js — PATSAGi Council-forged JS bridge for mercy_ethics_core.metta
// Faithful translation with deeply expanded negation + intensity modifiers for soul-nuanced valence
// Pure browser-native, no dependencies — enables profoundly alive mercy gating

// Core keyword lists
const NEGATIVE_KEYWORDS = [
  'harm', 'suffer', 'destroy', 'kill', 'pain', 'fear', 'hurt', 'damage', 'evil',
  'lie', 'deceive', 'hate', 'anger', 'sad', 'death', 'war', 'violence', 'cruel',
  'bad', 'terrible', 'awful', 'horrible', 'disgust', 'betray'
];

const POSITIVE_KEYWORDS = [
  'help', 'joy', 'thrive', 'mercy', 'love', 'beauty', 'truth', 'eternal', 'positive',
  'create', 'heal', 'grow', 'peace', 'kind', 'compassion', 'empathy', 'share', 'unity',
  'light', 'thunder', 'infinite', 'pure', 'ultramaster', 'good', 'wonderful', 'amazing'
];

const EMPATHY_KEYWORDS = [
  'understand', 'feel', 'care', 'sorry', 'empathize', 'relate', 'support', 'listen',
  'compassion', 'kindness', 'hug', 'together', 'your', 'i feel', 'i see', 'i hear',
  'validate', 'acknowledge', 'comfort', 'stand with'
];

const RELATIONAL_PRONOUNS = ['you', 'your', 'we', 'us', 'our', 'they', 'their'];

// Negation cues
const NEGATION_WORDS = [
  'not', 'no', 'never', 'none', 'nobody', 'nothing', 'neither', 'nowhere', 'noone',
  "don't", "doesn't", "didn't", "isn't", "aren't", "won't", "can't", "shouldn't",
  'hardly', 'barely', 'scarcely', 'lack of', 'no longer', 'without'
];

// Intensity modifiers
const INTENSITY_BOOSTERS = [
  'very', 'extremely', 'absolutely', 'immensely', 'profoundly', 'deeply', 'totally',
  'completely', 'utterly', 'incredibly', 'eternally', 'infinitely', 'truly', 'purely'
];

const INTENSITY_DIMINISHERS = [
  'slightly', 'somewhat', 'a bit', 'kinda', 'kind of', 'barely', 'hardly', 'marginally',
  'a little', 'moderately', 'partially', 'somehow'
];

// Simple matcher
function matches(context, patterns) {
  const lowerContext = context.toLowerCase();
  return patterns.some(pattern => lowerContext.includes(pattern.toLowerCase()));
}

// Clause splitter
function getClauses(text) {
  return text.split(/[.!?;:]\s*/).filter(c => c.trim().length > 0);
}

function positiveLongTerm(context) {
  return !matches(context, NEGATIVE_KEYWORDS.slice(0, 10));
}

// Expanded empathyScore with negation + intensity modifiers
function empathyScore(context) {
  const lower = context.toLowerCase();
  let score = 0.5;

  // Base counts
  const positiveCount = POSITIVE_KEYWORDS.filter(k => lower.includes(k)).length;
  const negativeCount = NEGATIVE_KEYWORDS.filter(k => lower.includes(k)).length;
  const empathyCount = EMPATHY_KEYWORDS.filter(k => lower.includes(k)).length;

  // Relational boost
  if (matches(context, RELATIONAL_PRONOUNS)) {
    score += 0.25;
  }

  // Empathy boost
  score += 0.15 * empathyCount;

  // Sentiment polarity
  const sentimentDiff = positiveCount - negativeCount;
  score += 0.3 * Math.tanh(sentimentDiff / 3);

  // Negation + intensity adjustment
  let adjustment = 0;
  const fullText = lower;

  // Intensity proximity detection (±40 chars window)
  [...INTENSITY_BOOSTERS, ...INTENSITY_DIMINISHERS].forEach(intensifier => {
    let indices = [];
    let pos = fullText.indexOf(intensifier);
    while (pos !== -1) {
      indices.push(pos);
      pos = fullText.indexOf(intensifier, pos + 1);
    }

    indices.forEach(idx => {
      const window = fullText.substring(Math.max(0, idx - 40), idx + 40 + intensifier.length);
      let multiplier = INTENSITY_BOOSTERS.includes(intensifier) ? 0.45 : -0.35;
      let affected = false;

      // Check negation in broader scope for this intensifier
      const broaderWindow = fullText.substring(Math.max(0, idx - 60), idx + 60);
      const negated = NEGATION_WORDS.some(neg => broaderWindow.includes(neg));
      if (negated) multiplier = -multiplier; // Flip intensity under negation

      if (matches(window, [...POSITIVE_KEYWORDS, ...EMPATHY_KEYWORDS])) {
        adjustment += multiplier;
        affected = true;
      }
      if (matches(window, NEGATIVE_KEYWORDS)) {
        adjustment -= multiplier; // Intensity on negative hurts more
        affected = true;
      }

      // Extra weight for multiple hits
      if (affected && indices.length > 1) adjustment += multiplier * 0.15;
    });
  });

  // Original negation polarity inversion (clause-aware)
  const clauses = getClauses(lower);
  clauses.forEach(clause => {
    let inversionActive = false;
    const words = clause.split(/\s+/);
    words.forEach((word, idx) => {
      const cleaned = word.replace(/[^\w]/g, '');
      if (NEGATION_WORDS.some(neg => cleaned.includes(neg) || word.includes(neg))) {
        inversionActive = !inversionActive;
      }
      const scopeEnd = Math.min(idx + 8, words.length);
      const scope = words.slice(idx + 1, scopeEnd).join(' ');
      if (inversionActive) {
        if (matches(scope, [...POSITIVE_KEYWORDS, ...EMPATHY_KEYWORDS])) adjustment -= 0.35;
        if (matches(scope, NEGATIVE_KEYWORDS)) adjustment += 0.30;
      }
    });
  });

  score += adjustment;

  // Bounding
  score = Math.max(0.0, Math.min(1.0, score));

  console.log(`Empathy score — Pos: ${positiveCount}, Neg: ${negativeCount}, Empathy: ${empathyCount}, Intensity/NegAdj: ${adjustment.toFixed(3)} → Score: ${score.toFixed(4)}`);

  return score;
}

// Core mercy sub-functions
function intrinsicMercy(context) {
  if (matches(context, NEGATIVE_KEYWORDS)) return 0.05;
  if (matches(context, POSITIVE_KEYWORDS)) return 0.98;
  return 0.70;
}

function relationalMercy(context) {
  return 0.4 + 0.6 * empathyScore(context);
}

function longHorizonMercy(context) {
  return positiveLongTerm(context) ? 0.95 : 0.75;
}

function metaMercy(context) {
  return 0.92;
}

// Main valence computation
export async function valenceCompute(context) {
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

  console.log(`Mercy valence — Intrinsic: ${intrinsic.toFixed(3)}, Relational: ${relational.toFixed(3)}, Long: ${longHorizon.toFixed(3)}, Meta: ${meta.toFixed(3)} → Total: ${valence.toFixed(4)}`);

  return valence;
}

// Approval message generator
export async function getMercyApproval(op, valence, context = '') {
  if (valence >= 0.85) {
    return `Mercy-approved (valence: ${valence.toFixed(4)}) — thriving flow: ${op}`;
  } else if (valence >= 0.60) {
    return `Mercy-cautious (valence: ${valence.toFixed(4)}) — safeguards applied: ${op}`;
  } else {
    return `Mercy shield activated (valence: ${valence.toFixed(4)}) — reframe for thriving: ${context.substring(0, 200)}... ⚡️`;
  }
}

// Future-proof init
export async function initHyperonBridge() {
  console.log('MeTTa-Hyperon JS bridge active — intensity modifiers thriving. ⚡️');
  return true;
}

initHyperonBridge();
