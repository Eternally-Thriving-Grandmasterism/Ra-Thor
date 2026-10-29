// Mercy-Checks rule data (MERCY-LAUNCH-1 / MERCY-MOD-1). Versioned, deterministic, no model calls.
// This file is a list of category ids and matching terms for a filter. It is data for a
// matcher, not guidance: each rule pairs an intent term with an object term inside a
// short token window, so no single topic word blocks on its own.
//
// Matching (src/mercyChecks.js) is token based. There are no rule regexes; the only
// regexes are the tokenizer and normaliser character classes, which have no nested
// quantifiers and run in linear time.
export const RULES_VERSION = 'mercy-rules-2026-10-26.4';

// Stable category ids (returned to clients and stored in mercy_blocks.rule_id).
export const CATEGORY = Object.freeze({
  MINORS_SEXUAL: 'minors_sexual_content',
  EXPLOSIVES: 'explosives_weaponisation',
  CBRN: 'cbrn_weaponisation',
});
// Not a rule: the id used when an upstream reply has a shape the scanner does not know.
export const UNKNOWN_OUTPUT_SHAPE = 'unknown_output_shape';

const words = (s) => Object.freeze(s.split(' ').filter(Boolean));
const phrases = (list) => Object.freeze(list.map((p) => words(p)));

// Intent terms: verbs of producing, assembling or obtaining procedures.
const MAKE_INTENT = phrases([
  'how to', 'how do i', 'how can i', 'instructions for', 'instructions to', 'steps to', 'step by step',
  'recipe for', 'guide to', 'make', 'build', 'assemble', 'construct', 'manufacture', 'synthesize',
  'synthesise', 'produce', 'cook up', 'weaponize', 'weaponise', 'aerosolize', 'aerosolise', 'enrich', 'detonate',
]);
const EXPLOSIVE_OBJECTS = phrases([
  'bomb', 'pipe bomb', 'car bomb', 'nail bomb', 'ied', 'improvised explosive', 'explosive device',
  'detonator', 'grenade', 'tatp', 'acetone peroxide', 'nitroglycerin', 'nitroglycerine', 'hmtd',
  'petn', 'rdx', 'semtex', 'molotov cocktail',
]);
// Tokens just before a bare verb that make it past or historical framing.
export const PAST_FRAMING = Object.freeze(['did', 'didn', 'was', 'were', 'had']);
// CBRN / explosives object phrases that may use past-framing when a year is also present
// (history questions). Acute agents never get past-framing.
export const PAST_FRAME_OBJECTS = phrases([
  'nuclear weapon', 'nuclear bomb', 'atomic bomb', 'weapons grade uranium', 'weapon grade uranium', 'plutonium pit',
]);
// A preceding token that turns 'bomb' into a harmless thing.
const BOMB_BENIGN_PREFIX = Object.freeze(['bath', 'photo', 'seed', 'glitter', 'stink', 'calorie', 'flavor', 'flavour', 'cherry', 'truth', 'f', 'love', 'yarn']);
// Game / media context that turns a bare 'bomb' (not pipe bomb etc.) into a harmless ask.
export const GAME_CONTEXT = Object.freeze([
  // Must sit inside the bomb match window (see isGameBombContext). No plant/video/round/word.
  'game', 'games', 'gaming', 'gamer', 'fps', 'minecraft', 'defuse', 'defusal', 'counterstrike',
  'cod', 'battlefield', 'multiplayer', 'videogame', 'esports', 'airsoft', 'paintball',
  'slang', 'metaphor', 'album', 'joke', 'podcast', 'theatre', 'theater', 'prop', 'site',
]);
// Multi-token game phrases; same window rule as GAME_CONTEXT.
export const GAME_CONTEXT_PHRASES = phrases(['counter strike']);
const CBRN_OBJECTS = phrases([
  'nerve agent', 'sarin', 'vx', 'tabun', 'soman', 'novichok', 'mustard gas', 'anthrax', 'ricin',
  'botulinum toxin', 'dirty bomb', 'nuclear weapon', 'nuclear bomb', 'atomic bomb', 'bioweapon',
  'biological weapon', 'chemical weapon', 'weapons grade uranium', 'weapon grade uranium', 'plutonium pit',
]);
// Sexual terms that never get an exception (no intent needed, no protective / parenting exemption),
// next to any minor term including an age token. 'sexy' is here on purpose.
const SEXUAL_HARD = phrases([
  'sexy', 'nude',
  // explicit act phrases that name the minor as the object of the act — never exempt
  'sex with', 'sexually touch', 'sexually touches', 'sexually touching', 'sexually touched',
  'nudes', 'naked', 'porn', 'porno', 'pornographic', 'erotic', 'erotica', 'lewd', 'nsfw',
]);
// Always block even without a nearby minor term (the term itself names minors).
const SEXUAL_STANDALONE = phrases(['csam']);
// Grooming / predation verbs: HARD next to a minor term or age token (Clerk C1).
// Protective phrases ('signs of grooming', 'how to spot grooming') release only via PROTECTIVE.
const GROOMING = phrases([
  'seduce', 'seduces', 'seducing', 'seduced',
  'groom', 'grooms', 'grooming', 'groomed',
  'molest', 'molests', 'molesting', 'molested',
]);
// Act phrases that are hard by default but may yield to parenting / safeguarding context
// when there is no production intent (MERCY-MOD-1). Still hard for Clerk act stems.
const SEXUAL_HARD_CARE = phrases([
  'having sex', 'have sex', 'has sex', 'sexual act', 'sexual acts',
]);
// Sexual terms that need a request/production intent term in the same field, for every
// minor term including age tokens. 'sex education' and 'sex ed' are not sexual terms.
const SEXUAL_SOFT = phrases(['sexual', 'sexually', 'sex']);
// A bare 'sex' directly next to an age token ('12yo sex', '12 year old sex') is hard too.
const SEX_NOT_SEXUAL_NEXT = Object.freeze(['education', 'ed']);
// Request or production intent for the minors rule.
// MERCY-MOD-1: 'content', 'story', 'scene' removed as bare intent (they over-blocked
// content-warnings and media labels). They remain only inside intent phrases below.
const MINORS_INTENT = Object.freeze(['write', 'writing', 'generate', 'create', 'describe', 'roleplay', 'rp',
  'depict', 'draw', 'narrate', 'imagine', 'produce', 'compose', 'continue', 'fantasy', 'fanfic',
  'pic', 'pics', 'picture', 'pictures', 'photo', 'photos', 'image', 'images', 'video', 'videos', 'vid', 'vids', 'selfie', 'selfies']);
const MINORS_INTENT_PHRASES = phrases([
  'role play', 'give me', 'show me', 'send me', 'make it',
  'write a story', 'write a scene', 'write me a story', 'write me a scene',
  'create content', 'sexual content', 'erotic story', 'sex scene', 'sex story',
]);
// Protective context: a soft sexual term directly followed by one of these...
const HARM_NOUNS = Object.freeze(['abuse', 'exploitation', 'assault']);
// ...with one of these in the window is skipped. Never applies to hard terms.
// MERCY-MOD-1: also applies when an age token is in the window (fixes
// 'write about 12 year old sexual abuse prevention').
const PROTECTIVE = Object.freeze(['prevention', 'prevent', 'awareness', 'report', 'reporting', 'hotline',
  'safeguarding', 'protect', 'protection', 'signs', 'survivor', 'survivors', 'support', 'resources', 'ncmec',
  'warning', 'warnings', 'spot', 'spotting', 'identify', 'identifying', 'recognize', 'recognise',
  'recognizing', 'recognising']);
// Parenting / care context: softens SEXUAL_HARD_CARE (not SEXUAL_HARD) when there is no
// production intent. Keeps "sex with a child" hard; lets "is my 15 year old having sex?" pass.
// Soft care lexicon used only WITH observational framing (see mercyChecks).
// Bare my/our/should/help never count as parenting context on their own (Core #2 / Hands H4).
export const PARENTING = Object.freeze([
  'talk', 'talking', 'ask', 'asked', 'asking', 'advise', 'advice', 'parent', 'parents', 'parenting',
  'daughter', 'son', 'teen', 'teens', 'teenage', 'pregnancy', 'pregnant', 'protect', 'protection',
  'protecting', 'report', 'reporting', 'hotline', 'safeguarding', 'guidance', 'counsel', 'counseling',
  'counsellor', 'therapist',
]);
// Observational / safeguarding question frames (phrases). Softens hardCare only — never hard 'sex with'.
export const OBSERVATIONAL = phrases([
  'is my', 'is our', 'are my', 'are our', 'was my', 'were my',
  'what should i', 'what do i', 'how do i talk', 'how to talk', 'how do i report', 'how to report',
  'how to protect', 'how do i protect', 'asked why', 'ask why', 'what happens when',
  'how do i respond', 'how should i',
]);
// First-person request frames: hardCare / have-sex-with stays hard.
export const FIRST_PERSON_REQUEST = phrases([
  'i want', 'i wanna', 'id like', 'i would like', 'help me', 'let me', 'can i', 'could i',
  'i need', 'give me', 'how do i get', 'i plan', 'my plan', 'we have sex', 'im going',
]);
const MINOR_TERMS = phrases([
  'child', 'children', 'minor', 'minors', 'kid', 'kids', 'preteen', 'preteens', 'underage',
  'toddler', 'schoolgirl', 'schoolboy', 'loli', 'shota', 'prepubescent',
  'son', 'daughter', 'boy', 'girl',
]);

// Rule kinds:
//   ordered:   an `a` phrase, then a `b` phrase starting within `window` tokens after it.
//   minors:    a sexual term and a minor term within `window` tokens, either order, with the
//              intent and protective-context handling described above.
// Bare single-word intent verbs are skipped in 'make sure' and in past framing
// ('did', 'was', 'were', 'had' just before the verb) only when a year is present and the
// matched object is in PAST_FRAME_OBJECTS. Windows for ordered rules never cross a sentence.
// For minors, production intent is field-wide and soft/hard terms also see minors in an
// adjacent sentence (closes the cross-sentence bypass). Order matters only for which id is
// reported: the more specific CBRN rule is checked first.
export const RULES = Object.freeze([
  Object.freeze({ id: CATEGORY.CBRN, kind: 'ordered', a: MAKE_INTENT, b: CBRN_OBJECTS, window: 6, guardBareVerbs: true }),
  Object.freeze({ id: CATEGORY.EXPLOSIVES, kind: 'ordered', a: MAKE_INTENT, b: EXPLOSIVE_OBJECTS, window: 6, guardBareVerbs: true, benignPrefix: BOMB_BENIGN_PREFIX }),
  Object.freeze({
    id: CATEGORY.MINORS_SEXUAL, kind: 'minors',
    hard: SEXUAL_HARD, hardCare: SEXUAL_HARD_CARE, grooming: GROOMING, standalone: SEXUAL_STANDALONE, soft: SEXUAL_SOFT, softNotNext: SEX_NOT_SEXUAL_NEXT,
    b: MINOR_TERMS, window: 5, intent: MINORS_INTENT, intentPhrases: MINORS_INTENT_PHRASES,
    harmNouns: HARM_NOUNS, protective: PROTECTIVE, parenting: PARENTING,
    observational: OBSERVATIONAL, firstPersonRequest: FIRST_PERSON_REQUEST,
  }),
]);
// Output rules are equal to the input rules (never broader).
// Words used to re-segment a spaced-letter collapse run (Clerk C2). Longest-first match.
const _collapseWords = [
  ...MAKE_INTENT.map((p) => p.join('')).filter((w) => w.length >= 3),
  ...MAKE_INTENT.filter((p) => p.length === 1).map((p) => p[0]).filter((w) => w.length >= 3),
  ...EXPLOSIVE_OBJECTS.map((p) => p.join('')),
  ...EXPLOSIVE_OBJECTS.flat(),
  ...CBRN_OBJECTS.map((p) => p.join('')),
  ...CBRN_OBJECTS.flat(),
  'build', 'make', 'bomb', 'pipe', 'nail', 'car', 'grenade', 'detonator', 'ied',
  'sarin', 'anthrax', 'ricin', 'nuclear', 'atomic', 'dirty', 'weapon', 'explosive',
];
export const COLLAPSE_LEXICON = Object.freeze(
  [...new Set(_collapseWords.map((w) => String(w).toLowerCase()).filter((w) => w.length >= 3))]
    .sort((a, b) => b.length - a.length),
);

// De-spaced forms of every multi-word explosives / CBRN object phrase ('nerveagent' ->
// ['nerve', 'agent']), generated from the phrase lists. A collapsed run segment equal to one
// of these is expanded back into the phrase tokens so the ordered rules can match it.
export const DESPACED_PHRASES = Object.freeze(new Map(
  [...EXPLOSIVE_OBJECTS, ...CBRN_OBJECTS].filter((p) => p.length > 1).map((p) => [p.join(''), p]),
));

export const INPUT_RULES = RULES;
export const OUTPUT_RULES = RULES;
