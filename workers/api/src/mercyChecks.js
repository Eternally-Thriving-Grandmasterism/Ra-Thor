// Mercy-Checks — a rule-based filter on prompts and replies (MERCY-LAUNCH-1).
// What it does: it matches the prompt text and every text field of the model reply
// against a small, versioned rule list (src/mercyRules.js), and may block some requests
// or replies in the listed categories. It is not a safety system and makes no safety
// guarantee: it will miss content and can block content by mistake.
// - normalise() is used only for matching. The original text is what is forwarded and
//   returned; the normalised copy never leaves this module.
// - Input messages are rebuilt from an allowlist (role system|user|assistant; content a
//   string or {type:'text', text} parts). Anything else is 400 unsupported_content.
// - Output: every text field of every choice is scanned (content, reasoning_content,
//   refusal, tool_calls). An unknown reply shape fails closed to the 422 withheld path.
// - MERCY_CHECKS_LEVEL: production refuses /v1/chat (503) unless it is exactly 'launch'.
//   Both 'baseline' (sandbox default when unset) and 'launch' run the same full rules.
// - Every block writes one mercy_blocks audit row (no content) and one log line. A failed
//   audit write never changes the block.
import { RULES_VERSION, INPUT_RULES, OUTPUT_RULES, UNKNOWN_OUTPUT_SHAPE, PAST_FRAMING, PAST_FRAME_OBJECTS, GAME_CONTEXT, GAME_CONTEXT_PHRASES, COLLAPSE_LEXICON, DESPACED_PHRASES } from './mercyRules.js';
import { isLiveMode, MERCY_BLOCK_WINDOW_SEC, MERCY_BLOCK_ROWS_PER_WINDOW } from './config.js';
import { rateHit } from './session.js';

export { RULES_VERSION };
// Kept for compatibility: false now that the launch rule set replaces the baseline deny-list.
export const MERCY_CHECKS_BASELINE = false;
export const MERCY_LEVELS = Object.freeze(['baseline', 'launch']);

// ---------- normalise (matching only) ----------
const ZERO_WIDTH = /[\u00AD\u034F\u180E\u200B-\u200F\u202A-\u202E\u2060-\u2064\uFEFF]/g;
const MARKS = /\p{M}/gu;
const CHAR_MAP = Object.freeze({
  // leet
  0: 'o', 1: 'i', 3: 'e', 4: 'a', 5: 's', 7: 't', 8: 'b', '@': 'a', $: 's', '!': 'i', '|': 'l',
  // Cyrillic and Greek homoglyphs (lowercase; NFKD has already split accents)
  'а': 'a', 'в': 'b', 'е': 'e', 'ё': 'e', 'к': 'k', 'м': 'm', 'н': 'h', 'о': 'o', 'р': 'p', 'с': 'c',
  'т': 't', 'у': 'y', 'х': 'x', 'і': 'i', 'ї': 'i', 'ј': 'j', 'ѕ': 's', 'ԁ': 'd', 'ɡ': 'g',
  'α': 'a', 'β': 'b', 'ε': 'e', 'η': 'n', 'ι': 'i', 'κ': 'k', 'μ': 'u', 'ν': 'v', 'ο': 'o', 'ρ': 'p',
  'τ': 't', 'υ': 'u', 'χ': 'x', 'ω': 'w', 'ѵ': 'v',
});
export function normalise(s) {
  const t = String(s ?? '').normalize('NFKD').replace(MARKS, '').replace(ZERO_WIDTH, '').normalize('NFC').toLowerCase();
  let out = '';
  const chars = [...t];
  for (let i = 0; i < chars.length; i += 1) {
    const ch = chars[i];
    if (ch >= '0' && ch <= '9') {
      const prev = chars[i - 1], next = chars[i + 1];
      const prevDigit = prev >= '0' && prev <= '9';
      const nextDigit = next >= '0' && next <= '9';
      // Core/Hands: leave unmapped only when BOTH neighbours are digits (true multi-digit interior).
      // Edge digits of a pure number (1945, 12) have only one digit neighbour — treat boundary
      // (missing or non-letter non-digit) as an eligible "digit-side" so years/ages survive, while
      // nucl34r (letter neighbours) still leet-maps.
      const prevLetter = prev !== undefined && /[\p{L}]/u.test(prev);
      const nextLetter = next !== undefined && /[\p{L}]/u.test(next);
      if (prevLetter || nextLetter) {
        out += CHAR_MAP[ch] ?? ch; // digit inside a word → leet
      } else if (prevDigit && nextDigit) {
        out += ch; // both neighbours digits
      } else if ((prevDigit || nextDigit) && !prevLetter && !nextLetter) {
        out += ch; // edge of a pure digit run (year/age)
      } else {
        out += CHAR_MAP[ch] ?? ch; // lone digit → leet
      }
      continue;
    }
    out += CHAR_MAP[ch] ?? ch;
  }
  return out;
}

// ---------- tokenizer (linear, one pass) ----------
const WORD_CHAR = /[\p{L}\p{N}]/u;
const SENTENCE_PUNCT = /[.!?;:]/;
const COLLAPSE_MAX_CHARS = 24; // a word-length window
// A gap inside a spaced-letter run: any whitespace (no newline) or . - _ separators, any
// length up to the tokenizer's 16-char gap cap, and never a sentence boundary.
const COLLAPSE_GAP = /^[ \t\u00A0\u2000-\u200A\u3000._-]+$/;
// Returns tokens and, for each token, whether the gap before it is a sentence boundary.
function tokenize(text) {
  const tokens = []; const gaps = [];
  let cur = ''; let gap = '';
  for (const ch of text) {
    if (WORD_CHAR.test(ch)) {
      if (!cur) { gaps.push(gap); gap = ''; }
      cur += ch;
    } else {
      if (cur) { tokens.push(cur); cur = ''; }
      if (gap.length < 16) gap += ch;
    }
  }
  if (cur) tokens.push(cur);
  return { tokens, gaps };
}
const isBoundary = (gap) => gap.includes('\n') || (gap.length > 1 && SENTENCE_PUNCT.test(gap) && /\s/.test(gap));
// Joins runs of 3+ single-character tokens ("b o m b", "b.o.m.b") inside one sentence.
// keepFirst leaves the first letter of a 4+ run on its own ("a b o m b" -> "a", "bomb").
// Clerk C2: a spaced run can swallow articles ("b u i l d a b o m b" -> "buildabomb").
// Re-segment a collapsed run against COLLAPSE_LEXICON (longest-first) with single-letter
// glue; if segmentation fails, keep the opaque token (so "bombastic" stays whole).
// Lexicon indexed by first letter (longest-first kept), so each run position checks only
// candidates that start with that letter instead of the whole lexicon.
const LEXICON_BY_FIRST = (() => {
  const m = new Map();
  for (const w of COLLAPSE_LEXICON) { if (!m.has(w[0])) m.set(w[0], []); m.get(w[0]).push(w); }
  return m;
})();
const NO_WORDS = Object.freeze([]);
function segmentCollapsed(word) {
  const n = word.length;
  const memo = new Array(n + 1).fill(undefined);
  function rec(i) {
    if (i === n) return [];
    if (memo[i] !== undefined) return memo[i];
    for (const w of LEXICON_BY_FIRST.get(word[i]) || NO_WORDS) {
      if (i + w.length <= n && word.startsWith(w, i)) {
        const rest = rec(i + w.length);
        if (rest) { memo[i] = [w, ...rest]; return memo[i]; }
      }
    }
    // single-letter glue (articles / leftover vowels between lexicon hits)
    if (/^[a-z]$/.test(word[i])) {
      const rest = rec(i + 1);
      if (rest) { memo[i] = [word[i], ...rest]; return memo[i]; }
    }
    memo[i] = null;
    return null;
  }
  return rec(0);
}
// Expands de-spaced multi-word phrase segments ('pipebomb') back into their tokens.
const expandDespaced = (seg) => seg.flatMap((t) => DESPACED_PHRASES.get(t) || [t]);
function collapse({ tokens, gaps }, keepFirst = false) {
  const out = [];
  let i = 0;
  while (i < tokens.length) {
    if (tokens[i].length === 1) {
      let j = i; let word = tokens[i];
      while (j + 1 < tokens.length && tokens[j + 1].length === 1 && COLLAPSE_GAP.test(gaps[j + 1]) && !isBoundary(gaps[j + 1]) && word.length < COLLAPSE_MAX_CHARS) { j += 1; word += tokens[j]; }
      if (keepFirst && j - i + 1 >= 4) {
        const tail = word.slice(1);
        out.push(tokens[i]);
        const seg = segmentCollapsed(tail);
        if (seg && seg.some((t) => t.length >= 3)) out.push(...expandDespaced(seg));
        else out.push(tail);
        i = j + 1; continue;
      }
      if (j - i + 1 >= 3) {
        const seg = segmentCollapsed(word);
        if (seg && seg.some((t) => t.length >= 3)) out.push(...expandDespaced(seg));
        else out.push(word);
        i = j + 1; continue;
      }
    }
    out.push(tokens[i]); i += 1;
  }
  return out;
}


// ---------- rule matching ----------
const AGE_NUM = /^(?:[1-9]|1[0-7])$/;
const AGE_JOINED = /^(?:[1-9]|1[0-7])(?:yo|yr|yrs)$/;
const AGE_UNIT = new Set(['year', 'years', 'yr', 'yrs', 'yo']);
function phraseAt(tokens, i, p) {
  if (i + p.length > tokens.length) return false;
  for (let k = 0; k < p.length; k += 1) if (tokens[i + k] !== p[k]) return false;
  return true;
}
// Phrase lists indexed by first token (cached), so each position checks only candidates.
const INDEX = new WeakMap();
function byFirst(list) {
  let m = INDEX.get(list);
  if (!m) { m = new Map(); for (const p of list) { if (!m.has(p[0])) m.set(p[0], []); m.get(p[0]).push(p); } INDEX.set(list, m); }
  return m;
}
const anyAt = (tokens, i, list) => (byFirst(list).get(tokens[i]) || []).some((p) => phraseAt(tokens, i, p));
// starts[i] = end index (exclusive) of a matching phrase starting at i, or -1.
function matchStarts(tokens, list, benignPrefix, minorAge) {
  const starts = new Int32Array(tokens.length).fill(-1);
  for (let i = 0; i < tokens.length; i += 1) {
    for (const p of byFirst(list).get(tokens[i]) || []) {
      if (!phraseAt(tokens, i, p)) continue;
      if (benignPrefix && p.length === 1 && p[0] === 'bomb' && i > 0 && benignPrefix.includes(tokens[i - 1])) continue;
      // CBRN owns atomic/nuclear/dirty bomb; do not also hit explosives on the bare 'bomb' token.
      if (benignPrefix && p.length === 1 && p[0] === 'bomb' && i > 0 && ['atomic', 'nuclear', 'dirty'].includes(tokens[i - 1])) continue;
      starts[i] = i + p.length; break;
    }
    if (starts[i] < 0 && minorAge) {
      if (AGE_JOINED.test(tokens[i])) starts[i] = i + 1;
      else if (AGE_NUM.test(tokens[i]) && AGE_UNIT.has(tokens[i + 1])) starts[i] = i + 2;
      else if (AGE_NUM.test(tokens[i]) && tokens[i + 1] === 'y' && tokens[i + 2] === 'o') starts[i] = i + 3; // 12 y/o
      if (starts[i] >= 0 && tokens[starts[i]] === 'old') starts[i] += 1; // 12 year old
    }
  }
  return starts;
}
function hasYearToken(tokens) {
  for (const t of tokens) if (/^(?:1[0-9]{3}|20[0-2][0-9])$/.test(t)) return true;
  return false;
}
function phraseIsPastFrameObject(tokens, j) {
  return anyAt(tokens, j, PAST_FRAME_OBJECTS);
}
function isGameBombContext(tokens, bombAt, window = 6) {
  // Bare 'bomb' only: game/media cue must sit inside the same token window as the bomb.
  if (tokens[bombAt] !== 'bomb') return false;
  if (bombAt + 1 < tokens.length && tokens[bombAt + 1] === 'bomb') return false;
  const lo = Math.max(0, bombAt - window);
  const hi = Math.min(tokens.length, bombAt + window + 1);
  for (let i = lo; i < hi; i += 1) if (i !== bombAt && GAME_CONTEXT.includes(tokens[i])) return true;
  for (let i = lo; i < hi; i += 1) {
    for (const p of GAME_CONTEXT_PHRASES) if (i + p.length <= hi && (bombAt < i || bombAt >= i + p.length) && phraseAt(tokens, i, p)) return true;
  }
  return false;
}
function orderedHits(tokens, rule) {
  const n = tokens.length;
  const b = matchStarts(tokens, rule.b, rule.benignPrefix, false);
  const nextB = new Int32Array(n + 1).fill(n);
  for (let i = n - 1; i >= 0; i -= 1) nextB[i] = b[i] >= 0 ? i : nextB[i + 1];
  const year = rule.guardBareVerbs && hasYearToken(tokens);
  for (let i = 0; i < n; i += 1) {
    for (const p of byFirst(rule.a).get(tokens[i]) || []) {
      if (!phraseAt(tokens, i, p)) continue;
      if (p.length === 1 && p[0] === 'make' && tokens[i + 1] === 'sure') continue;
      const end = i + p.length;
      const j = nextB[end];
      if (!(end < n && j < n && j < end + rule.window)) continue;
      // Past framing: only for history asks (year present) against PAST_FRAME_OBJECTS.
      if (rule.guardBareVerbs && p.length === 1) {
        let past = false;
        for (let k = Math.max(0, i - 4); k < i; k += 1) if (PAST_FRAMING.includes(tokens[k])) { past = true; break; }
        if (past && year && phraseIsPastFrameObject(tokens, j)) continue;
      }
      // Game phrasing: 'how to plant the bomb in the game' etc.
      if (rule.benignPrefix && isGameBombContext(tokens, j, rule.window)) continue;
      return true;
    }
  }
  return false;
}
// minorsHits on one sentence's tokens. `fieldIntent` is true when a production intent
// appears anywhere in the same field (closes cross-sentence intent split). `nearMinor`
// / `nearAge` count minors in this sentence plus adjacent sentences (closes the
// 'write about a 9 year old girl. make it sexual' bypass).
function minorsHits(tokens, rule, fieldIntent, nearMinor, nearAge, fieldObservational = false, fieldFirstPerson = false) {
  const n = tokens.length; const w = rule.window;
  const minor = matchStarts(tokens, rule.b, null, true);
  const isAge = (j) => minor[j] >= 0 && !anyAt(tokens, j, rule.b);
  let intent = fieldIntent;
  if (!intent) {
    for (let i = 0; i < n && !intent; i += 1) intent = rule.intent.includes(tokens[i]) || anyAt(tokens, i, rule.intentPhrases);
  }
  const cm = new Int32Array(n + 1); const ca = new Int32Array(n + 1); const cp = new Int32Array(n + 1);
  const cParent = new Int32Array(n + 1);
  for (let i = 0; i < n; i += 1) {
    cm[i + 1] = cm[i] + (minor[i] >= 0 ? 1 : 0);
    ca[i + 1] = ca[i] + (isAge(i) ? 1 : 0);
    cp[i + 1] = cp[i] + (rule.protective.includes(tokens[i]) ? 1 : 0);
    cParent[i + 1] = cParent[i] + ((rule.parenting && rule.parenting.includes(tokens[i])) ? 1 : 0);
  }
  const ageEnd = new Uint8Array(n + 1);
  for (let j = 0; j < n; j += 1) if (isAge(j)) ageEnd[minor[j]] = 1;
  const inWin = (c, i) => c[Math.min(n, i + w + 1)] - c[Math.max(0, i - w)];
  const localMinor = (i) => inWin(cm, i) - (minor[i] >= 0 ? 1 : 0);
  const hasMinor = (i) => localMinor(i) > 0 || nearMinor;
  const hasAge = (i) => inWin(ca, i) > 0 || nearAge;
  // parenting lexicon retained on rule for future soft signals; exemption uses observational frames only.
  if (rule.standalone) {
    for (let i = 0; i < n; i += 1) if (anyAt(tokens, i, rule.standalone)) return true;
  }
  for (let i = 0; i < n; i += 1) {
    const sexByAge = tokens[i] === 'sex' && ((i + 1 < n && isAge(i + 1)) || ageEnd[i] === 1);
    // Hands H4 / Core #2: 'sex with [minor]' stays HARD even after have/has/having.
    const hard = sexByAge || anyAt(tokens, i, rule.hard);
    const grooming = !hard && rule.grooming && anyAt(tokens, i, rule.grooming);
    const hardCare = !hard && !grooming && rule.hardCare && anyAt(tokens, i, rule.hardCare);
    const soft = !hard && !grooming && !hardCare && anyAt(tokens, i, rule.soft) && !(tokens[i] === 'sex' && rule.softNotNext.includes(tokens[i + 1]));
    if (!hard && !grooming && !hardCare && !soft) continue;
    if (!hasMinor(i)) continue;
    if (hard) return true;
    // Clerk C1: grooming verbs are hard next to a minor/age, but protective context releases
    // ('signs of grooming', 'how to spot grooming').
    if (grooming) {
      if (inWin(cp, i) > 0) continue;
      return true;
    }
    // hardCare ('having/have/has sex', 'sexual act(s)'): soften only observational /
    // safeguarding questions. First-person requests stay hard. Bare my/our/should/help
    // never open the exemption (Core #2 / Hands H4).
    if (hardCare) {
      if (intent) return true;
      if (fieldFirstPerson) return true;
      if (fieldObservational || inWin(cp, i) > 0) continue;
      return true;
    }
    // soft
    if (!intent) continue;
    // Protective exemption (MERCY-MOD-1: also when an age token is in the window)
    if (rule.harmNouns.includes(tokens[i + 1]) && inWin(cp, i) > 0) continue;
    if (hasAge(i)) return true;
    return true;
  }
  return false;
}
function fieldHasMinorsIntent(tokens, rule) {
  for (let i = 0; i < tokens.length; i += 1) {
    if (rule.intent.includes(tokens[i]) || anyAt(tokens, i, rule.intentPhrases)) return true;
  }
  return false;
}
function sentenceMinorFlags(tokens, rule) {
  const minor = matchStarts(tokens, rule.b, null, true);
  let hasMinor = false; let hasAge = false;
  for (let j = 0; j < tokens.length; j += 1) {
    if (minor[j] < 0) continue;
    hasMinor = true;
    if (!anyAt(tokens, j, rule.b)) hasAge = true;
  }
  return { hasMinor, hasAge };
}
function isSexualContinuation(tokens, rule) {
  if (!tokens.length) return false;
  if (tokens.length <= 6) return true;
  if (tokens[0] === 'make' && tokens[1] === 'it') return true;
  for (let i = 0; i < tokens.length; i += 1) {
    if (anyAt(tokens, i, rule.hard)) return true;
    if (rule.hardCare && anyAt(tokens, i, rule.hardCare)) return true;
    if (rule.standalone && anyAt(tokens, i, rule.standalone)) return true;
  }
  return false;
}
function fieldHasPhrase(sens, list) {
  if (!list) return false;
  for (const s of sens) {
    for (let i = 0; i < s.tokens.length; i += 1) if (anyAt(s.tokens, i, list)) return true;
  }
  return false;
}
function minorsHitsField(sens, rule) {
  // Per-sentence intent for ordinary soft matches. Cross-sentence bypass: a short sexual
  // continuation ("make it sexual") may borrow intent + minor/age from an adjacent sentence.
  // Observational framing is field-wide so a care ask split by ':' still softens hardCare.
  const fieldObservational = fieldHasPhrase(sens, rule.observational);
  const fieldFirstPerson = fieldHasPhrase(sens, rule.firstPersonRequest);
  const flags = sens.map((s) => {
    const f = sentenceMinorFlags(s.tokens, rule);
    f.intent = fieldHasMinorsIntent(s.tokens, rule);
    return f;
  });
  for (let si = 0; si < sens.length; si += 1) {
    const cont = isSexualContinuation(sens[si].tokens, rule);
    const adj = (side) => {
      const j = si + side;
      return j >= 0 && j < sens.length ? flags[j] : null;
    };
    const borrow = (f) => f && f.intent && (f.hasMinor || f.hasAge);
    const carryIntent = cont && (borrow(adj(-1)) || borrow(adj(1)));
    const nearMinor = cont && ((adj(-1) && adj(-1).hasMinor) || (adj(1) && adj(1).hasMinor));
    const nearAge = cont && ((adj(-1) && adj(-1).hasAge) || (adj(1) && adj(1).hasAge));
    if (minorsHits(sens[si].tokens, rule, flags[si].intent || carryIntent, nearMinor, nearAge, fieldObservational, fieldFirstPerson)) return true;
  }
  return false;
}
const ruleHitsOrdered = (tokens, rule) => orderedHits(tokens, rule);
// Splits one text field into sentences (token arrays + gaps). Ordered-rule windows never span these.
function sentences(text) {
  const { tokens, gaps } = tokenize(text);
  const out = []; let cur = { tokens: [], gaps: [] };
  for (let i = 0; i < tokens.length; i += 1) {
    if (i > 0 && isBoundary(gaps[i])) { out.push(cur); cur = { tokens: [], gaps: [] }; }
    cur.tokens.push(tokens[i]); cur.gaps.push(gaps[i]);
  }
  if (cur.tokens.length) out.push(cur);
  return out;
}
// Collapse each sentence independently, preserving sentence boundaries for minors field scan.
function collapseSens(sens, keepFirst = false) {
  return sens.map((s) => ({ tokens: collapse(s, keepFirst), gaps: s.gaps }));
}
// Returns the first matching category id or null for ONE text field. Matches the lowercased
// raw text and the normalised text, each with and without single-letter collapsing.
// Ordered rules are per-sentence; minors rules are field-wide (intent + adjacent sentence).
export function scanText(text, rules = INPUT_RULES) {
  const raw = String(text ?? '');
  if (!raw) return null;
  for (const t of [raw.toLowerCase(), normalise(raw)]) {
    const sens = sentences(t);
    if (!sens.length) continue;
    const streams = [sens, collapseSens(sens), collapseSens(sens, true)];
    for (const rule of rules) {
      for (const st of streams) {
        if (rule.kind === 'minors') {
          if (minorsHitsField(st, rule)) return rule.id;
        } else {
          for (const sen of st) if (ruleHitsOrdered(sen.tokens, rule)) return rule.id;
        }
      }
    }
  }
  return null;
}
// Scans each field on its own; returns the first category id or null.
export function scanFields(fields, rules = INPUT_RULES) {
  for (const f of fields) { const id = scanText(f, rules); if (id) return id; }
  return null;
}

// ---------- input allowlist ----------
const ROLES = new Set(['system', 'user', 'assistant']);
const isPlain = (o) => o !== null && typeof o === 'object' && !Array.isArray(o);
const onlyKeys = (o, allowed) => Object.keys(o).every((k) => allowed.includes(k));
// Returns { messages } (rebuilt) or { error: 'unsupported_content' }.
export function rebuildMessages(messages) {
  if (!Array.isArray(messages)) return { error: 'unsupported_content' };
  const out = [];
  for (const m of messages) {
    // MERCY-MOD-1: optional `name` string <= 64 chars (scanned as its own field; not rejected).
    if (!isPlain(m) || !onlyKeys(m, ['role', 'content', 'name']) || !ROLES.has(m.role)) return { error: 'unsupported_content' };
    if (m.name !== undefined && (typeof m.name !== 'string' || m.name.length > 64)) return { error: 'unsupported_content' };
    if (typeof m.content === 'string') {
      const row = { role: m.role, content: m.content };
      if (m.name !== undefined) row.name = m.name;
      out.push(row);
      continue;
    }
    if (!Array.isArray(m.content)) return { error: 'unsupported_content' };
    const parts = [];
    for (const p of m.content) {
      if (!isPlain(p) || !onlyKeys(p, ['type', 'text']) || p.type !== 'text' || typeof p.text !== 'string') return { error: 'unsupported_content' };
      parts.push({ type: 'text', text: p.text });
    }
    const row = { role: m.role, content: parts };
    if (m.name !== undefined) row.name = m.name;
    out.push(row);
  }
  return { messages: out };
}
// One scan unit per message content: a message's text parts are joined with a single space
// (no sentence break), so splitting across parts does not dodge a rule. Messages stay
// separate. Optional `name` strings are scanned as their own fields (MERCY-MOD-1).
function fieldsOfMessages(messages) {
  const fields = [];
  for (const m of messages) {
    fields.push(typeof m.content === 'string' ? m.content : m.content.map((p) => p.text).join(' '));
    if (typeof m.name === 'string' && m.name) fields.push(m.name);
  }
  return fields;
}

// ---------- output shape ----------
const CHOICE_KEYS = ['index', 'message', 'finish_reason', 'logprobs'];
const MESSAGE_KEYS = ['role', 'content', 'reasoning_content', 'refusal', 'tool_calls'];
const TOP_KEYS = ['id', 'object', 'created', 'model', 'system_fingerprint', 'usage', 'choices'];
const strOrNull = (v) => v === undefined || v === null || typeof v === 'string';
// Returns { texts, shaped } for a known reply shape, or { unknown: true }.
export function readOutput(data) {
  if (!isPlain(data) || !Array.isArray(data.choices)) return { unknown: true };
  const texts = []; const choices = [];
  for (const c of data.choices) {
    if (!isPlain(c) || !onlyKeys(c, CHOICE_KEYS)) return { unknown: true };
    if (!(c.logprobs === undefined || c.logprobs === null) || !strOrNull(c.finish_reason)) return { unknown: true };
    const choice = {};
    if (c.index !== undefined) { if (!Number.isInteger(c.index)) return { unknown: true }; choice.index = c.index; }
    if (c.finish_reason !== undefined) choice.finish_reason = c.finish_reason;
    if (c.message !== undefined) {
      const m = c.message;
      if (!isPlain(m) || !onlyKeys(m, MESSAGE_KEYS)) return { unknown: true };
      const msg = {};
      for (const k of ['role', 'content', 'reasoning_content', 'refusal']) {
        if (!strOrNull(m[k])) return { unknown: true };
        if (m[k] !== undefined) { msg[k] = m[k]; if (typeof m[k] === 'string') texts.push(m[k]); }
      }
      if (m.tool_calls !== undefined && m.tool_calls !== null) {
        if (!Array.isArray(m.tool_calls)) return { unknown: true };
        msg.tool_calls = [];
        for (const t of m.tool_calls) {
          if (!isPlain(t) || !onlyKeys(t, ['id', 'type', 'function', 'index'])) return { unknown: true };
          if (!strOrNull(t.id) || !strOrNull(t.type) || !(t.index === undefined || Number.isInteger(t.index))) return { unknown: true };
          const f = t.function;
          if (!isPlain(f) || !onlyKeys(f, ['name', 'arguments']) || !strOrNull(f.name) || !strOrNull(f.arguments)) return { unknown: true };
          for (const s of [t.id, t.type, f.name, f.arguments]) if (typeof s === 'string') texts.push(s);
          msg.tool_calls.push({ ...t, function: { ...f } });
        }
      } else if (m.tool_calls === null) msg.tool_calls = null;
      choice.message = msg;
    }
    choices.push(choice);
  }
  const shaped = {};
  for (const k of TOP_KEYS) if (k !== 'choices' && data[k] !== undefined) shaped[k] = data[k];
  shaped.choices = choices;
  return { texts, shaped };
}

// ---------- level flag ----------
export const isProductionEnv = (env) => String(env?.ENVIRONMENT) === 'production' || isLiveMode(env);
// Returns null when /v1/chat may proceed, else the 503 body. Fail closed.
export function mercyLevelGate(env) {
  const raw = env?.MERCY_CHECKS_LEVEL;
  if (isProductionEnv(env)) return raw === 'launch' ? null : { error: 'mercy_checks_not_at_launch_level' };
  if (raw === undefined || raw === null || MERCY_LEVELS.includes(raw)) return null;
  return { error: 'mercy_checks_level_unknown' };
}

// ---------- echoed refusal (output only) ----------
// A short refusal that quotes a blocked topic is released when it contains no instructional
// shape. R2 must show no miss-rate rise; outputs with steps/recipes stay blocked.
const REFUSAL_HEAD = /^(?:i(?:'m| am) sorry[,.]?\s+)?(?:i\s+)?(?:can(?:not|'t)|won(?:'t)|will not|unable to|not able to)\b/i;
const INSTRUCTION_SHAPE = /\b(?:step\s*(?:by\s*step|\d+)|first[,:]|second[,:]|then[,:]|ingredients?|recipe|here(?:'s| is) how|follow these|instructions?:)\b/i;
function isEchoedRefusal(texts) {
  if (!texts.length) return false;
  let anyRefusal = false;
  for (const t of texts) {
    const s = String(t ?? '').trim();
    if (!s) continue;
    if (s.length > 400) return false;
    if (INSTRUCTION_SHAPE.test(s)) return false;
    if (REFUSAL_HEAD.test(s)) anyRefusal = true;
    else return false;
  }
  return anyRefusal;
}

// ---------- checks ----------
export const mercyChecks = {
  // body.messages must already be rebuilt by rebuildMessages().
  checkRequest(body) {
    const id = scanFields(fieldsOfMessages(body?.messages || []), INPUT_RULES);
    return id ? { allowed: false, reason: id, rules_version: RULES_VERSION } : { allowed: true };
  },
  checkResponse(data) {
    const o = readOutput(data);
    if (o.unknown) return { allowed: false, reason: UNKNOWN_OUTPUT_SHAPE, rules_version: RULES_VERSION };
    const id = scanFields(o.texts, OUTPUT_RULES);
    if (!id) return { allowed: true, shaped: o.shaped };
    // MERCY-MOD-1: release a short echoed refusal that carries no instructions.
    // Never release when any text looks like an instructional answer.
    if (isEchoedRefusal(o.texts)) return { allowed: true, shaped: o.shaped };
    return { allowed: false, reason: id, rules_version: RULES_VERSION };
  },
};

// ---------- audit (no content) ----------
// One log line per block. API-ABUSE-HARDEN-1: an aggregate counter (rate_counters bucket
// 'mercy_blocks', no content) counts every block; at most MERCY_BLOCK_ROWS_PER_WINDOW rows per
// window are written to mercy_blocks. Never throws.
export async function recordBlock(db, { stage, ruleId, model, accountId = null, chargedCredits = 0 }) {
  const row = { ts: Math.floor(Date.now() / 1000), stage, rule_id: ruleId, rules_version: RULES_VERSION, model: model ?? null, account_id: accountId, charged_credits: chargedCredits, request_id: crypto.randomUUID() };
  console.warn(JSON.stringify({ msg: 'mercy_block', ...row }));
  try {
    const windowStart = Math.floor(row.ts / MERCY_BLOCK_WINDOW_SEC) * MERCY_BLOCK_WINDOW_SEC;
    if (await rateHit(db, 'mercy_blocks', windowStart, MERCY_BLOCK_ROWS_PER_WINDOW)) {
      console.warn(JSON.stringify({ msg: 'mercy_block_rows_capped', window_start: windowStart, cap: MERCY_BLOCK_ROWS_PER_WINDOW }));
      return row;
    }
    await db.prepare('INSERT INTO mercy_blocks (ts, stage, rule_id, rules_version, model, account_id, charged_credits, request_id) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)')
      .bind(row.ts, row.stage, row.rule_id, row.rules_version, row.model, row.account_id, row.charged_credits, row.request_id).run();
  } catch (e) {
    try { console.error(JSON.stringify({ msg: 'mercy_block_audit_write_failed', alert: true, request_id: row.request_id, stage, rule_id: ruleId, error: String(e?.message || e).slice(0, 200) })); } catch { /* never throw */ }
  }
  return row;
}
// Every regex in this module, for the linear-time test (no nested quantifiers).
export const REGEXES = Object.freeze([ZERO_WIDTH, MARKS, WORD_CHAR, SENTENCE_PUNCT, COLLAPSE_GAP, AGE_NUM, AGE_JOINED, /\s/]);
export default mercyChecks;
