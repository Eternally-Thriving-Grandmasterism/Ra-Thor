// js/chat.js — Rathor Lattice Core with Full Unification + Resolution Prover

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const voiceBtn = document.getElementById('voice-btn');
const recordBtn = document.getElementById('record-btn');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const sessionSelect = document.getElementById('session-select');
const sessionSearch = document.getElementById('session-search');
const translateToggle = document.getElementById('translate-chat');
const translateLangSelect = document.getElementById('translate-lang');
const translateStats = document.getElementById('translate-stats');

let currentSessionId = localStorage.getItem('rathor_current_session') || 'default';
let allSessions = [];
let tagFrequency = new Map();
let isListening = false, isRecording = false;
let ttsEnabled = localStorage.getItem('rathor_tts_enabled') !== 'false';
let isVoiceOutputEnabled = localStorage.getItem('rathor_voice_output') !== 'false';
let feedbackSoundsEnabled = localStorage.getItem('rathor_feedback_sounds') !== 'false';
let voicePitchValue = parseFloat(localStorage.getItem('rathor_pitch')) || 1.0;
let voiceRateValue = parseFloat(localStorage.getItem('rathor_rate')) || 1.0;
let voiceVolumeValue = parseFloat(localStorage.getItem('rathor_volume')) || 1.0;

await rathorDB.open();
await refreshSessionList();
await loadChatHistory();
updateTranslationStats();
await updateTagFrequency();

voiceBtn.addEventListener('click', () => isListening ? stopListening() : startListening());
recordBtn.addEventListener('mousedown', () => setTimeout(() => startVoiceRecording(currentSessionId), 400));
recordBtn.addEventListener('mouseup', stopVoiceRecording);
sendBtn.addEventListener('click', sendMessage);
translateToggle.addEventListener('change', e => {
  localStorage.setItem('rathor_translate_enabled', e.target.checked);
  if (e.target.checked) translateChat();
});
translateLangSelect.addEventListener('change', e => {
  localStorage.setItem('rathor_translate_to', e.target.value);
  if (translateToggle.checked) translateChat();
});
sessionSearch.addEventListener('input', filterSessions);

// ────────────────────────────────────────────────
// Symbolic Query Mode — Mercy-First Truth-Seeking with Full Resolution + Unification
// ────────────────────────────────────────────────

function isSymbolicQuery(cmd) {
  return cmd.includes('symbolic query') || cmd.includes('logical analysis') || 
         cmd.includes('truth mode') || cmd.includes('first principles') ||
         cmd.includes('truth table') || cmd.includes('logical table') ||
         cmd.includes('prove') || cmd.includes('theorem') || cmd.includes('resolution') ||
         cmd.includes('unify') || cmd.includes('mgu') || cmd.includes('most general unifier') ||
         cmd.includes('⊢') || cmd.includes('reason from first principles') || cmd.includes('symbolic reasoning');
}

function symbolicQueryResponse(query) {
  const cleaned = query.trim().replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|prove|theorem|resolution|unify|mgu|most general unifier/gi, '').trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Speak from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Try resolution theorem proving first
  const proof = resolutionProve(cleaned);
  if (proof) {
    response.push("\n**Resolution Proof (with unification):**");
    response.push(proof);
    response.push("\n**Mercy Conclusion:** Theorem proven by contradiction. Positive valence eternal.");
  } else {
    // Fallback to truth-table for propositional
    const table = generateTruthTable(cleaned);
    if (table) {
      response.push("\n**Truth Table (propositional logic):**");
      response.push(table);
      const conclusion = analyzeTruthTable(cleaned, table);
      response.push(`\n**Mercy Conclusion:** ${conclusion}`);
    } else {
      // Try simple unification
      const mgu = computeMGU(cleaned);
      if (mgu) {
        response.push("\n**Most General Unifier (MGU) Found:**");
        response.push(mgu);
        response.push("\n**Mercy Conclusion:** Terms are compatible under this substitution. Positive valence eternal.");
      } else {
        response.push("\n**Parser note:** Expression too complex for current engine. Mercy asks: simplify premises?");
      }
    }
  }

  // Mercy rewrite
  const mercyRewrite = cleaned
    .replace(/not/gi, '¬')
    .replace(/and/gi, '∧')
    .replace(/or/gi, '∨')
    .replace(/if/gi, '→')
    .replace(/then/gi, '')
    .replace(/implies/gi, '→')
    .replace(/iff/gi, '↔');

  response.push(`\n**Mercy Rewrite:** ${mercyRewrite}`);

  response.push("\nTruth-seeking continues: What is the core axiom behind the symbols? Positive valence eternal.");

  return response.join('\n\n');
}

// ────────────────────────────────────────────────
// Full Resolution + Unification Theorem Prover
// ────────────────────────────────────────────────

function resolutionProve(expr) {
  // Parse premises ⊢ conclusion
  const parts = expr.split('⊢');
  if (parts.length !== 2) return null;

  const premisesStr = parts[0].trim();
  const conclusionStr = parts[1].trim();

  const premises = premisesStr.split(',').map(p => p.trim());
  const clauses = premises.flatMap(p => parseClause(p));

  // Negate conclusion and add to clauses
  const negatedConclusion = negateClause(parseClause(conclusionStr)[0]);
  clauses.push(negatedConclusion);

  // Resolution loop with unification
  let steps = 0;
  const trace = ["Initial clauses (negated conclusion added):"];
  clauses.forEach((c, i) => trace.push(`${i+1}. ${clauseToString(c)}`));

  while (steps < 30) {
    steps++;
    for (let i = 0; i < clauses.length; i++) {
      for (let j = i+1; j < clauses.length; j++) {
        const resolvent = resolveClausesWithUnification(clauses[i], clauses[j]);
        if (!resolvent) continue;

        if (resolvent.length === 0) {
          trace.push(`\nEmpty clause derived after ${steps} steps — contradiction proven.`);
          return trace.join('\n');
        }

        // Only add if not subsumed
        if (!clauses.some(c => subsumes(c, resolvent))) {
          clauses.push(resolvent);
          trace.push(`${clauses.length}. ${clauseToString(resolvent)} (from ${i+1} + ${j+1})`);
        }
      }
    }
  }

  return null; // no proof within step limit
}

// ────────────────────────────────────────────────
// Unification Helpers (full Martelli–Montanari)
// ────────────────────────────────────────────────

function unify(t1, t2) {
  let subst = {};
  let stack = [[t1, t2]];

  while (stack.length > 0) {
    let [s, t] = stack.pop();

    // Delete
    if (termEqual(s, t)) continue;

    // Orient
    if (isVar(t) && !isVar(s)) [s, t] = [t, s];

    // Eliminate
    if (isVar(s)) {
      if (occurs(s, t)) return null; // occurs-check
      subst[s] = t;
      // Apply subst to stack
      stack = stack.map(([a,b]) => [applySubst(subst, a), applySubst(subst, b)]);
      continue;
    }

    // Decompose
    if (isCompound(s) && isCompound(t) && s.fun === t.fun && s.args.length === t.args.length) {
      for (let i = s.args.length - 1; i >= 0; i--) {
        stack.push([s.args[i], t.args[i]]);
      }
      continue;
    }

    // Conflict
    return null;
  }

  return subst;
}

function resolveClausesWithUnification(c1, c2) {
  for (let i = 0; i < c1.length; i++) {
    for (let j = 0; j < c2.length; j++) {
      const lit1 = c1[i];
      const lit2 = c2[j];
      if (lit1.startsWith('-') === lit2.startsWith('-')) continue; // same polarity

      const atom1 = lit1.startsWith('-') ? lit1.slice(1) : lit1;
      const atom2 = lit2.startsWith('-') ? lit2.slice(1) : lit2;

      const subst = unify(parseTerm(atom1), parseTerm(atom2));
      if (!subst) continue;

      // Build resolvent
      let resolvent = [
        ...c1.slice(0, i), ...c1.slice(i+1),
        ...c2.slice(0, j), ...c2.slice(j+1)
      ];

      // Apply substitution
      resolvent = resolvent.map(lit => applySubstToLit(subst, lit));

      return resolvent;
    }
  }
  return null;
}

function applySubstToLit(subst, lit) {
  const negated = lit.startsWith('-');
  const atom = negated ? lit.slice(1) : lit;
  const newAtom = termToString(applySubst(subst, parseTerm(atom)));
  return negated ? '-' + newAtom : newAtom;
}

// ... existing helper functions (parseTerm, termToString, isVar, isCompound, occurs, applySubst) remain ...

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...
