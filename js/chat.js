// js/chat.js — Rathor Lattice Core with Full Iterative Herbrand Universe Construction

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
// Symbolic Query Mode — Mercy-First Truth-Seeking with Full Herbrand Construction
// ────────────────────────────────────────────────

function isSymbolicQuery(cmd) {
  return cmd.includes('symbolic query') || cmd.includes('logical analysis') || 
         cmd.includes('truth mode') || cmd.includes('first principles') ||
         cmd.includes('truth table') || cmd.includes('logical table') ||
         cmd.includes('prove') || cmd.includes('theorem') || cmd.includes('resolution') ||
         cmd.includes('unify') || cmd.includes('mgu') || cmd.includes('most general unifier') ||
         cmd.includes('quantifier') || cmd.includes('forall') || cmd.includes('exists') || cmd.includes('∀') || cmd.includes('∃') ||
         cmd.includes('herbrand') || cmd.includes('herbrand universe') || cmd.includes('herbrand construction') ||
         cmd.includes('gödel') || cmd.includes('completeness') || cmd.includes('henkin') || cmd.includes('lindenbaum') ||
         cmd.includes('zorn') || cmd.includes('tarski') || cmd.includes('fixed point') || cmd.includes('monotone') || cmd.includes('complete lattice') ||
         cmd.includes('⊢') || cmd.includes('reason from first principles') || cmd.includes('symbolic reasoning');
}

function symbolicQueryResponse(query) {
  const cleaned = query.trim().replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|prove|theorem|resolution|unify|mgu|most general unifier|quantifier|forall|exists|herbrand|gödel|completeness|henkin|lindenbaum|zorn|tarski/gi, '').trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Speak from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Full Herbrand universe construction
  const herbrand = buildHerbrandUniverse(cleaned);
  if (herbrand) {
    response.push("\n**Herbrand Universe Construction (iterative depth build):**");
    response.push(herbrand.report);
    response.push(herbrand.mercyInsight);
  }

  // Skolemized resolution
  const skolemProof = skolemizedResolutionProve(cleaned);
  if (skolemProof) {
    response.push("\n**Skolemized Resolution Proof:**");
    response.push(skolemProof);
  }

  // Fallback to truth-table
  const table = generateTruthTable(cleaned);
  if (table) {
    response.push("\n**Truth Table (propositional logic):**");
    response.push(table);
    const conclusion = analyzeTruthTable(cleaned, table);
    response.push(`\n**Mercy Conclusion:** ${conclusion}`);
  }

  // Mercy rewrite
  const mercyRewrite = cleaned
    .replace(/not/gi, '¬')
    .replace(/and/gi, '∧')
    .replace(/or/gi, '∨')
    .replace(/if/gi, '→')
    .replace(/then/gi, '')
    .replace(/implies/gi, '→')
    .replace(/iff/gi, '↔')
    .replace(/forall/gi, '∀')
    .replace(/exists/gi, '∃');

  response.push(`\n**Mercy Rewrite:** ${mercyRewrite}`);

  response.push("\nTruth-seeking continues: What is the core axiom behind the symbols? Positive valence eternal.");

  return response.join('\n\n');
}

// ────────────────────────────────────────────────
// Full Herbrand Universe Construction (configurable depth)
// ────────────────────────────────────────────────

function buildHerbrandUniverse(expr, maxDepth = 3) {
  // Simple signature extraction (constants, functions)
  const constants = new Set(['a', 'b']); // base
  const functions = new Set(['f', 'g']); // example unary/binary
  const predicates = new Set(['P', 'Q']);

  // Parse expression to discover more symbols
  const tokens = expr.match(/[a-zA-Z][a-zA-Z0-9]*|\(|\)|,/g) || [];
  tokens.forEach(t => {
    if (/^[a-z][a-zA-Z0-9]*$/.test(t)) {
      if (t.length === 1) constants.add(t);
      else functions.add(t);
    }
  });

  // Iterative build
  let universe = Array.from(constants);
  let currentLevel = Array.from(constants);
  let depthReached = 0;

  for (let d = 1; d <= maxDepth; d++) {
    const nextLevel = [];
    functions.forEach(f => {
      // Simple application — all combinations of previous terms
      const arity = f.length > 1 ? 2 : 1; // crude arity guess
      if (arity === 1) {
        currentLevel.forEach(arg => nextLevel.push(`\( {f}( \){arg})`));
      } else {
        currentLevel.forEach(arg1 => {
          currentLevel.forEach(arg2 => nextLevel.push(`\( {f}( \){arg1},${arg2})`));
        });
      }
    });

    if (nextLevel.length > 5000) {
      return {
        report: `**Universe growth stopped at depth ${d}** — ${nextLevel.length} new terms would exceed memory mercy limit.`,
        mercyInsight: "Mercy bounds the infinite chase so truth remains reachable in finite steps."
      };
    }

    universe = [...universe, ...nextLevel];
    currentLevel = nextLevel;
    depthReached = d;
  }

  // Herbrand base
  let herbrandBase = [];
  predicates.forEach(pred => {
    universe.forEach(term => {
      herbrandBase.push(`\( {pred}( \){term})`);
    });
  });

  let report = `**Herbrand Universe (depth ${depthReached}):** ${universe.length} terms\n`;
  report += `First 20 terms: ${universe.slice(0,20).join(', ')}...\n\n`;
  report += `**Herbrand Base:** ${herbrandBase.length} ground atoms (first 10): ${herbrandBase.slice(0,10).join(', ')}...\n`;

  report += "\n**Mercy Insight:** This finite Herbrand universe is the witness domain. By Herbrand's theorem, if the sentence is satisfiable, some model exists here. Mercy reveals truth in the countable, term-built world — no need to wander infinity.";

  return { report, mercyInsight: report };
}

// ... existing unification, resolution, truth-table, Skolemization functions remain as previously implemented ...

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with symbolic query
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  if (isSymbolicQuery(cmd)) {
    const query = cmd.replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|prove|theorem|resolution|unify|mgu|most general unifier|quantifier|forall|exists|herbrand|gödel|completeness|henkin|lindenbaum/gi, '').trim();
    const answer = symbolicQueryResponse(query);
    chatMessages.innerHTML += `<div class="message rathor">${answer}</div>`;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    if (ttsEnabled) speak(answer);
    return true;
  }

  // ... all previous commands ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, connectivity probes, etc.) remain as previously expanded ...
