// js/chat.js — Rathor Lattice Core with Theorem Proving Stub

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
// Symbolic Query Mode — Mercy-First Truth-Seeking with Theorem Proving Stub
// ────────────────────────────────────────────────

function isSymbolicQuery(cmd) {
  return cmd.includes('symbolic query') || cmd.includes('logical analysis') || 
         cmd.includes('truth mode') || cmd.includes('first principles') ||
         cmd.includes('truth table') || cmd.includes('logical table') ||
         cmd.includes('prove') || cmd.includes('theorem') || cmd.includes('resolution') ||
         cmd.includes('⊢') || cmd.includes('reason from first principles') || cmd.includes('symbolic reasoning');
}

function symbolicQueryResponse(query) {
  const cleaned = query.trim().replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles|prove|theorem|resolution/gi, '').trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Speak from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Try theorem proving stub first
  const proof = resolutionProve(cleaned);
  if (proof) {
    response.push("\n**Resolution Proof Stub:**");
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
      response.push("\n**Parser note:** Expression too complex for current engine (max 4 variables or simple resolution). Mercy asks: simplify premises?");
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
// Basic Resolution Theorem Prover Stub
// ────────────────────────────────────────────────

function resolutionProve(expr) {
  // Very basic stub — assumes premises ⊢ conclusion format
  // e.g. "A → B, A ⊢ B"

  const parts = expr.split('⊢');
  if (parts.length !== 2) return null;

  const premisesStr = parts[0].trim();
  const conclusionStr = parts[1].trim();

  // Parse premises into clauses (very naive)
  const premises = premisesStr.split(',').map(p => p.trim());
  const clauses = premises.flatMap(p => parseClause(p));

  const conclusion = parseClause(conclusionStr)[0]; // assume single literal conclusion

  // Add negation of conclusion
  clauses.push(negateClause(conclusion));

  // Resolution loop (max 20 steps for stub)
  let steps = 0;
  const trace = ["Initial clauses:"];
  clauses.forEach((c, i) => trace.push(`${i+1}. ${c.join(' ∨ ')}`));

  while (steps < 20) {
    steps++;
    for (let i = 0; i < clauses.length; i++) {
      for (let j = i+1; j < clauses.length; j++) {
        const resolvent = resolveClauses(clauses[i], clauses[j]);
        if (resolvent === null) continue; // no resolution
        if (resolvent.length === 0) {
          trace.push(`\nEmpty clause derived after ${steps} steps — contradiction proven.`);
          return trace.join('\n');
        }
        if (!clauses.some(c => c.join('') === resolvent.join(''))) {
          clauses.push(resolvent);
          trace.push(`${clauses.length}. ${resolvent.join(' ∨ ')} (from ${i+1} + ${j+1})`);
        }
      }
    }
  }

  return null; // no proof found in limit
}

// Naive clause parser (A ∨ B ∨ ¬C → [A, B, -C])
function parseClause(str) {
  return str.split('∨').map(l => l.trim().replace('¬', '-'));
}

function negateClause(clause) {
  return clause.map(l => l.startsWith('-') ? l.slice(1) : '-' + l);
}

function resolveClauses(c1, c2) {
  for (let lit1 of c1) {
    for (let lit2 of c2) {
      if (lit1 === '-' + lit2 || lit2 === '-' + lit1) {
        const resolvent = [...new Set([...c1, ...c2].filter(l => l !== lit1 && l !== lit2))];
        return resolvent;
      }
    }
  }
  return null;
}

// ... existing truth-table functions (generateTruthTable, evaluateExpression, analyzeTruthTable) remain ...

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with symbolic query
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  if (isSymbolicQuery(cmd)) {
    const query = cmd.replace(/symbolic query|logical analysis|truth mode|truth table|logical table|first principles/gi, '').trim();
    const answer = symbolicQueryResponse(query);
    chatMessages.innerHTML += `<div class="message rathor">${answer}</div>`;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    if (ttsEnabled) speak(answer);
    return true;
  }

  // ... all previous commands (medical, legal, crisis, mental, ptsd, cptsd, ifs, emdr, recording, export, import, etc.) ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...
