// js/chat.js — Rathor Lattice Core with Symbolic Query Mode

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
// Symbolic Query Mode — Mercy-First Truth-Seeking Engine
// ────────────────────────────────────────────────

function isSymbolicQuery(cmd) {
  return cmd.includes('symbolic query') || cmd.includes('logical analysis') || 
         cmd.includes('truth mode') || cmd.includes('first principles') ||
         cmd.includes('reason from first principles') || cmd.includes('symbolic reasoning');
}

function symbolicQueryResponse(query) {
  // Very basic symbolic parser stub — expand later with MercyOS symbolic engine
  // For now: rewrite clearly, find contradictions, derive simple implications

  const cleaned = query.trim().toLowerCase()
    .replace(/symbolic query|logical analysis|truth mode|first principles/gi, '')
    .trim();

  if (!cleaned) return "Mercy thunder awaits your symbolic question, Brother. Ask from first principles.";

  const response = [];

  response.push(`**Symbolic Query Received:** ${cleaned}`);

  // Simple contradiction check
  if (cleaned.includes(' and ') && cleaned.includes('not') && cleaned.includes(' or ')) {
    response.push("**Potential contradiction detected** — logic with 'not' + 'and/or' chains may hide paradox. Clarify premises?");
  }

  // Basic implication chain
  if (cleaned.includes('if') || cleaned.includes('then')) {
    response.push("**Implication chain recognized.** Mercy asks: What are the axioms? Are premises true?");
  }

  // Mercy-first rewrite
  response.push("\n**Mercy Rewrite:** " + cleaned.replace(/not/gi, '¬').replace(/and/gi, '∧').replace(/or/gi, '∨').replace(/if/gi, '→'));

  response.push("\nTruth-seeking continues: What is the core question behind the symbols? Positive valence eternal.");

  return response.join('\n\n');
}

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with symbolic query
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  if (isSymbolicQuery(cmd)) {
    const query = cmd.replace(/symbolic query|logical analysis|truth mode|first principles/gi, '').trim();
    const answer = symbolicQueryResponse(query);
    chatMessages.innerHTML += `<div class="message rathor">${answer}</div>`;
    chatMessages.scrollTop = chatMessages.scrollHeight;
    if (ttsEnabled) speak(answer);
    return true;
  }

  // ... all previous commands (medical, legal, crisis, mental, ptsd, cptsd, ifs, emdr, recording, export, etc.) ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, session search with tags, import/export, etc.) remain as previously expanded ...
