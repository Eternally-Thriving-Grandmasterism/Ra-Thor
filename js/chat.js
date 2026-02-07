// js/chat.js — Rathor Lattice Core (with expanded session search + tags)

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

// Send message
function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  chatMessages.innerHTML += `<div class="message user">${text}</div>`;
  chatInput.value = '';
  chatMessages.scrollTop = chatMessages.scrollHeight;
  rathorDB.saveMessage(currentSessionId, 'user', text);
  if (ttsEnabled) speak(text);
  if (translateToggle.checked) translateChat();
}

// Speak function (TTS)
function speak(text) {
  if (!ttsEnabled || !text) return;
  const utt = new SpeechSynthesisUtterance(text);
  utt.pitch = voicePitchValue;
  utt.rate = voiceRateValue;
  utt.volume = voiceVolumeValue;
  utt.lang = localStorage.getItem('rathor_output_lang') || 'en-US';
  speechSynthesis.speak(utt);
}

// Session search with tags + color indicators
function filterSessions() {
  const filter = sessionSearch.value.toLowerCase().trim();
  if (!filter) {
    Array.from(sessionSelect.options).forEach(opt => opt.style.display = '');
    return;
  }

  Array.from(sessionSelect.options).forEach(opt => {
    const session = allSessions.find(s => s.id === opt.value);
    if (!session) {
      opt.style.display = 'none';
      return;
    }

    const nameMatch = (session.name || session.id).toLowerCase().includes(filter);
    const tagMatch = session.tags?.toLowerCase().includes(filter);
    opt.style.display = nameMatch || tagMatch ? '' : 'none';

    // Add visual indicators
    if (opt.style.display !== 'none') {
      let indicator = opt.querySelector('.session-indicator');
      if (!indicator) {
        indicator = document.createElement('span');
        indicator.className = 'session-indicator';
        indicator.style.cssText = 'display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; vertical-align: middle;';
        opt.insertBefore(indicator, opt.firstChild);
      }
      indicator.style.background = session.color || '#ffaa00';
    }
  });
}

// ... (rest of functions: voice recognition, recording, processVoiceCommand, refreshSessionList, loadChatHistory, updateTranslationStats, bridge pings, etc. remain as previously deployed)

// Example session object shape (for reference)
async function refreshSessionList() {
  allSessions = await rathorDB.getAllSessions();
  sessionSelect.innerHTML = '';
  allSessions.forEach(session => {
    const option = document.createElement('option');
    option.value = session.id;
    option.textContent = session.name || session.id;
    if (session.id === currentSessionId) option.selected = true;
    sessionSelect.appendChild(option);
  });
}
