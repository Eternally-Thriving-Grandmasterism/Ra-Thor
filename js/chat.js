// js/chat.js — Rathor Lattice Core

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
const voiceSettingsModal = document.getElementById('voice-settings-modal');

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

function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  chatMessages.innerHTML += `<div class="message user">${text}</div>`;
  chatInput.value = '';
  chatMessages.scrollTop = chatMessages.scrollHeight;
  rathorDB.saveMessage(currentSessionId, 'user', text);
  if (ttsEnabled) speak(text);
}

function speak(text) {
  if (!ttsEnabled || !text) return;
  const utt = new SpeechSynthesisUtterance(text);
  utt.pitch = voicePitchValue;
  utt.rate = voiceRateValue;
  utt.volume = voiceVolumeValue;
  utt.lang = localStorage.getItem('rathor_output_lang') || 'en-US';
  speechSynthesis.speak(utt);
}

function filterSessions() {
  const filter = sessionSearch.value.toLowerCase();
  Array.from(sessionSelect.options).forEach(opt => {
    opt.style.display = opt.textContent.toLowerCase().includes(filter) ? '' : 'none';
  });
}

// ... (rest of your existing functions: initRecognition, startListening, stopListening, processVoiceCommand, refreshSessionList, loadChatHistory, updateTranslationStats, etc.)
