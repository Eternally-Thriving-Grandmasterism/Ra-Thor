// js/chat.js — Chat page core

// DOM references (chat-specific)
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const voiceBtn = document.getElementById('voice-btn');
const recordBtn = document.getElementById('record-btn');
const voiceOutputBtn = document.getElementById('voice-output-btn');
const voiceSettingsBtn = document.getElementById('voice-settings-btn');
const sessionSelect = document.getElementById('session-select');
const sessionSearch = document.getElementById('session-search');
const sessionSearchClear = document.getElementById('session-search-clear');
const editSessionBtn = document.getElementById('edit-session-btn');
const exportSessionBtn = document.getElementById('export-session-btn');
const importSessionBtn = document.getElementById('import-session-btn');
const duplicateSessionBtn = document.getElementById('duplicate-session-btn');
const deleteSessionBtn = document.getElementById('delete-session-btn');
const newSessionBtn = document.getElementById('new-session-btn');
const translateToggle = document.getElementById('translate-chat');
const translateLangSelect = document.getElementById('translate-lang');
const translateStatsDisplay = document.getElementById('translate-stats');

// Variables
let currentSessionId = 'default';
let isListening = false;
let isVoiceOutputEnabled = localStorage.getItem('rathor_voice_output') !== 'false';
let ttsEnabled = localStorage.getItem('rathor_tts_enabled') !== 'false';
let voicePitchValue = parseFloat(localStorage.getItem('rathor_pitch')) || 1.0;
let voiceRateValue = parseFloat(localStorage.getItem('rathor_rate')) || 1.0;
let voiceVolumeValue = parseFloat(localStorage.getItem('rathor_volume')) || 1.0;
let feedbackVolumeValue = parseFloat(localStorage.getItem('rathor_feedback_volume')) || 0.6;
let feedbackSoundsEnabled = localStorage.getItem('rathor_feedback_sounds') !== 'false';
let selectedInputLang = localStorage.getItem('rathor_input_lang') || 'auto';
let selectedOutputLang = localStorage.getItem('rathor_output_lang') || 'auto';

// Audio Context & TTS
let audioContext = null;
let ttsUtterance = null;

function initAudioContext() {
  if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
}

function speak(text) {
  if (!ttsEnabled || !text) return;
  if (ttsUtterance) speechSynthesis.cancel();
  ttsUtterance = new SpeechSynthesisUtterance(text);
  ttsUtterance.lang = selectedOutputLang === 'auto' ? navigator.language : selectedOutputLang;
  ttsUtterance.pitch = voicePitchValue;
  ttsUtterance.rate = voiceRateValue;
  ttsUtterance.volume = voiceVolumeValue;
  speechSynthesis.speak(ttsUtterance);
}

// ... full voice recognition, recording, processVoiceCommand, session CRUD, bridge status, etc. from previous complete script ...

window.addEventListener('load', async () => {
  await initI18n(); // from common.js
  updateContent();
  // ... chat-specific init ...
});
