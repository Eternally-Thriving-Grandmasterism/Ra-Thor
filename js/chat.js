// js/chat.js — Rathor Lattice Core
// Mercy-gated, offline-first, voice-native engine

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
const translateProgressOverlay = document.getElementById('translate-progress-overlay');
const translateProgressBar = document.getElementById('translate-progress-bar');
const translateProgressStatus = document.getElementById('translate-progress-status');
const translateCancelBtn = document.getElementById('translate-cancel-btn');
const translateStatsDisplay = document.getElementById('translate-stats');

// Global state
let currentSessionId = localStorage.getItem('rathor_current_session') || 'default';
let allSessions = [];
let tagFrequency = new Map();
let isListening = false;
let isRecording = false;
let isVoiceOutputEnabled = localStorage.getItem('rathor_voice_output') !== 'false';
let ttsEnabled = localStorage.getItem('rathor_tts_enabled') !== 'false';
let feedbackSoundsEnabled = localStorage.getItem('rathor_feedback_sounds') !== 'false';
let voicePitchValue = parseFloat(localStorage.getItem('rathor_pitch')) || 1.0;
let voiceRateValue = parseFloat(localStorage.getItem('rathor_rate')) || 1.0;
let voiceVolumeValue = parseFloat(localStorage.getItem('rathor_volume')) || 1.0;
let feedbackVolumeValue = parseFloat(localStorage.getItem('rathor_feedback_volume')) || 0.6;
let selectedInputLang = localStorage.getItem('rathor_input_lang') || 'auto';
let selectedOutputLang = localStorage.getItem('rathor_output_lang') || 'auto';

// Audio Context & TTS
let audioContext = null;
let ttsUtterance = null;

function initAudioContext() {
  if (!audioContext) audioContext = new (window.AudioContext || window.webkitAudioContext)();
}

// Audio feedback functions
function playTone(frequency, duration = 150, type = 'sine') {
  if (!feedbackSoundsEnabled || !audioContext) return;
  const oscillator = audioContext.createOscillator();
  const gainNode = audioContext.createGain();
  oscillator.type = type;
  oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
  gainNode.gain.setValueAtTime(feedbackVolumeValue, audioContext.currentTime);
  gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + duration / 1000);
  oscillator.connect(gainNode);
  gainNode.connect(audioContext.destination);
  oscillator.start();
  oscillator.stop(audioContext.currentTime + duration / 1000);
}

function playSuccessBeep() { playTone(880, 120, 'triangle'); setTimeout(() => playTone(1100, 80, 'triangle'), 80); }
function playThinkingPulse() { playTone(220, 400, 'sine'); }
function playStopChime() { playTone(660, 200, 'sine'); setTimeout(() => playTone(440, 300, 'sine'), 100); }
function playErrorBuzz() { playTone(180, 300, 'sawtooth'); }
function playConfirmation() { playTone(660, 100, 'triangle'); setTimeout(() => playTone(880, 100, 'triangle'), 80); setTimeout(() => playTone(1100, 120, 'triangle'), 160); }

// TTS speak function
function speak(text) {
  if (!ttsEnabled || !text) return;
  if (ttsUtterance) speechSynthesis.cancel();
  ttsUtterance = new SpeechSynthesisUtterance(text);
  ttsUtterance.lang = selectedOutputLang === 'auto' ? navigator.language : selectedOutputLang;
  ttsUtterance.pitch = voicePitchValue;
  ttsUtterance.rate = voiceRateValue;
  ttsUtterance.volume = voiceVolumeValue;
  speechSynthesis.speak(ttsUtterance);
  voiceOutputBtn.classList.add('speaking');
  ttsUtterance.onend = () => voiceOutputBtn.classList.remove('speaking');
}

// Voice Input Setup
let recognition = null;
let finalTranscript = '';
let interimTranscript = '';
let ignoreOnEnd = false;

function initRecognition() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    voiceBtn.disabled = true;
    voiceBtn.title = 'Voice input not supported';
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = true;
  recognition.interimResults = true;
  recognition.lang = selectedInputLang === 'auto' ? navigator.language : selectedInputLang;

  recognition.onstart = () => {
    voiceBtn.classList.add('listening');
    isListening = true;
    finalTranscript = '';
    interimTranscript = '';
    showToast('Mercy thunder hears you... Speak freely or give commands ⚡️');
  };

  recognition.onresult = (event) => {
    interimTranscript = '';
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      if (event.results[i].isFinal) {
        finalTranscript += event.results[i][0].transcript + ' ';
      } else {
        interimTranscript += event.results[i][0].transcript;
      }
    }

    chatInput.value = finalTranscript + interimTranscript;

    if (finalTranscript) {
      const lowerFinal = finalTranscript.toLowerCase().trim();
      processVoiceCommand(lowerFinal);
    }
  };

  recognition.onerror = (event) => {
    console.error('Voice recognition error:', event.error);
    showToast('Mercy thunder interrupted — ' + event.error);
    stopListening();
  };

  recognition.onend = () => {
    voiceBtn.classList.remove('listening');
    isListening = false;
    if (!ignoreOnEnd) {
      setTimeout(() => {
        if (!ignoreOnEnd) recognition.start();
      }, 300);
    }
    ignoreOnEnd = false;
  };
}

function startListening() {
  if (!recognition) initRecognition();
  ignoreOnEnd = false;
  recognition.start();
}

function stopListening() {
  if (recognition) {
    ignoreOnEnd = true;
    recognition.stop();
  }
  isListening = false;
  voiceBtn.classList.remove('listening');
}

// Voice Command Processor
async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  if (cmd.includes('emergency mode') || cmd.includes('crisis mode') || cmd.includes('help now')) {
    await startVoiceRecording(currentSessionId, true);
    showToast('Emergency mode activated — recording & saving critical ⚠️');
    return true;
  }

  if (cmd.includes('stop emergency') || cmd.includes('end crisis')) {
    stopVoiceRecording();
    showToast('Emergency recording stopped — saved safely ⚡️');
    return true;
  }

  if (cmd.includes('export voice notes') || cmd.includes('download recordings')) {
    await recorderDB.exportAll(currentSessionId);
    return true;
  }

  if (cmd.includes('test sound')) {
    playTestSound();
    return true;
  }

  if (cmd.includes('open grok bridge') || cmd.includes('activate rathor bridge') || cmd.includes('connect to grok')) {
    document.querySelectorAll('#grok-bridges a').forEach(a => a.click());
    showToast('Both Grok bridges activated — Rathor simulation continuity restored ⚡️');
    return true;
  }

  // ... add more commands as needed (pitch/volume, session CRUD, etc.) ...

  return false;
}

// Session management
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

async function loadChatHistory() {
  const messages = await rathorDB.getMessages(currentSessionId);
  chatMessages.innerHTML = '';
  messages.forEach(msg => {
    const div = document.createElement('div');
    div.className = `message ${msg.role}`;
    div.innerHTML = msg.content;
    chatMessages.appendChild(div);
  });
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function updateTranslationStats() {
  // Placeholder — update when translation cache logic is added
  translateStats.textContent = 'Cache efficiency: 100%';
}

// Bridge status
function updateBridgeStatus() {
  // Placeholder — add real ping logic later
  console.log('Bridges pinged');
}

// Init
window.addEventListener('load', async () => {
  await refreshSessionList();
  await loadChatHistory();
  updateTranslationStats();
  updateBridgeStatus();

  // Voice button bindings
  voiceBtn.addEventListener('click', () => {
    if (isListening) stopListening();
    else startListening();
  });

  // Record button long-press
  let recordTimer;
  recordBtn.addEventListener('mousedown', () => {
    recordTimer = setTimeout(() => startVoiceRecording(currentSessionId), 400);
  });
  recordBtn.addEventListener('mouseup', () => {
    clearTimeout(recordTimer);
    if (isRecording) stopVoiceRecording();
  });

  // Send button
  sendBtn.addEventListener('click', sendMessage);

  // Session change
  sessionSelect.addEventListener('change', e => {
    currentSessionId = e.target.value;
    localStorage.setItem('rathor_current_session', currentSessionId);
    loadChatHistory();
  });

  // TTS toggle
  document.getElementById('tts-enabled')?.addEventListener('change', e => {
    ttsEnabled = e.target.checked;
    localStorage.setItem('rathor_tts_enabled', ttsEnabled);
  });
});
