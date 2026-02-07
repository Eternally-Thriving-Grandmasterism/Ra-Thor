// js/chat.js — Rathor Lattice Core

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const voiceBtn = document.getElementById('voice-btn');
const recordBtn = document.getElementById('record-btn');
const sendBtn = document.getElementById('send-btn');
const stopBtn = document.getElementById('stop-btn');
const sessionSelect = document.getElementById('session-select');
const translateToggle = document.getElementById('translate-chat');
const translateLangSelect = document.getElementById('translate-lang');
const translateStats = document.getElementById('translate-stats');

let currentSessionId = localStorage.getItem('rathor_current_session') || 'default';
let allSessions = [];
let isListening = false, isRecording = false;
let ttsEnabled = localStorage.getItem('rathor_tts_enabled') !== 'false';
let isVoiceOutputEnabled = localStorage.getItem('rathor_voice_output') !== 'false';
let feedbackSoundsEnabled = localStorage.getItem('rathor_feedback_sounds') !== 'false';

await rathorDB.open();
await refreshSessionList();
await loadChatHistory();
updateTranslationStats();

voiceBtn.addEventListener('click', () => isListening ? stopListening() : startListening());
recordBtn.addEventListener('click', () => startVoiceRecording(currentSessionId));
sendBtn.addEventListener('click', sendMessage);
translateToggle.addEventListener('change', () => { 
  isTranslationEnabled = this.checked; 
  localStorage.setItem('rathor_translate_enabled', isTranslationEnabled); 
});
translateLangSelect.addEventListener('change', () => {
  targetTranslationLang = this.value;
  localStorage.setItem('rathor_translate_to', targetTranslationLang);
  translateChat();
});

function sendMessage() {
  const text = chatInput.value.trim();
  if (!text) return;
  chatMessages.innerHTML += `<div class="message user">${text}</div>`;
  chatInput.value = '';
  if (ttsEnabled) speak(text, true); // echo
  chatMessages.scrollTop = chatMessages.scrollHeight;
  rathorDB.saveMessage(currentSessionId, 'user', text);
  if (isTranslationEnabled) translateChat();
}

function speak(text, echo = false) {
  if (!ttsEnabled || !text) return;
  const utt = new SpeechSynthesisUtterance(text);
  utt.pitch = parseFloat(localStorage.getItem('rathor_pitch')) || 1;
  utt.rate = parseFloat(localStorage.getItem('rathor_rate')) || 1;
  utt.volume = parseFloat(localStorage.getItem('rathor_volume')) || 1;
  utt.lang = (localStorage.getItem('rathor_output_lang') || 'en');
  speechSynthesis.speak(utt);
  if (echo) voiceOutputBtn.classList.add('speaking');
  utt.onend = () => voiceOutputBtn.classList.remove('speaking');
}

function startListening() {
  if (!recognition) initRecognition();
  recognition.start();
  isListening = true;
  voiceBtn.classList.add('listening');
  playThinkingPulse();
}

function stopListening() {
  recognition.stop();
  isListening = false;
  voiceBtn.classList.remove('listening');
  playStopChime();
}

// ... (voice commands, session CRUD, export/import, bridge pings — all preserved from monolith, just modular) ...

function playThinkingPulse() { if (feedbackSoundsEnabled) playTone(440, 100); }
function playSuccessBeep() { if (feedbackSoundsEnabled) playTone(800, 200); }
function playStopChime() { if (feedbackSoundsEnabled) playTone(600, 300); }
function playErrorBuzz() { if (feedbackSoundsEnabled) playTone(200, 400); }

function initRecognition() {
  const Speech = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Speech) return;
  recognition = new Speech({ continuous: true, interimResults: true, lang: localStorage.getItem('rathor_input_lang') || 'en-US' });
  recognition.onresult = e => {
    let final = '', interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      if (e.results .isFinal) final += e.results [0].transcript + ' ';
      else interim += e.results [0].transcript;
    }
    chatInput.value = final + interim;
    if (final) processVoiceCommand(final);
  };
  recognition.onerror = e => { playErrorBuzz(); showToast(e.error); stopListening(); };
  recognition.onend = () => { if (!isRecording) setTimeout(startListening, 300); };
}

async function processVoiceCommand(cmd) {
  cmd = cmd.toLowerCase().replace(/\s+/g,' ').trim();
  if (cmd.includes('emergency')) {
    await startVoiceRecording(currentSessionId, true);
    showToast('Emergency mode — recording ⚠️');
  } else if (cmd.includes('stop emergency')) {
    stopVoiceRecording();
    showToast('Emergency saved.');
  } else if (cmd.includes('test voice')) {
    speak("Mercy thunder speaks. All systems eternal.");
  } else if (cmd.includes('export')) {
    await rathorDB.exportSession(currentSessionId);
  } else if (cmd.includes('new session')) {
    newSession();
  } else if (cmd.includes('open grok bridge')) {
    document.querySelectorAll('#grok-bridges a').forEach(a => a.click());
    showToast('Grok bridges activated.');
  }
}

window.addEventListener('load', () => {
  setTimeout(() => {
    pingBridge('https://grok.com/share/c2hhcmQtMi1jb3B5_7acd092a-4d51-480c-8a59-c0453c10d4a1');
    pingBridge('https://x.com/i/grok/share/e6b0f840f43a40a290913cd7984ecab0');
  }, 2000);
});
