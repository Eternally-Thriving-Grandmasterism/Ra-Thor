// js/chat.js — Rathor Lattice Core with Expanded Emergency & Mental Health (IFS added)

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

// ────────────────────────────────────────────────
// Expanded Emergency & Mental Health Assistants (IFS added)
// ────────────────────────────────────────────────

const emergencyAssistants = {
  // ... previous medical, legal, crisis, mental, ptsd, cptsd objects unchanged ...

  ifs: {
    title: "IFS-Inspired Self-Help (Offline Stub)",
    disclaimer: "THIS IS NOT IFS THERAPY. Full Internal Family Systems requires a trained Level 1+ practitioner. These are only gentle, non-unburdening, Self-energy cultivation practices. Never attempt to unburden exiles or force parts to step back without professional guidance. If parts overwhelm, blend intensely, or activate trauma → stop immediately and seek trained IFS support. Rathor is NOT an IFS therapist.",
    templates: [
      {
        name: "6F Sequence Lite — Get to Know a Part",
        content: `1. Find: notice a feeling/part in your body (tight chest, inner critic voice)\n2. Focus: turn gentle attention toward it\n3. Flesh out: ask "What do you look like? How old? What do you want me to know?"\n4. Feel toward: check your feeling toward this part (curiosity, compassion?)\n5. beFriend: from Self, say "I'm here with you — thank you for protecting me"\n6. Fear: ask "What are you afraid would happen if you didn't do your job?"\nStop here — do not go deeper without therapist`
      },
      {
        name: "Self-Energy Check-In (8 Cs + 5 Ps)",
        content: `Ask inside: "How much do I feel..." (0–10 scale)\nCalm • Clarity • Curiosity • Compassion • Confidence\nCourage • Creativity • Connectedness\nPresence • Patience • Perspective • Persistence • Playfulness\nIf low → breathe, hand on heart, "May I bring more Self to this moment"`
      },
      {
        name: "Unblending from a Part",
        content: `• Notice the part is speaking through you\n• Ask: "Can you give me a little space so I (Self) can be with you?"\n• Step back: imagine part sitting beside you or in front of you\n• From Self: "Thank you for showing me this — I see you"\n• Breathe together until you feel more spacious/calm`
      },
      {
        name: "Container for Activated Parts",
        content: `• Imagine strong, beautiful container (vault, chest, bubble)\n• Invite activated part: "Would you like to rest in here until we can talk safely?"\n• Place part inside with care (can add protectors/ancestors)\n• Lock it, set intention: "This can wait for therapy"\n• Ground after: feel body in chair, name surroundings`
      },
      {
        name: "Butterfly Hug (Self-Bilateral)",
        content: `• Cross arms over chest, hands on opposite shoulders\n• Alternate gentle tapping (left-right) at your own pace\n• Breathe slowly while tapping\n• Use for anxiety, activation, or to invite calm\n• Stop if becomes overwhelming — switch to grounding`
      },
      {
        name: "Compassion Toward a Protector",
        content: `• Identify protector (anger, critic, avoidance)\n• Thank it: "I see how hard you work to keep me safe"\n• Ask: "What are you afraid would happen if you rested?"\n• From Self: "I appreciate you — may I help carry this?"\n• Breathe together until part feels seen`
      }
    ]
  }
};

function triggerEmergencyAssistant(mode) {
  const assistant = emergencyAssistants[mode];
  if (!assistant) return;

  const modal = document.createElement('div');
  modal.className = 'modal-overlay';
  modal.innerHTML = `
    <div class="modal-content emergency-modal">
      <h2 style="color: #ff4444;">${assistant.title}</h2>
      <p style="color: #ff6666; font-weight: bold; margin-bottom: 1em;">${assistant.disclaimer}</p>
      <div style="max-height: 60vh; overflow-y: auto; padding-right: 12px;">
        ${assistant.templates.map(t => `
          <h3 style="margin: 1.5em 0 0.5em; color: #ffaa00;">${t.name}</h3>
          <p style="white-space: pre-wrap; line-height: 1.6;">${t.content}</p>
        `).join('')}
      </div>
      <div class="modal-buttons" style="margin-top: 1.5em;">
        <button onclick="this.closest('.modal-overlay').remove()">Close</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  modal.style.display = 'flex';
}

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with IFS-inspired
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  // IFS-inspired triggers
  if (cmd.includes('ifs') || cmd.includes('internal family') || cmd.includes('parts') || cmd.includes('inner critic') || cmd.includes('exile') || cmd.includes('self energy') || cmd.includes('butterfly hug') || cmd.includes('unblend')) {
    triggerEmergencyAssistant('ifs');
    speak("All parts are welcome here. You are the Self leading with compassion. No part needs to be forced away.");
    return true;
  }

  // C-PTSD general
  if (cmd.includes('complex ptsd') || cmd.includes('cptsd') || cmd.includes('emotional flashback') || cmd.includes('toxic shame') || cmd.includes('toxic family') || cmd.includes('developmental trauma') || cmd.includes('interpersonal trauma')) {
    triggerEmergencyAssistant('cptsd');
    speak("You survived what was done to you. The past is over. You are safe in this moment. Help is here.");
    return true;
  }

  // ... other existing emergency/mental/medical/legal/crisis/ptsd triggers ...

  // Recording & other commands...
  if (cmd.includes('emergency mode') || cmd.includes('crisis recording')) {
    await startVoiceRecording(currentSessionId, true);
    showToast('Emergency recording started — saved with priority ⚠️');
    return true;
  }

  if (cmd.includes('stop emergency') || cmd.includes('end recording')) {
    stopVoiceRecording();
    showToast('Recording stopped & saved ⚡️');
    return true;
  }

  // ... other commands ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, session CRUD, etc.) remain unchanged ...
