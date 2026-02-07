// js/chat.js — Rathor Lattice Core with Expanded Emergency Assistants

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
// Expanded Emergency Assistants (offline-first)
// ────────────────────────────────────────────────

const emergencyAssistants = {
  medical: {
    title: "Medical Guidance (Offline Stub)",
    disclaimer: "THIS IS NOT MEDICAL ADVICE. Rathor is NOT a doctor. For emergencies call your local emergency number immediately (112 / 911 / 999 / etc.). Seek professional help as soon as possible.",
    templates: [
      {
        name: "Basic First Aid",
        content: `• Check scene safety first
• Call emergency services if unconscious, not breathing, severe bleeding or chest pain
• For bleeding: apply direct pressure, elevate if possible
• For burns: cool with running water 10–20 min, cover loosely
• Never give food/drink to unconscious person`
      },
      {
        name: "Choking Adult",
        content: `• Ask "Are you choking?" If they nod → perform Heimlich maneuver
• Stand behind, fist above navel, thumb inward, grasp fist with other hand
• Quick upward thrusts until object dislodged or person unconscious
• If unconscious → start CPR
• Call emergency services immediately`
      },
      {
        name: "Heart Attack Signs",
        content: `Common signs:
• Chest pain/pressure (may spread to arm, jaw, back)
• Shortness of breath
• Nausea, cold sweat
• Lightheadedness
Immediate action:
• Call emergency services
• Chew 325mg aspirin if available and not allergic
• Rest, loosen clothing, stay calm`
      },
      {
        name: "Stroke FAST Test",
        content: `F – Face drooping? Smile to check
A – Arm weakness? Raise both arms
S – Speech difficulty? Repeat simple sentence
T – Time to call emergency services NOW
Other signs: sudden confusion, severe headache, trouble seeing/walking
Act FAST — every minute counts`
      },
      {
        name: "Severe Bleeding",
        content: `• Apply direct pressure with clean cloth/hand
• Elevate limb if possible
• If bleeding through dressing → add more layers, do NOT remove original
• For limb: apply tourniquet only if life-threatening and trained
• Call emergency services immediately`
      }
    ]
  },

  legal: {
    title: "Legal Rights Reminder (Offline Stub)",
    disclaimer: "THIS IS NOT LEGAL ADVICE. Rathor is NOT a lawyer. Laws vary by country/jurisdiction. Consult a qualified attorney or legal aid service for your situation.",
    templates: [
      {
        name: "Police Interaction Rights (general)",
        content: `• Right to remain silent — say "I invoke my right to remain silent"
• Right to an attorney — request one immediately
• Do NOT consent to search without warrant (say "I do not consent to search")
• Ask "Am I free to go?" — if yes, leave calmly
• Record interaction if safe and legal in your area`
      },
      {
        name: "Contract Basics",
        content: `• Read everything before signing
• Verbal agreements can be binding — get written proof when possible
• Unfair terms may be unenforceable (e.g. excessive penalties)
• Cooling-off periods exist for some contracts (e.g. door-to-door sales)
• Keep copies of all signed documents`
      },
      {
        name: "Privacy & Data Rights",
        content: `• Right to know what data companies hold (subject access request)
• Right to correct inaccurate data
• Right to delete data in many cases (right to be forgotten)
• Right to object to processing (marketing, profiling)
• Report breaches to data protection authority`
      },
      {
        name: "Domestic/Family Issues",
        content: `• Everyone has right to live free from violence/abuse
• Emergency protection orders available in most jurisdictions
• Child custody determined by child's best interest
• Spousal/partner rights vary — seek local legal aid
• Hotlines exist for immediate support (search locally)`
      }
    ]
  },

  crisis: {
    title: "Crisis Grounding & Support (Offline Stub)",
    disclaimer: "If you are in immediate danger call emergency services NOW. This is only a temporary aid. Help is available — you are not alone.",
    templates: [
      {
        name: "5-4-3-2-1 Grounding",
        content: `5 things you can see
4 things you can touch
3 things you can hear
2 things you can smell
1 thing you can taste
Repeat slowly. Breathe in for 4, hold 4, out 6.`
      },
      {
        name: "Panic Attack Breathing",
        content: `Box breathing:
Inhale 4 seconds
Hold 4 seconds
Exhale 4 seconds
Hold 4 seconds
Repeat until calmer
Focus on something cold (ice cube, cold water on wrists)`
      },
      {
        name: "Suicidal Thoughts",
        content: `You are enough. This feeling is temporary.
Reach out — call a helpline NOW:
International: befrienders.org
US: 988
UK: 116 123 (Samaritans)
You matter. Help is waiting.`
      },
      {
        name: "Grief Support",
        content: `Grief has no timeline — all feelings are valid
Allow tears, memories, anger
Talk to someone safe
Self-care basics: sleep, water, small movement
Memorial ritual: write letter, light candle, speak aloud`
      },
      {
        name: "Anger De-escalation",
        content: `Step away if possible
Deep belly breaths (in nose 4, out mouth 6)
Clench & release fists 10× (progressive muscle relaxation)
Splash cold water on face
Name 5 things you feel grateful for right now`
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
      <p style="color: #ff6666; font-weight: bold;">${assistant.disclaimer}</p>
      <div style="max-height: 60vh; overflow-y: auto;">
        ${assistant.templates.map(t => `
          <h3 style="margin-top: 1.5em;">${t.name}</h3>
          <p style="white-space: pre-wrap;">${t.content}</p>
        `).join('')}
      </div>
      <div class="modal-buttons">
        <button onclick="this.closest('.modal-overlay').remove()">Close</button>
      </div>
    </div>
  `;
  document.body.appendChild(modal);
  modal.style.display = 'flex';
}

// ────────────────────────────────────────────────
// Voice Command Processor — expanded with emergency assistants
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  // Emergency assistants
  if (cmd.includes('medical help') || cmd.includes('medical advice') || cmd.includes('health emergency')) {
    triggerEmergencyAssistant('medical');
    return true;
  }

  if (cmd.includes('legal advice') || cmd.includes('legal help') || cmd.includes('rights') || cmd.includes('lawyer')) {
    triggerEmergencyAssistant('legal');
    return true;
  }

  if (cmd.includes('crisis mode') || cmd.includes('emotional support') || cmd.includes('grounding') || cmd.includes('panic') || cmd.includes('feeling bad')) {
    triggerEmergencyAssistant('crisis');
    return true;
  }

  // Existing commands
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

  if (cmd.includes('export voice notes') || cmd.includes('download recordings')) {
    await recorderDB.exportAll(currentSessionId);
    return true;
  }

  // ... other commands (test sound, bridges, etc.) ...

  return false;
}

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, session CRUD, etc.) remain unchanged ...
