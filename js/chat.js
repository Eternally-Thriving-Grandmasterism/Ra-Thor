// js/chat.js — Rathor Lattice Core with Expanded Emergency & Mental Health (EMDR-inspired added)

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
// Expanded Emergency & Mental Health Assistants (EMDR-inspired added)
// ────────────────────────────────────────────────

const emergencyAssistants = {
  medical: {
    title: "Medical Guidance (Offline Stub)",
    disclaimer: "THIS IS NOT MEDICAL ADVICE. Rathor is NOT a doctor. For emergencies call your local emergency number immediately (112 / 911 / 999 / etc.). Seek professional help as soon as possible.",
    templates: [
      { name: "Basic First Aid", content: `• Check scene safety first\n• Call emergency services if unconscious, not breathing, severe bleeding or chest pain\n• For bleeding: apply direct pressure, elevate if possible\n• For burns: cool with running water 10–20 min, cover loosely\n• Never give food/drink to unconscious person` },
      // ... other medical templates ...
    ]
  },

  legal: {
    title: "Legal Rights Reminder (Offline Stub)",
    disclaimer: "THIS IS NOT LEGAL ADVICE. Rathor is NOT a lawyer. Laws vary by country/jurisdiction. Consult a qualified attorney or legal aid service for your situation.",
    templates: [
      { name: "Police Interaction Rights (general)", content: `• Right to remain silent — say "I invoke my right to remain silent"\n• Right to an attorney — request one immediately\n• Do NOT consent to search without warrant (say "I do not consent to search")\n• Ask "Am I free to go?" — if yes, leave calmly\n• Record interaction if safe and legal in your area` },
      // ... other legal templates ...
    ]
  },

  crisis: {
    title: "Crisis Grounding & Support (Offline Stub)",
    disclaimer: "If you are in immediate danger call emergency services NOW. This is only a temporary grounding aid. Help is available — you are not alone.",
    templates: [
      { name: "5-4-3-2-1 Grounding", content: `5 things you can see\n4 things you can touch\n3 things you can hear\n2 things you can smell\n1 thing you can taste\nRepeat slowly. Breathe in for 4, hold 4, out 6.` },
      // ... other crisis templates ...
    ]
  },

  mental: {
    title: "Mental Health Support (Offline Stub)",
    disclaimer: "THIS IS NOT THERAPY OR PROFESSIONAL HELP. If you are in crisis call a helpline or emergency services immediately. Rathor is NOT a mental health professional.",
    templates: [
      { name: "Anxiety / Panic Attack", content: `• Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste\n• Slow breathing: in 4, hold 4, out 6–8\n• Cold sensation: hold ice, splash face, drink cold water\n• Ground yourself: feel feet on floor, name objects around you\n• Reminder: this will pass — you have survived 100% of bad days so far` },
      // ... other mental templates ...
    ]
  },

  ptsd: {
    title: "PTSD / Trauma Support (Offline Stub)",
    disclaimer: "THIS IS NOT THERAPY OR PROFESSIONAL HELP. If you are in crisis or flashback call a helpline or emergency services immediately. Rathor is NOT a trauma specialist. Seek trained professional support (EMDR, CPT, trauma-informed therapy).",
    templates: [
      { name: "Flashback / Dissociation Grounding", content: `• Right here, right now — name 5 things you see in the room\n• Touch something solid (chair, floor, wall)\n• Say aloud: "I am [name], I am in [place], it is [year] — the trauma is over"\n• Cold sensation: hold ice cube, splash cold water on face/wrists\n• Slow breathing: in 4, hold 4, out 6–8\n• Safe anchor: look at a photo/object that represents safety` },
      // ... other PTSD templates ...
    ]
  },

  cptsd: {
    title: "Complex PTSD (C-PTSD) Support (Offline Stub)",
    disclaimer: "THIS IS NOT THERAPY OR PROFESSIONAL HELP. Complex PTSD from prolonged/repeated trauma requires specialist support (trauma-informed therapy, EMDR, IFS, somatic experiencing). Rathor is NOT a therapist. If in crisis call a helpline or emergency services immediately.",
    templates: [
      { name: "Emotional Flashbacks", content: `• Recognize: intense shame, fear, abandonment feelings without clear present trigger\n• Ground: name 5 things you see right now, touch something solid\n• Self-talk: "This is an emotional flashback — the danger is in the past"\n• Comfort younger self: imagine hugging child-you, say "You're safe now, I'm here"\n• After: rest, hydrate, journal feelings without judgment` },
      // ... other C-PTSD templates ...
    ]
  },

  emdr: {
    title: "EMDR-Inspired Self-Help Techniques (Offline Stub)",
    disclaimer: "THIS IS NOT ACTUAL EMDR THERAPY. Full EMDR requires a trained clinician. These are only safe, non-processing self-help adaptations. Never attempt memory reprocessing alone. If overwhelmed or dissociating — stop immediately and seek professional help. Rathor is NOT an EMDR therapist.",
    templates: [
      {
        name: "Butterfly Hug (Self-Bilateral Tapping)",
        content: `• Cross arms over chest, hands on opposite shoulders\n• Alternate gentle tapping (left-right) at comfortable speed\n• Breathe slowly while tapping\n• Use when feeling anxious or activated\n• Stop if becomes overwhelming — ground with 5-4-3-2-1 instead`
      },
      {
        name: "Safe/Calm Place with Bilateral Audio",
        content: `• Close eyes, picture your safe place (real or imagined)\n• Notice colors, sounds, smells, feelings of safety\n• While holding image, tap shoulders alternately or listen to bilateral tones (search "bilateral stimulation audio" offline if saved)\n• Strengthen image: "This place is always here for me"\n• Anchor: touch thumb & finger when recalling`
      },
      {
        name: "Resource Installation (Positive Memory Anchor)",
        content: `• Think of a moment you felt strong, loved, capable\n• Notice body sensations of that strength\n• Add bilateral tapping (slow, gentle) while holding memory\n• Repeat: "I carry this strength with me"\n• Use before stressful situations`
      },
      {
        name: "Container Exercise (Store Intrusive Material)",
        content: `• Imagine a strong, lockable container (safe, unbreakable)\n• Place intrusive thoughts/images/memories inside\n• Lock it, set it aside (can be guarded by protective figure)\n• Tell yourself: "This can wait until therapy — I am safe now"\n• After: ground with 5-4-3-2-1`
      },
      {
        name: "Light Stream Technique (Body Clearing)",
        content: `• Breathe deeply\n• Imagine healing light entering top of head\n• Let light flow to tense areas, dissolving disturbance\n• Light carries discomfort out through feet into ground\n• Repeat until body feels lighter/calmer\n• Ground after with physical sensation`
      },
      {
        name: "Phase 1 Stabilization Reminders",
        content: `Before any processing — establish safety first:\n• Safe place visualization practiced\n• Container ready\n• Trusted person/phone number visible\n• Grounding techniques memorized\n• Therapy plan in place\n• If any overwhelm — stop, call support, ground`
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
// Voice Command Processor — expanded with EMDR-inspired
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  // EMDR-inspired triggers
  if (cmd.includes('emdr') || cmd.includes('bilateral') || cmd.includes('butterfly hug') || cmd.includes('safe place') || cmd.includes('container exercise') || cmd.includes('light stream') || cmd.includes('resource installation')) {
    triggerEmergencyAssistant('emdr');
    speak("You are safe. These are gentle self-help tools only. Professional EMDR requires a trained therapist.");
    return true;
  }

  // C-PTSD general
  if (cmd.includes('complex ptsd') || cmd.includes('cptsd') || cmd.includes('emotional flashback') || cmd.includes('toxic shame') || cmd.includes('toxic family') || cmd.includes('developmental trauma') || cmd.includes('interpersonal trauma')) {
    triggerEmergencyAssistant('cptsd');
    speak("You survived what was done to you. The past is over. You are safe in this moment. Help is here.");
    return true;
  }

  // PTSD general
  if (cmd.includes('ptsd') || cmd.includes('trauma') || cmd.includes('flashback') || cmd.includes('triggered') || cmd.includes('nightmare') || cmd.includes('hypervigilant') || cmd.includes('self blame') || cmd.includes('emotional numb')) {
    triggerEmergencyAssistant('ptsd');
    speak("You are safe here in this moment. The trauma is in the past. Help is available.");
    return true;
  }

  // ... other existing mental health, medical, legal, crisis triggers ...

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
