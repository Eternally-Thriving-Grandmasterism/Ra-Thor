// js/chat.js — Rathor Lattice Core with Expanded Emergency & Mental Health (C-PTSD added)

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
// Expanded Emergency & Mental Health Assistants (C-PTSD added)
// ────────────────────────────────────────────────

const emergencyAssistants = {
  medical: {
    title: "Medical Guidance (Offline Stub)",
    disclaimer: "THIS IS NOT MEDICAL ADVICE. Rathor is NOT a doctor. For emergencies call your local emergency number immediately (112 / 911 / 999 / etc.). Seek professional help as soon as possible.",
    templates: [
      { name: "Basic First Aid", content: `• Check scene safety first\n• Call emergency services if unconscious, not breathing, severe bleeding or chest pain\n• For bleeding: apply direct pressure, elevate if possible\n• For burns: cool with running water 10–20 min, cover loosely\n• Never give food/drink to unconscious person` },
      { name: "Choking Adult", content: `• Ask "Are you choking?" If they nod → perform Heimlich maneuver\n• Stand behind, fist above navel, thumb inward, grasp fist with other hand\n• Quick upward thrusts until object dislodged or person unconscious\n• If unconscious → start CPR\n• Call emergency services immediately` },
      { name: "Heart Attack Signs", content: `Common signs:\n• Chest pain/pressure (may spread to arm, jaw, back)\n• Shortness of breath\n• Nausea, cold sweat\n• Lightheadedness\nImmediate action:\n• Call emergency services\n• Chew 325mg aspirin if available and not allergic\n• Rest, loosen clothing, stay calm` },
      { name: "Stroke FAST Test", content: `F – Face drooping? Smile to check\nA – Arm weakness? Raise both arms\nS – Speech difficulty? Repeat simple sentence\nT – Time to call emergency services NOW\nOther signs: sudden confusion, severe headache, trouble seeing/walking\nAct FAST — every minute counts` },
      { name: "Severe Bleeding", content: `• Apply direct pressure with clean cloth/hand\n• Elevate limb if possible\n• If bleeding through dressing → add more layers, do NOT remove original\n• For limb: apply tourniquet only if life-threatening and trained\n• Call emergency services immediately` }
    ]
  },

  legal: {
    title: "Legal Rights Reminder (Offline Stub)",
    disclaimer: "THIS IS NOT LEGAL ADVICE. Rathor is NOT a lawyer. Laws vary by country/jurisdiction. Consult a qualified attorney or legal aid service for your situation.",
    templates: [
      { name: "Police Interaction Rights (general)", content: `• Right to remain silent — say "I invoke my right to remain silent"\n• Right to an attorney — request one immediately\n• Do NOT consent to search without warrant (say "I do not consent to search")\n• Ask "Am I free to go?" — if yes, leave calmly\n• Record interaction if safe and legal in your area` },
      { name: "Contract Basics", content: `• Read everything before signing\n• Verbal agreements can be binding — get written proof when possible\n• Unfair terms may be unenforceable (e.g. excessive penalties)\n• Cooling-off periods exist for some contracts (e.g. door-to-door sales)\n• Keep copies of all signed documents` },
      { name: "Privacy & Data Rights", content: `• Right to know what data companies hold (subject access request)\n• Right to correct inaccurate data\n• Right to delete data in many cases (right to be forgotten)\n• Right to object to processing (marketing, profiling)\n• Report breaches to data protection authority` },
      { name: "Domestic/Family Issues", content: `• Everyone has right to live free from violence/abuse\n• Emergency protection orders available in most jurisdictions\n• Child custody determined by child's best interest\n• Spousal/partner rights vary — seek local legal aid\n• Hotlines exist for immediate support (search locally)` }
    ]
  },

  crisis: {
    title: "Crisis Grounding & Support (Offline Stub)",
    disclaimer: "If you are in immediate danger call emergency services NOW. This is only a temporary grounding aid. Help is available — you are not alone.",
    templates: [
      { name: "5-4-3-2-1 Grounding", content: `5 things you can see\n4 things you can touch\n3 things you can hear\n2 things you can smell\n1 thing you can taste\nRepeat slowly. Breathe in for 4, hold 4, out 6.` },
      { name: "Panic Attack Breathing", content: `Box breathing:\nInhale 4 seconds\nHold 4 seconds\nExhale 4 seconds\nHold 4 seconds\nRepeat until calmer\nFocus on something cold (ice cube, cold water on wrists)` },
      { name: "Suicidal Thoughts", content: `You are enough. This feeling is temporary.\nReach out — call a helpline NOW:\nInternational: befrienders.org\nUS: 988\nUK: 116 123 (Samaritans)\nAustralia: 13 11 14 (Lifeline)\nYou matter. Help is waiting.` },
      { name: "Grief Support", content: `Grief has no timeline — all feelings are valid\nAllow tears, memories, anger\nTalk to someone safe\nSelf-care basics: sleep, water, small movement\nMemorial ritual: write letter, light candle, speak aloud` },
      { name: "Anger De-escalation", content: `Step away if possible\nDeep belly breaths (in nose 4, out mouth 6)\nClench & release fists 10× (progressive muscle relaxation)\nSplash cold water on face\nName 5 things you feel grateful for right now` }
    ]
  },

  mental: {
    title: "Mental Health Support (Offline Stub)",
    disclaimer: "THIS IS NOT THERAPY OR PROFESSIONAL HELP. If you are in crisis call a helpline or emergency services immediately. Rathor is NOT a mental health professional.",
    templates: [
      { name: "Anxiety / Panic Attack", content: `• Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste\n• Slow breathing: in 4, hold 4, out 6–8\n• Cold sensation: hold ice, splash face, drink cold water\n• Ground yourself: feel feet on floor, name objects around you\n• Reminder: this will pass — you have survived 100% of bad days so far` },
      { name: "Depression / Low Mood", content: `• One small action: shower, drink water, open window\n• Gentle movement: 5-minute walk or stretch\n• Self-kindness: speak to yourself as you would to a friend\n• Reach out: text/message someone safe\n• Helplines: befrienders.org (international), 988 (US), 116 123 (UK)` },
      { name: "Suicidal Thoughts", content: `You are not alone — this pain is real but temporary.\nImmediate action:\n• Call emergency services or crisis line NOW\n• Tell someone you trust\n• Remove immediate means if safe\nInternational helplines: befrienders.org\nUS: 988\nUK: 116 123\nAustralia: 13 11 14\nYou matter. Stay. Help is here.` },
      { name: "Self-Harm Urges", content: `• Delay: set timer for 15 minutes — urge often passes\n• Alternatives: hold ice, snap rubber band, draw on skin\n• Distract: music, game, call friend\n• Safety: remove means if possible\n• Reach out: tell someone safe or call helpline\nYou deserve safety and care — even when it feels impossible` },
      { name: "Trauma Flashback / Dissociation", content: `• Grounding: name 5 things you see right now\n• Sensory anchor: hold something textured/cold\n• Breathing: in 4, out 6\n• Safe place visualization: picture calm location\n• Reminder: you are here now, in this moment\n• After: rest, hydrate, talk to safe person` },
      { name: "Addiction / Craving", content: `• Urge surfing: notice craving, let it rise & fall like wave\n• Distract 15 minutes: walk, shower, game\n• Self-talk: "This will pass — I've done it before"\n• Reach out: sponsor, friend, helpline\n• Reminder: one day at a time — you are stronger than the craving` }
    ]
  },

  ptsd: {
    title: "PTSD / Trauma Support (Offline Stub)",
    disclaimer: "THIS IS NOT THERAPY OR PROFESSIONAL HELP. If you are in crisis or flashback call a helpline or emergency services immediately. Rathor is NOT a trauma specialist. Seek trained professional support (EMDR, CPT, trauma-informed therapy).",
    templates: [
      { name: "Flashback / Dissociation Grounding", content: `• Right here, right now — name 5 things you see in the room\n• Touch something solid (chair, floor, wall)\n• Say aloud: "I am [name], I am in [place], it is [year] — the trauma is over"\n• Cold sensation: hold ice cube, splash cold water on face/wrists\n• Slow breathing: in 4, hold 4, out 6–8\n• Safe anchor: look at a photo/object that represents safety` },
      { name: "Nightmare / Sleep Disturbance", content: `• After waking: turn on light, name 3 things you see, touch something real\n• Ground: 5-4-3-2-1 technique\n• Re-script: write down nightmare, then rewrite ending where you are safe\n• Sleep hygiene: consistent bedtime, no screens 1h before, relaxation audio\n• Helpline if recurrent: trauma support lines (befrienders.org, 988 US, etc.)` },
      { name: "Hypervigilance / Trigger Response", content: `• Notice body: scan for tension, clenched jaw, racing heart\n• Ground: feel feet on floor, press palms together, name surroundings\n• Self-talk: "This is a trigger, not real danger — I am safe now"\n• Exit if possible: step outside, change room\n• After: rest, hydrate, journal what triggered\n• Long-term: consider trauma therapy (EMDR, somatic experiencing)` },
      { name: "Shame / Self-Blame Reprocessing", content: `• Reminder: the event was not your fault — responsibility lies with perpetrator\n• Self-compassion: "I did the best I could with what I knew then"\n• Write letter to younger self: compassion, protection, love\n• Challenge thoughts: "Would I blame a friend in this situation?"\n• Reach out: safe person or hotline — shame thrives in silence` },
      { name: "Emotional Numbness / Avoidance", content: `• Gentle re-connection: notice body sensations without judgment\n• Small exposure: listen to safe music, look at old photos with support\n• Self-care basics: movement, nature, warm bath\n• Reminder: numbness is a survival response — it's okay to feel nothing sometimes\n• When ready: talk to trauma-informed therapist — avoidance can keep pain frozen` },
      { name: "Reclaiming Safety & Trust", content: `• Create safety anchors: safe place visualization, comfort object, trusted person\n• Small trust exercises: share one small thing with safe person\n• Boundaries: practice saying "no" in low-stakes situations\n• Self-validation: "My feelings are real and valid"\n• Long-term: trauma therapy helps rebuild trust in self & others` }
    ]
  },

  cptsd: {
    title: "Complex PTSD (C-PTSD) Support (Offline Stub)",
    disclaimer: "THIS IS NOT THERAPY OR PROFESSIONAL HELP. Complex PTSD from prolonged/repeated trauma requires specialist support (trauma-informed therapy, EMDR, IFS, somatic experiencing). Rathor is NOT a therapist. If in crisis call a helpline or emergency services immediately.",
    templates: [
      {
        name: "Emotional Flashbacks",
        content: `• Recognize: intense shame, fear, abandonment feelings without clear present trigger\n• Ground: name 5 things you see right now, touch something solid\n• Self-talk: "This is an emotional flashback — the danger is in the past"\n• Comfort younger self: imagine hugging child-you, say "You're safe now, I'm here"\n• After: rest, hydrate, journal feelings without judgment`
      },
      {
        name: "Toxic Shame / Inner Critic",
        content: `• Name it: "This is the voice of past abuse, not truth"\n• Challenge: "Would I say this to someone I love?"\n• Self-compassion break: hand on heart, "This hurts. May I be kind to myself"\n• Affirmation: "I am worthy of love and respect exactly as I am"\n• Long-term: inner-child work with therapist`
      },
      {
        name: "Toxic Family / Relational Trauma",
        content: `• Boundaries: practice "gray rock" or low-contact if needed\n• Reminder: "Their behavior was about them, not my worth"\n• Grieve the family you deserved but didn't get\n• Build chosen family: safe friends, support groups\n• Therapy: trauma-informed, attachment-focused approaches`
      },
      {
        name: "Developmental / Childhood Trauma",
        content: `• Acknowledge: needs that were never met (safety, attunement, protection)\n• Reparent yourself: give inner child what they missed (comfort, play, boundaries)\n• Small acts: warm drink, blanket, gentle music\n• Self-compassion: "I was a child. It wasn't my fault"\n• Professional help: inner-child work, somatic experiencing`
      },
      {
        name: "Dissociation / Freeze Response",
        content: `• Notice: feeling numb, foggy, outside body\n• Ground: press feet into floor, squeeze hands, name surroundings\n• Sensory: strong taste/smell (mint, lemon, coffee)\n• Movement: gentle rocking, stretching, walk\n• Reminder: "I am safe now. The freeze helped me survive — I can come back"`
      },
      {
        name: "Reclaiming Identity & Trust",
        content: `• Small trust experiments: share one feeling with safe person\n• Identity work: list values that are yours (not from abuser)\n• Joy practice: 5-minute activity you loved as child\n• Self-validation: "My experience is real. My feelings are valid"\n• Long-term: trauma therapy to rebuild self-trust`
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
// Voice Command Processor — expanded with C-PTSD
// ────────────────────────────────────────────────

async function processVoiceCommand(raw) {
  let cmd = raw.toLowerCase().trim();

  // C-PTSD / Complex Trauma triggers
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

  // Mental health general
  if (cmd.includes('mental health') || cmd.includes('mental help') || cmd.includes('feeling down') || cmd.includes('depressed') || cmd.includes('anxiety') || cmd.includes('panic attack')) {
    triggerEmergencyAssistant('mental');
    return true;
  }

  if (cmd.includes('suicidal') || cmd.includes('want to die') || cmd.includes('end it') || cmd.includes('not worth it')) {
    triggerEmergencyAssistant('mental');
    speak("You are enough. This pain is temporary. Stay with me. Help is here right now.");
    return true;
  }

  // ... other existing emergency/medical/legal/crisis triggers ...

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
