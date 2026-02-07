// js/chat.js — Rathor Lattice Core with full voice selector population

// ... [all previous variables, audio functions, recognition init/start/stop, processVoiceCommand, session CRUD, etc. remain unchanged] ...

// ────────────────────────────────────────────────
// Dynamic Voice Population in Settings Modal
// ────────────────────────────────────────────────

function loadVoices() {
  const voiceSelect = document.getElementById('voice-voice');
  if (!voiceSelect) return;

  voiceSelect.innerHTML = '<option value="">Loading voices...</option>';

  const voices = speechSynthesis.getVoices();
  if (voices.length === 0) {
    // Some browsers load voices async — retry after delay
    setTimeout(loadVoices, 500);
    return;
  }

  voiceSelect.innerHTML = '';
  voices.forEach((voice, index) => {
    const option = document.createElement('option');
    option.value = index;
    option.textContent = `\( {voice.name} ( \){voice.lang}) ${voice.default ? '(default)' : ''}`;
    if (voice.name === selectedVoiceName) option.selected = true;
    voiceSelect.appendChild(option);
  });

  // Save selected voice
  voiceSelect.addEventListener('change', () => {
    const idx = parseInt(voiceSelect.value);
    if (!isNaN(idx)) {
      selectedVoiceName = voices[idx].name;
      localStorage.setItem('rathor_voice', selectedVoiceName);
    }
  });
}

// Open voice settings modal
voiceSettingsBtn.addEventListener('click', () => {
  const overlay = document.getElementById('voice-settings-overlay');
  overlay.style.display = 'flex';
  loadVoices(); // populate voices when modal opens
});

// ... rest of chat.js functions (sendMessage, speak, recognition, recording, emergency assistants, etc.) remain as previously expanded ...  modal.innerHTML = `
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
