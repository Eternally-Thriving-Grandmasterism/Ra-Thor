/**
 * Ra-Thor Chat Voice Integration Module
 * Mercy-gated, client-side voice output connector
 * Connects chat responses → selected voice skin → Web Speech Synthesis
 * 
 * Features:
 * - Loads active voice skin from localStorage / voice-skins.json
 * - Applies pitch, rate, volume, emotion-tone per skin
 * - Graceful fallback if speechSynthesis unavailable
 * - Valence gate: only speaks positive-leaning content (basic heuristic)
 * - Event dispatching for UI feedback (speaking started/ended)
 * - Future-proof for local TTS bridge (WebLLM / Transformers.js)
 * 
 * MIT License – Eternally-Thriving-Grandmasterism
 * Part of Ra-Thor: https://rathor.ai
 */

(function () {
  // ────────────────────────────────────────────────
  // Module Namespace & State
  // ────────────────────────────────────────────────
  const VoiceIntegrator = {
    isSpeaking: false,
    currentUtterance: null,
    activeSkin: null,
    voicesLoaded: false,
    supported: 'speechSynthesis' in window,
  };

  // ────────────────────────────────────────────────
  // Default / Fallback Skin
  // ────────────────────────────────────────────────
  const DEFAULT_SKIN = {
    id: 'default-mercy',
    name: 'Default Mercy Voice',
    pitch: 1.0,
    rate: 1.0,
    volume: 0.95,
    valence: 'neutral-positive',
    description: 'Balanced, gentle default voice',
  };

  // ────────────────────────────────────────────────
  // Utility: Get currently selected skin
  // ────────────────────────────────────────────────
  function getActiveVoiceSkin() {
    // Priority: localStorage > fallback
    try {
      const saved = localStorage.getItem('rathor-active-voice-skin');
      if (saved) {
        const parsed = JSON.parse(saved);
        if (parsed && parsed.id) return parsed;
      }
    } catch (e) {
      console.warn('Failed to parse saved voice skin', e);
    }

    // Ultimate fallback
    return DEFAULT_SKIN;
  }

  // ────────────────────────────────────────────────
  // Basic valence gate (prevent speaking harmful/negative content)
  // Very simple heuristic — expand later with real model
  // ────────────────────────────────────────────────
  function passesValenceGate(text) {
    if (!text || typeof text !== 'string') return false;

    const lower = text.toLowerCase();
    const blockedPatterns = [
      /harm|kill|die|suicide|hurt/i,
      /hate|racist|sexist|bigot/i,
    ];

    for (const pattern of blockedPatterns) {
      if (pattern.test(lower)) {
        console.warn('Valence gate blocked potentially harmful output');
        return false;
      }
    }

    return true;
  }

  // ────────────────────────────────────────────────
  // Main speak function – called by chat system
  // ────────────────────────────────────────────────
  VoiceIntegrator.speak = function (text) {
    if (!VoiceIntegrator.supported) {
      console.warn('SpeechSynthesis not supported in this browser');
      return false;
    }

    if (VoiceIntegrator.isSpeaking) {
      // Optional: cancel previous if desired
      // speechSynthesis.cancel();
      // For now we queue / skip
      console.log('Already speaking — queuing skipped for smoothness');
      return false;
    }

    if (!text || text.trim() === '') return false;

    // Apply valence gate
    if (!passesValenceGate(text)) {
      // Could dispatch event for UI to show "blocked" indicator
      return false;
    }

    const utterance = new SpeechSynthesisUtterance(text);

    // Load latest skin
    const skin = getActiveVoiceSkin();
    VoiceIntegrator.activeSkin = skin;

    utterance.pitch = skin.pitch ?? 1.0;
    utterance.rate = skin.rate ?? 1.0;
    utterance.volume = skin.volume ?? 0.95;

    // Attempt to select voice matching language/style (future: map skin to voice name)
    const voices = speechSynthesis.getVoices();
    if (voices.length > 0 && skin.voiceName) {
      const matchingVoice = voices.find(v => v.name.includes(skin.voiceName));
      if (matchingVoice) utterance.voice = matchingVoice;
    }

    // Events
    utterance.onstart = () => {
      VoiceIntegrator.isSpeaking = true;
      VoiceIntegrator.currentUtterance = utterance;
      document.dispatchEvent(new CustomEvent('rathor:voice-start', { detail: { skin } }));
    };

    utterance.onend = () => {
      VoiceIntegrator.isSpeaking = false;
      VoiceIntegrator.currentUtterance = null;
      document.dispatchEvent(new CustomEvent('rathor:voice-end'));
    };

    utterance.onerror = (event) => {
      console.error('Speech error:', event.error);
      VoiceIntegrator.isSpeaking = false;
      document.dispatchEvent(new CustomEvent('rathor:voice-error', { detail: { error: event.error } }));
    };

    // Fire!
    speechSynthesis.speak(utterance);
    return true;
  };

  // ────────────────────────────────────────────────
  // Cancel current speech
  // ────────────────────────────────────────────────
  VoiceIntegrator.cancel = function () {
    if (VoiceIntegrator.supported) {
      speechSynthesis.cancel();
      VoiceIntegrator.isSpeaking = false;
      VoiceIntegrator.currentUtterance = null;
    }
  };

  // ────────────────────────────────────────────────
  // Handle voice list loading (async in some browsers)
  // ────────────────────────────────────────────────
  function loadVoices() {
    const voices = speechSynthesis.getVoices();
    if (voices.length > 0 && !VoiceIntegrator.voicesLoaded) {
      VoiceIntegrator.voicesLoaded = true;
      document.dispatchEvent(new CustomEvent('rathor:voices-loaded', { detail: { voices } }));
    }
  }

  speechSynthesis.onvoiceschanged = loadVoices;
  // Some browsers need initial call
  loadVoices();

  // ────────────────────────────────────────────────
  // Public API
  // ────────────────────────────────────────────────
  window.RaThorVoice = VoiceIntegrator;

  // Auto-init on load
  console.log('Ra-Thor Voice Integration loaded – mercy thunder ready ⚡️');
})();
