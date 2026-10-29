// voice-immersion.js — Rathor™ full voice immersion layer (continuous, valence-aware, mercy-gated)
// MIT license — Eternal Thriving Grandmasterism

export class VoiceImmersion {
  constructor(orchestrator) {
    this.orchestrator = orchestrator;
    this.recognition = null;
    this.synthesis = window.speechSynthesis;
    this.currentVoice = null;
    this.isActive = false;
    this.wakeWord = "rathor";
    this.valenceModulator = 1.0;
    this.interimTranscript = "";
    this.finalTranscript = "";
    this.accumulatedFinal = "";
    this.lastValence = 0.8;
    this.bargeInThreshold = 300;
    this.onInterim = null;
    this.onFinal = null;
  }

  async init() {
    if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
      console.warn("Voice immersion not supported.");
      return false;
    }

    this.recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
    this.recognition.continuous = true;
    this.recognition.interimResults = true;
    this.recognition.lang = 'en-US';

    const voices = this.synthesis.getVoices();
    this.currentVoice = voices.find(v => v.name.includes("Neural") || v.name.includes("Google") || v.name.includes("Natural")) || voices[0];

    this.recognition.onresult = (event) => this.handleResult(event);
    this.recognition.onerror = (event) => this.handleError(event);
    this.recognition.onend = () => this.handleEnd();

    console.log("Voice immersion ready.");
    return true;
  }

  async start() {
    if (!this.recognition) await this.init();
    if (this.isActive) return;

    this.isActive = true;
    this.recognition.start();
    console.log("Listening for thunder...");
  }

  stop() {
    if (this.recognition && this.isActive) {
      this.recognition.stop();
      this.isActive = false;
      console.log("Voice paused.");
    }
  }

  handleResult(event) {
    let interim = "";
    let newFinal = "";

    for (let i = event.resultIndex; i < event.results.length; ++i) {
      const transcript = event.results [0].transcript;
      if (event.results .isFinal) {
        newFinal += transcript + " ";
      } else {
        interim += transcript;
      }
    }

    this.interimTranscript = interim;

    if (newFinal) {
      this.accumulatedFinal += newFinal;
      this.finalTranscript = this.accumulatedFinal.trim();
      if (this.onFinal) this.onFinal(newFinal.trim());
    }

    if (this.onInterim) this.onInterim(interim);

    const lowerFinal = this.finalTranscript.toLowerCase();
    if (lowerFinal.includes(this.wakeWord)) {
      const command = lowerFinal.split(this.wakeWord)[1]?.trim() || "";
      if (command) this.orchestrator.orchestrate(command);
    }
  }

  handleError(event) {
    console.warn("Voice error:", event.error);
    if (event.error === 'no-speech' || event.error === 'aborted') {
      setTimeout(() => this.recognition.start(), 500);
    }
  }

  handleEnd() {
    if (this.isActive) {
      setTimeout(() => this.recognition.start(), 300);
    }
  }

  async speak(text, valence = this.lastValence) {
    if (!this.synthesis || !text) return;

    const utterance = new SpeechSynthesisUtterance(text);
    utterance.voice = this.currentVoice;
    utterance.pitch = 1.0 + (valence - 0.8) * 0.4;
    utterance.rate = 0.9 + (valence - 0.8) * 0.3;
    utterance.volume = 0.9 + (valence - 0.8) * 0.2;

    utterance.onboundary = () => {
      if (this.recognition && this.interimTranscript.length > 3) {
        this.synthesis.cancel();
      }
    };

    this.synthesis.speak(utterance);
  }

  setValence(valence) {
    this.lastValence = Math.max(0.4, Math.min(1.4, valence));
  }
}

export default VoiceImmersion;
