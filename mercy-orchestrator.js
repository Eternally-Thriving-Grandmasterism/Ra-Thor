// Append inside MercyOrchestrator class

constructor() {
  // ... existing constructor code ...
  this.voice = new VoiceImmersion(this);
}

async init() {
  // ... existing init code ...
  await this.voice.init();
  // Auto-start voice immersion on page load (uncomment to enable)
  // await this.voice.start();
}

async generateResponse(userInput) {
  // ... existing generation logic ...
  const responseText = /* your generation result */;

  // Voice immersion output (only if active)
  if (this.voice.isActive) {
    await this.voice.speak(responseText, this.lastValence);
  }

  // ... existing post-processing ...
  return responseText;
}

// Optional: toggle voice immersion via command or UI button
async toggleVoiceImmersion() {
  if (this.voice.isActive) {
    this.voice.stop();
    return "Voice immersion paused ⚡️";
  } else {
    await this.voice.start();
    return "Voice immersion active — listening for thunder... ⚡️";
  }
}

// Example: add voice command handling inside orchestrate()
async orchestrate(userInput) {
  const lower = userInput.toLowerCase();

  if (lower.includes("voice on") || lower.includes("start voice") || lower.includes("listen")) {
    return await this.toggleVoiceImmersion();
  }
  if (lower.includes("voice off") || lower.includes("stop voice") || lower.includes("silent")) {
    return await this.toggleVoiceImmersion();
  }

  // ... existing routing ...
}
