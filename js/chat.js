// js/chat.js — Chat lattice core

// ... all chat variables, recognition init, processVoiceCommand, session CRUD, offline recording, TTS speak(), etc. ...

window.addEventListener('load', async () => {
  await initI18n(); // from common.js
  updateContent();
  initLanguageSearch(await loadLanguageRegistry());
  // ... rest of chat init ...
});
