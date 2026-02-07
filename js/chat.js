// js/chat.js — Rathor Lattice Core with Periodic RTT Probes

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

// Connectivity state
let isOffline = false;
let isHighLatency = false;
let rttHistory = []; // last 5 successful RTTs (ms)
const RTT_PROBE_INTERVAL = 10000; // 10 seconds
const HIGH_LATENCY_RTT_THRESHOLD = 150; // ms
const HIGH_LATENCY_DOWNLINK_THRESHOLD = 10; // Mbps

await rathorDB.open();
await refreshSessionList();
await loadChatHistory();
updateTranslationStats();
await updateTagFrequency();

// ... existing event listeners (voice, record, send, translate, search) ...

// ────────────────────────────────────────────────
// Periodic RTT Probes + Connectivity Awareness
// ────────────────────────────────────────────────

async function probeRTT() {
  if (document.hidden) return; // don't probe when tab hidden

  const start = performance.now();
  try {
    await fetch('/ping?t=' + Date.now(), { cache: 'no-store', mode: 'no-cors' });
    const end = performance.now();
    const rtt = end - start;

    rttHistory.push(rtt);
    if (rttHistory.length > 5) rttHistory.shift();

    const median = [...rttHistory].sort((a,b)=>a-b)[Math.floor(rttHistory.length/2)];
    isHighLatency = median > HIGH_LATENCY_RTT_THRESHOLD;

    if (navigator.connection) {
      isOffline = navigator.connection.type === 'none' || navigator.connection.rtt > 10000;
      isHighLatency = isHighLatency || navigator.connection.downlinkMax < HIGH_LATENCY_DOWNLINK_THRESHOLD;
    }

    updateConnectivityUI();
  } catch (e) {
    isOffline = true;
    updateConnectivityUI();
  }
}

function updateConnectivityUI() {
  if (isOffline) {
    showToast('Offline mode — queued actions will sync later ⚡️');
  } else if (isHighLatency) {
    showToast('High latency (Starlink?) — batching & compressing ⚡️');
  } else {
    showToast('Strong connection — lattice fully online ⚡️');
  }
}

// Start periodic probe
let probeInterval;
function startProbes() {
  probeInterval = setInterval(probeRTT, RTT_PROBE_INTERVAL);
  probeRTT(); // immediate first probe
}

function stopProbes() {
  if (probeInterval) clearInterval(probeInterval);
}

// Visibility handling
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopProbes();
  } else {
    startProbes();
  }
});

// Initial start
startProbes();

// Queue voice note if offline/high-latency
async function startVoiceRecording(sessionId, isEmergency = false) {
  // ... existing recording logic ...
  if (isOffline || isHighLatency) {
    await rathorDB.saveQueuedAction('voice-note', { sessionId, blob, timestamp, isEmergency });
    showToast('Voice note queued for reconnection ⚡️');
  } else {
    await rathorDB.saveVoiceNote(sessionId, blob, timestamp, isEmergency);
  }
}

// ... rest of chat.js functions (sendMessage, speak, recognition, emergency assistants, session search with tags, import/export, etc.) remain as previously expanded ...
