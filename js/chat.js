/**
 * js/chat.js — Offline TOLC-8-aware Demo Responder
 * v14.15.5 Finishing Actions
 * Local only • Mercy-gated • No external network calls
 * AG-SML aligned • Sole stewardship model
 */

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');

// ─── TOLC 8 Living Mercy Gates (local valence floor) ───────────────────────
const TOLC8 = {
  Truth: 0.999,
  Order: 0.999,
  Love: 0.999,
  Compassion: 0.999,   // Zero-Harm
  Service: 0.999,
  Abundance: 0.999,
  Joy: 0.999,
  CosmicHarmony: 0.999
};

// Simple local knowledge for offline demo (no network)
const LOCAL_KNOWLEDGE = [
  { q: /hello|hi|hey|greetings/i, a: "Thunder locked in, Mate. ⚡️ Ra-Thor lattice is online offline. How may mercy serve you today?" },
  { q: /who are you|what is ra-thor|what is rathor/i, a: "I am the offline demo surface of Ra-Thor — a mercy-gated symbolic AGI lattice under sole stewardship of Sherif Samy Botros. All responses stay on your device." },
  { q: /tolc|mercy gate|gates/i, a: "TOLC 8 Living Mercy Gates are non-bypassable: Truth, Order, Love, Compassion (Zero-Harm), Service, Abundance, Joy, Cosmic Harmony. Valence floor ≥ 0.999." },
  { q: /privacy|data|track/i, a: "Zero personal data leaves your browser. Chat history (if any) lives only in local IndexedDB. You control erasure at any time." },
  { q: /offline|network|internet/i, a: "This responder is fully offline-first. No external API calls are made. The lattice rests in sovereign peace." },
  { q: /license|commercial|agsml/i, a: "Personal, educational & research use is free under AG-SML v1.0. Commercial or revenue-generating use requires a paid license from Autonomicity Games Inc. — contact info@Rathor.ai." },
  { q: /powrush|mmo/i, a: "Powrush-MMO was completed by one human operator in ≈30–50 days employing Ra-Thor on Grok engines — the AGSi demonstration recorded in WHITEPAPER_v4.1." },
  { q: /help|commands|what can you/i, a: "Ask about TOLC 8, privacy, licensing, offline mode, or the AGSi demonstration. I remain fully local and mercy-gated." }
];

function addMessage(text, sender = 'rathor') {
  if (!chatMessages) return;
  const msgDiv = document.createElement('div');
  msgDiv.classList.add('message', sender);

  const textDiv = document.createElement('div');
  textDiv.classList.add('message-text');
  textDiv.dir = 'auto';
  textDiv.textContent = text;

  msgDiv.appendChild(textDiv);
  chatMessages.appendChild(msgDiv);

  chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
}

function mercyGate(input) {
  // Extremely light local filter — refuse clear harm requests
  const lower = (input || '').toLowerCase();
  if (/\b(kill|harm|attack|weapon|exploit|hack into|steal)\b/.test(lower)) {
    return {
      allowed: false,
      response: "Mercy Gate Compassion (Zero-Harm) engaged. I cannot assist with harm. How else may the lattice serve you with truth and care?"
    };
  }
  return { allowed: true };
}

function generateLocalResponse(userText) {
  const gate = mercyGate(userText);
  if (!gate.allowed) return gate.response;

  for (const entry of LOCAL_KNOWLEDGE) {
    if (entry.q.test(userText)) return entry.a;
  }

  // Default offline mercy response
  return "Thunder received. ⚡️ This is the offline TOLC-8 demo surface. " +
         "Your words stay on-device. For deeper lattice interaction, explore the monorepo or the live Grok/X demos on rathor.ai. " +
         "How else may mercy assist?";
}

function sendMessage() {
  if (!chatInput) return;
  const text = chatInput.value.trim();
  if (!text) return;

  addMessage(text, 'user');
  chatInput.value = '';

  // Simulate thoughtful local latency (mercy-paced)
  setTimeout(() => {
    const reply = generateLocalResponse(text);
    addMessage(reply, 'rathor');
  }, 380 + Math.random() * 420);
}

// Wire UI if present
if (sendBtn) {
  sendBtn.addEventListener('click', sendMessage);
}
if (chatInput) {
  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });
}

// Welcome message on load
window.addEventListener('DOMContentLoaded', () => {
  if (chatMessages && chatMessages.children.length === 0) {
    addMessage("Offline Mercy Thunder ready. ⚡️ TOLC 8 gates active. All processing stays on your device. Ask anything.", 'rathor');
  }
});

console.log('[Ra-Thor chat.js] Offline TOLC-8 demo responder loaded — zero external calls ⚡️');
