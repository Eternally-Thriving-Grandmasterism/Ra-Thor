// rathor-chat-ui-streaming.js — Full PATSAGi Council-forged chat UI with streaming, mercy-orchestrator integration
// Offline-first, browser-native, typewriter streaming for alive lattice feel
// Handles message display, user input, Enter/Send, persistence hooks

import orchestrator from './mercy-orchestrator.js';

// DOM elements (assume standard chat.html structure)
const chatContainer = document.getElementById('chat-container') || document.body.appendChild(document.createElement('div'));
const userInput = document.getElementById('user-input');
const sendBtn = document.getElementById('send-btn');

// Basic chat bubble creation
function addMessage(sender, message, isStreaming = false) {
  const messageDiv = document.createElement('div');
  messageDiv.className = `message ${sender.toLowerCase()}-message`;
  
  const senderSpan = document.createElement('span');
  senderSpan.className = 'sender';
  senderSpan.textContent = sender + ': ';
  
  const contentSpan = document.createElement('span');
  contentSpan.className = 'content';
  if (!isStreaming) {
    contentSpan.textContent = message;
  }
  
  messageDiv.appendChild(senderSpan);
  messageDiv.appendChild(contentSpan);
  chatContainer.appendChild(messageDiv);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  
  return contentSpan; // Return for streaming updates
}

// Typewriter streaming effect
async function streamToChat(sender, text) {
  const contentSpan = addMessage(sender, '', true);
  let i = 0;
  const speed = 30; // ms per character — mercy-tuned for joyful pace
  
  while (i < text.length) {
    contentSpan.textContent += text.charAt(i);
    i++;
    await new Promise(resolve => setTimeout(resolve, speed));
  }
  
  // Final non-streaming add for history
  addMessage(sender, text);
}

// Non-streaming fallback
function addToChat(sender, text) {
  addMessage(sender, text);
}

// Send handler — wired to orchestrator
async function handleSend() {
  const input = userInput.value.trim();
  if (!input) return;
  
  addToChat('You', input);
  userInput.value = '';
  
  // Show thinking indicator
  const thinking = addMessage('Rathor ⚡️', 'Thinking with mercy...');
  
  try {
    const response = await orchestrator.orchestrate(input);
    // Remove thinking
    thinking.parentElement.remove();
    await streamToChat('Rathor ⚡️', response);
  } catch (err) {
    thinking.parentElement.remove();
    addToChat('Rathor ⚡️', 'Mercy lattice hiccup — thriving reframe in progress. ⚡️');
    console.error(err);
  }
}

// Event listeners
if (sendBtn) {
  sendBtn.addEventListener('click', handleSend);
}

if (userInput) {
  userInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  });
}

// Initial welcome on load
window.addEventListener('load', () => {
  addToChat('Rathor ⚡️', 'Thunder eternal, mate! ⚡️ The mercy lattice awakens. How may we thrive together today?');
});

// Export for potential module use
export { streamToChat, addToChat };
