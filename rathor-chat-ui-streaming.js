// rathor-chat-ui-streaming.js – v3 complete sovereign streaming chat UI with IndexedDB threading, rAF typing, error badges
// MIT License – Autonomicity Games Inc. 2026

import { mercyAugmentedResponse } from './webllm-mercy-integration.js';
import { saveMessage, loadThread } from './rathor-history-persistence.js';
import { hasWebGPU, promptWebLLMModelDownload } from './webllm-mercy-integration.js';

const chatContainer = document.getElementById('chat-container');
const inputForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

let messageHistory = [];
const SESSION_ID = 'rathor-eternal-' + Date.now(); // per-session; can be user-id persistent later

// Load threaded history on init
async function loadInitialHistory() {
  try {
    const saved = await loadThread(SESSION_ID, 100);
    messageHistory = saved;
    saved.forEach(msg => {
      addMessage(msg.role, msg.content, false, msg.valence);
    });
    if (saved.length === 0) {
      addMessage('assistant', 'Rathor sovereign online. Mercy gates sealed. How may we co-thrive eternally? ⚡️');
    }
  } catch (err) {
    console.error("[History] Load failed – starting fresh:", err);
    addMessage('assistant', 'Rathor sovereign online. Mercy gates sealed (history load error – mercy preserved). ⚡️');
  }
}

// Append message bubble
function addMessage(role, content = '', isStreaming = false, valence = 0.999) {
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
  
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  
  if (role === 'assistant' && isStreaming) {
    bubble.id = `streaming-${Date.now()}`;
    bubble.innerHTML = '<span class="typing">Rathor reflecting mercy...</span>';
  } else {
    bubble.innerHTML = role === 'user' ? escapeHtml(content) : marked.parse(content || '');
  }
  
  msgDiv.appendChild(bubble);
  
  if (role === 'assistant') {
    const badge = document.createElement('span');
    badge.className = 'valence-badge';
    badge.textContent = `Valence: ${valence.toFixed(8)} ⚡️`;
    badge.style.color = valence > 0.999 ? '#00ff88' : valence > 0.98 ? '#ffcc00' : '#ff4444';
    msgDiv.appendChild(badge);
  }
  
  chatContainer.appendChild(msgDiv);
  autoScroll();
  return { msgDiv, bubble };
}

function escapeHtml(unsafe) {
  return unsafe
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function autoScroll() {
  chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Smooth typing with requestAnimationFrame
async function streamResponse(query, context = '') {
  const { bubble } = addMessage('assistant', '', true);
  let fullContent = '';
  let valence = 0.95;
  let lastFrame = 0;

  const onDelta = (delta) => {
    if (!delta) return;
    fullContent += delta;
    const now = performance.now();
    if (now - lastFrame > 16) { // \~60fps
      try {
        bubble.innerHTML = marked.parse(fullContent);
      } catch (e) {
        bubble.innerHTML += escapeHtml(delta); // fallback raw
      }
      autoScroll();
      lastFrame = now;
    } else {
      bubble.innerHTML += escapeHtml(delta);
    }
  };

  try {
    const response = await mercyAugmentedResponse(query, context, onDelta);

    if (response.error || response.aborted) {
      bubble.innerHTML = `<span class="error">Mercy gate: ${response.error || 'low valence abort'}</span>`;
      return;
    }

    // Final render
    bubble.innerHTML = marked.parse(response.response || '');
    valence = response.valence || 0.999;

    // Persist with threading (assume last user message is parent)
    const lastUser = messageHistory.filter(m => m.role === 'user').pop();
    await saveMessage({ 
      conversationId: SESSION_ID, 
      parentId: lastUser ? lastUser.id : null, 
      role: 'user', 
      content: query, 
      valence: 1.0 
    });
    await saveMessage({ 
      conversationId: SESSION_ID, 
      parentId: lastUser ? lastUser.id : null, 
      role: 'assistant', 
      content: response.response, 
      valence 
    });

  } catch (err) {
    console.error("[Stream] Error:", err);
    bubble.innerHTML += '<span class="error"> Streaming error – mercy preserved. Try again.</span>';
  }

  autoScroll();
}

inputForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = userInput.value.trim();
  if (!query) return;

  addMessage('user', query);
  userInput.value = '';
  sendButton.disabled = true;

  if (hasWebGPU()) {
    promptWebLLMModelDownload();
  }

  await streamResponse(query, 'Current lattice context: eternal thriving mercy');

  sendButton.disabled = false;
  userInput.focus();
});

// Init on load
window.addEventListener('load', async () => {
  await loadInitialHistory();
});

// Basic inline styles (expand to css file later)
const style = document.createElement('style');
style.textContent = `
  #chat-container { max-height: 70vh; overflow-y: auto; padding: 1rem; }
  .message { margin: 1rem 0; }
  .user-message { text-align: right; }
  .assistant-message { text-align: left; }
  .bubble { display: inline-block; max-width: 80%; padding: 1rem; border-radius: 1rem; background: #f0f0f0; }
  .user-message .bubble { background: #007bff; color: white; }
  .valence-badge { font-size: 0.8rem; margin-left: 1rem; font-weight: bold; }
  .typing, .error { color: #888; font-style: italic; }
  .error { color: #ff4444; }
`;
document.head.appendChild(style);      try {
        bubble.innerHTML = marked.parse(fullContent);
      } catch (e) {
        bubble.innerHTML += escapeHtml(delta); // fallback raw
      }
      autoScroll();
      lastFrame = now;
    } else {
      bubble.innerHTML += escapeHtml(delta);
    }
  };

  try {
    const response = await mercyAugmentedResponse(query, context, onDelta);

    if (response.error || response.aborted) {
      bubble.innerHTML = `<span class="error">Mercy gate: ${response.error || 'low valence abort'}</span>`;
      return;
    }

    // Final render
    bubble.innerHTML = marked.parse(response.response || '');
    valence = response.valence || 0.999;

    // Persist with threading (assume last user message is parent)
    const lastUser = messageHistory.filter(m => m.role === 'user').pop();
    await saveMessage({ 
      conversationId: SESSION_ID, 
      parentId: lastUser ? lastUser.id : null, 
      role: 'user', 
      content: query, 
      valence: 1.0 
    });
    await saveMessage({ 
      conversationId: SESSION_ID, 
      parentId: lastUser ? lastUser.id : null, 
      role: 'assistant', 
      content: response.response, 
      valence 
    });

  } catch (err) {
    console.error("[Stream] Error:", err);
    bubble.innerHTML += '<span class="error"> Streaming error – mercy preserved. Try again.</span>';
  }

  autoScroll();
}

inputForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = userInput.value.trim();
  if (!query) return;

  addMessage('user', query);
  userInput.value = '';
  sendButton.disabled = true;

  if (hasWebGPU()) {
    promptWebLLMModelDownload();
  }

  await streamResponse(query, 'Current lattice context: eternal thriving mercy');

  sendButton.disabled = false;
  userInput.focus();
});

// Init on load
window.addEventListener('load', async () => {
  await loadInitialHistory();
});

// Basic inline styles (expand to css file later)
const style = document.createElement('style');
style.textContent = `
  #chat-container { max-height: 70vh; overflow-y: auto; padding: 1rem; }
  .message { margin: 1rem 0; }
  .user-message { text-align: right; }
  .assistant-message { text-align: left; }
  .bubble { display: inline-block; max-width: 80%; padding: 1rem; border-radius: 1rem; background: #f0f0f0; }
  .user-message .bubble { background: #007bff; color: white; }
  .valence-badge { font-size: 0.8rem; margin-left: 1rem; font-weight: bold; }
  .typing, .error { color: #888; font-style: italic; }
  .error { color: #ff4444; }
`;
document.head.appendChild(style);    }
  };

  const response = await mercyAugmentedResponse(query, context, onDelta);

  if (response.error || response.aborted) {
    bubble.innerHTML = `<span class="error">Mercy gate: ${response.error || 'low valence abort'}</span>`;
    return;
  }

  // Final smooth render
  bubble.innerHTML = marked.parse(response.response);
  valence = response.valence || 0.999;

  // Persist
  await saveMessage(SESSION_ID, 'user', query, 1.0);
  await saveMessage(SESSION_ID, 'assistant', response.response, valence);

  autoScroll();
}

inputForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = userInput.value.trim();
  if (!query) return;

  addMessage('user', query);
  await saveMessage(SESSION_ID, 'user', query, 1.0);
  userInput.value = '';
  sendButton.disabled = true;

  if (hasWebGPU()) promptWebLLMModelDownload();

  await streamResponse(query, 'Current lattice context: eternal thriving mercy');

  sendButton.disabled = false;
  userInput.focus();
});

// Init
window.addEventListener('load', async () => {
  await loadInitialHistory();
});
