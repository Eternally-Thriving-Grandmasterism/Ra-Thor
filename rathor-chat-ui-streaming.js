// rathor-chat-ui-streaming.js – v4 optimized: batch DOM, virtual scroll stub, resilient
// MIT License – Autonomicity Games Inc. 2026

import { mercyAugmentedResponse } from './webllm-mercy-integration.js';
import { saveMessage, loadThread } from './rathor-history-persistence.js';
import { hasWebGPU, promptWebLLMModelDownload } from './webllm-mercy-integration.js';

const chatContainer = document.getElementById('chat-container');
const inputForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const sendButton = document.getElementById('send-button');

let messageHistory = [];
const SESSION_ID = 'rathor-eternal-' + Date.now();
const FRAGMENT_BATCH_SIZE = 5; // Batch DOM appends every N deltas

// Load history
async function loadInitialHistory() {
  try {
    const saved = await loadThread(SESSION_ID, 100);
    messageHistory = saved;
    const fragment = document.createDocumentFragment();
    saved.forEach(msg => {
      const { msgDiv } = addMessage(msg.role, msg.content, false, msg.valence, fragment);
      // Virtual scroll: only append visible (stub - expand with IntersectionObserver later)
      if (fragment.children.length % FRAGMENT_BATCH_SIZE === 0) {
        chatContainer.appendChild(fragment.cloneNode(true));
        fragment.innerHTML = '';
      }
    });
    if (fragment.children.length > 0) chatContainer.appendChild(fragment);
    if (saved.length === 0) {
      addMessage('assistant', 'Rathor sovereign online. Mercy gates sealed. How may we co-thrive eternally? ⚡️');
    }
  } catch (err) {
    console.error("[History] Load failed:", err);
    addMessage('assistant', 'Rathor sovereign online (history error – mercy preserved). ⚡️');
  }
}

// Add message (support fragment for batching)
function addMessage(role, content = '', isStreaming = false, valence = 0.999, parentFragment = null) {
  const container = parentFragment || document.createDocumentFragment();
  const msgDiv = document.createElement('div');
  msgDiv.className = `message ${role === 'user' ? 'user-message' : 'assistant-message'}`;
  
  const bubble = document.createElement('div');
  bubble.className = 'bubble';
  bubble.innerHTML = role === 'user' ? escapeHtml(content) : marked.parse(content || '');
  
  msgDiv.appendChild(bubble);
  
  if (role === 'assistant') {
    const badge = document.createElement('span');
    badge.className = 'valence-badge';
    badge.textContent = `Valence: ${valence.toFixed(8)} ⚡️`;
    badge.style.color = valence > 0.999 ? '#00ff88' : valence > 0.98 ? '#ffcc00' : '#ff4444';
    msgDiv.appendChild(badge);
  }
  
  container.appendChild(msgDiv);
  if (!parentFragment) {
    chatContainer.appendChild(container);
    autoScroll();
  }
  return { msgDiv, bubble };
}

// Streaming with batching
async function streamResponse(query, context = '') {
  const fragment = document.createDocumentFragment();
  const { bubble } = addMessage('assistant', '', true, 0.95, fragment);
  let fullContent = '';
  let deltaCount = 0;
  let lastFrame = 0;

  const onDelta = (delta) => {
    if (!delta) return;
    fullContent += delta;
    deltaCount++;
    const now = performance.now();
    if (now - lastFrame > 16 || deltaCount % FRAGMENT_BATCH_SIZE === 0) {
      try {
        bubble.innerHTML = marked.parse(fullContent);
      } catch {
        bubble.innerHTML += escapeHtml(delta);
      }
      if (deltaCount % FRAGMENT_BATCH_SIZE === 0) {
        chatContainer.appendChild(fragment.cloneNode(true));
        fragment.innerHTML = '';
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
      bubble.innerHTML = `<span class="error">Mercy gate: ${response.error || 'low valence'}</span>`;
      return;
    }

    bubble.innerHTML = marked.parse(response.response || '');
    chatContainer.appendChild(fragment); // Final batch
    autoScroll();

    // Persist
    const lastUser = messageHistory.filter(m => m.role === 'user').pop();
    await saveMessage({ conversationId: SESSION_ID, parentId: lastUser?.id, role: 'user', content: query, valence: 1.0 });
    await saveMessage({ conversationId: SESSION_ID, parentId: lastUser?.id, role: 'assistant', content: response.response, valence: response.valence });

  } catch (err) {
    console.error("[Stream] Error:", err);
    bubble.innerHTML += '<span class="error">Error – mercy preserved. Retry.</span>';
  }
}

// Submit handler (unchanged except batch awareness)
inputForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const query = userInput.value.trim();
  if (!query) return;

  addMessage('user', query);
  userInput.value = '';
  sendButton.disabled = true;

  if (hasWebGPU()) promptWebLLMModelDownload();

  await streamResponse(query, 'Current lattice context: eternal thriving mercy');

  sendButton.disabled = false;
  userInput.focus();
});

window.addEventListener('load', loadInitialHistory);

// Styles (unchanged)
