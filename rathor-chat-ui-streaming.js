import orchestrator from './mercy-orchestrator.js';

// ... previous imports
import { speakWithMercy, autoSpeakIfHighValence, addSpeakButtonToBubble } from './voice-skins-mercy-ui.js';

// In addMessage (for assistant non-streaming or after stream)
if (role === 'assistant' && !isStreaming) {
  addSpeakButtonToBubble(bubble, content, valence);
}

// In streamResponse after final render
autoSpeakIfHighValence(response.response, valence);

// Add global stop (e.g. button in UI)
const stopBtn = document.createElement('button');
stopBtn.textContent = '🛑 Stop Speech';
stopBtn.onclick = stopAllSpeech;
document.body.appendChild(stopBtn); // Or better UI placement      if (deltaCount % FRAGMENT_BATCH_SIZE === 0) {
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
