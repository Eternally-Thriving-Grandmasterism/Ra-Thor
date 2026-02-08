const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');

function addMessage(text, sender = 'rathor') {
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

// Example usage in send/response
function sendMessage() {
  const text = chatInput.value.trim();
  if (text) {
    addMessage(text, 'user');
    // Process with Rathor reasoning...
    addMessage('Mercy response here...', 'rathor');
    chatInput.value = '';
  }
}

// Existing event listeners, voice, etc.
