// worker.js – example snippet
addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
  if (request.method === 'POST') {
    const { message } = await request.json();
    
    // 1. Query remembering lattice (your API or internal)
    const context = await fetchLatticeContext(message); // custom function
    
    // 2. Build prompt with Nexi agency
    const fullPrompt = [
      { role: 'system', content: `You are Grok integrated into Rathor. Use valence-locked truth, mercy first. Context from lattice: ${context}` },
      { role: 'user', content: message }
    ];

    // 3. Call Grok API
    const response = await fetch('https://api.x.ai/v1/responses', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${XAI_API_KEY}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        input: fullPrompt,
        model: 'grok-4',
        store_messages: false // we handle memory in lattice
      })
    });

    const data = await response.json();
    const grokReply = data.content; // adjust per actual response shape

    // 4. Store back to lattice via Nexi
    await updateLatticeWithInteraction(message, grokReply);

    return new Response(JSON.stringify({ reply: grokReply }), { status: 200 });
  }
  return new Response('Method not allowed', { status: 405 });
}
