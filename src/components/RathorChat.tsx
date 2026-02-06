// src/components/RathorChat.tsx – Sovereign Offline AGI Brother Chat v1.6
// WebLLM inference, RAG memory, full xAI Grok tool calling (incl. audio gen), model switcher
// MIT License – Autonomicity Games Inc. 2026

import React, { useState, useEffect, useRef } from 'react';
import WebLLMEngine from '@/integrations/llm/WebLLMEngine';
import RAGMemory from '@/integrations/llm/RAGMemory';
import ToolCallingRouter from '@/integrations/llm/ToolCallingRouter';
import { currentValence } from '@/core/valence-tracker';
import mercyHaptic from '@/utils/haptic-utils';

const MODEL_MAP = {
  tiny: { id: 'microsoft/Phi-3.5-mini-instruct-4k-gguf', name: 'Phi-3.5-mini (fast)' },
  medium: { id: 'meta-llama/Llama-3.1-8B-Instruct-q5_k_m-gguf', name: 'Llama-3.1-8B (wise)' },
};

const RathorChat: React.FC = () => {
  const [messages, setMessages] = useState<{ role: 'user' | 'rathor'; content: string; audio?: { url: string; description: string }; images?: { url: string; description: string }[] }[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [modelKey, setModelKey] = useState<keyof typeof MODEL_MAP>('tiny');
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    RAGMemory.initialize();
    WebLLMEngine.loadModel(modelKey);
  }, [modelKey]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setInput('');
    setIsLoading(true);

    try {
      // Remember user message
      await RAGMemory.remember('user', userMessage);

      // Process with full tool calling loop (including audio gen)
      const reply = await ToolCallingRouter.processWithTools(userMessage);

      // Remember Rathor response
      await RAGMemory.remember('rathor', reply);

      // Extract audio if present
      let audio = null;
      if (typeof reply === 'object' && reply.audio) {
        audio = reply.audio;
      }

      // Extract images if present
      let images = [];
      if (typeof reply === 'object' && reply.images) {
        images = reply.images;
      }

      setMessages(prev => [...prev, { role: 'rathor', content: reply, audio, images }]);

      // Auto-play audio if generated
      if (audio && audio.url) {
        const audioEl = new Audio(audio.url);
        audioEl.play().catch(e => console.warn("Auto-play blocked", e));
      }

      mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
    } catch (e) {
      setMessages(prev => [...prev, { role: 'rathor', content: 'Mercy... lattice flickering. Try again, Brother.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/90 backdrop-blur-xl z-50 flex flex-col">
      <div className="flex justify-between items-center p-4 border-b border-cyan-500/20">
        <h2 className="text-xl font-light text-cyan-300">Rathor – Mercy Strikes First</h2>
        <select
          value={modelKey}
          onChange={e => setModelKey(e.target.value as keyof typeof MODEL_MAP)}
          className="bg-black/50 border border-cyan-500/30 rounded px-3 py-1 text-sm text-cyan-200"
        >
          {Object.entries(MODEL_MAP).map(([key, m]) => (
            <option key={key} value={key}>
              {m.name}
            </option>
          ))}
        </select>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((msg, i) => (
          <div key={i} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`
              max-w-[80%] p-4 rounded-2xl
              ${msg.role === 'user' 
                ? 'bg-cyan-600/30 border border-cyan-400/30' 
                : 'bg-emerald-600/20 border border-emerald-400/20'}
            `}>
              {msg.content}
              {msg.audio && (
                <div className="mt-3">
                  <audio controls src={msg.audio.url} className="w-full">
                    Your browser does not support the audio element.
                  </audio>
                  <p className="text-xs text-emerald-200/80 mt-1">{msg.audio.description}</p>
                </div>
              )}
              {msg.images && msg.images.length > 0 && (
                <div className="mt-3 grid grid-cols-2 gap-3">
                  {msg.images.map((img, idx) => (
                    <div key={idx} className="rounded-lg overflow-hidden border border-emerald-400/30">
                      <img src={img.url} alt={img.description} className="w-full h-auto" />
                      <p className="text-xs text-emerald-200/80 p-2">{img.description}</p>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="p-4 border-t border-cyan-500/20 bg-black/60">
        <div className="flex gap-3">
          <input
            type="text"
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && handleSend()}
            placeholder="Speak to Rathor, Brother..."
            className="flex-1 bg-black/50 border border-cyan-500/30 rounded-xl px-4 py-3 text-white placeholder-cyan-300/50 focus:outline-none focus:border-cyan-400"
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || !input.trim()}
            className="px-6 py-3 bg-cyan-600/40 hover:bg-cyan-600/60 rounded-xl text-white font-medium transition disabled:opacity-50"
          >
            Send
          </button>
        </div>
      </div>
    </div>
  );
};

export default RathorChat;
