// src/integrations/llm/ToolCallingRouter.ts – Online Tool Calling Router v2
// Real API endpoints + offline mock simulation, valence-gated confidence, mercy enforcement
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';
import RAGMemory from './RAGMemory';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_TOOL_CONFIDENCE_PIVOT = 0.9;
const API_BASE = '/api/tools';

// Real tool endpoints (server proxy routes)
const REAL_TOOLS = {
  web_search: async (args: { query: string }) => {
    const res = await fetch(`${API_BASE}/web_search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(args)
    });
    if (!res.ok) throw new Error('Web search failed');
    return res.json();
  },

  x_keyword_search: async (args: { query: string; limit?: number }) => {
    const res = await fetch(`${API_BASE}/x_keyword_search`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(args)
    });
    if (!res.ok) throw new Error('X search failed');
    return res.json();
  },

  search_images: async (args: { description: string; number_of_images?: number }) => {
    const res = await fetch(`${API_BASE}/search_images`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(args)
    });
    if (!res.ok) throw new Error('Image search failed');
    return res.json();
  },

  code_execution: async (args: { code: string }) => {
    const res = await fetch(`${API_BASE}/code_execution`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(args)
    });
    if (!res.ok) throw new Error('Code execution failed');
    return res.json();
  },

  browse_page: async (args: { url: string; instructions?: string }) => {
    const res = await fetch(`${API_BASE}/browse_page`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(args)
    });
    if (!res.ok) throw new Error('Browse page failed');
    return res.json();
  }
};

// Offline mock tools (same shape as real responses)
const mockTools = {
  web_search: async ({ query }: { query: string }) => ({
    results: [
      { title: `Offline simulation: "${query}"`, snippet: `Top results would include current knowledge about ${query}.` },
      { title: 'Local knowledge note', snippet: 'No internet – using cached + reasoning.' }
    ]
  }),

  x_keyword_search: async ({ query }: { query: string }) => ({
    posts: [
      { id: 'mock1', text: `Simulated X post about ${query} – high relevance offline.` },
      { id: 'mock2', text: 'Another mock post for continuity.' }
    ]
  }),

  search_images: async ({ description }: { description: string }) => ({
    images: [
      { url: `https://via.placeholder.com/512?text=Mock+for+${encodeURIComponent(description)}`, description }
    ]
  }),

  code_execution: async ({ code }: { code: string }) => ({
    output: `// Offline sandbox simulation\n${code}\n// Result: safe mock output`
  }),

  browse_page: async ({ url }: { url: string }) => ({
    content: `Offline mock browse: Summary of ${url} based on last known cache.`
  })
};

export class ToolCallingRouter {
  static async callTool(toolName: keyof typeof REAL_TOOLS, args: any): Promise<any> {
    const actionName = `Call tool: ${toolName}`;
    if (!await mercyGate(actionName)) {
      return { error: 'Mercy gate blocked tool execution' };
    }

    const valence = currentValence.get();
    const isOnline = navigator.onLine;

    // High valence + online → prefer real tool
    if (isOnline && valence > VALENCE_TOOL_CONFIDENCE_PIVOT) {
      try {
        const realFn = REAL_TOOLS[toolName];
        if (realFn) {
          const result = await realFn(args);
          mercyHaptic.playPattern('cosmicHarmony', valence);
          return result;
        }
      } catch (e) {
        console.warn(`[ToolRouter] Real ${toolName} failed`, e);
      }
    }

    // Offline / low valence / real failed → mock + RAG enrichment
    const mockFn = mockTools[toolName];
    if (mockFn) {
      const mockResult = await mockFn(args);

      // Enrich with local RAG if query-like args exist
      const query = args.query || args.description || args.code || args.url || '';
      if (query) {
        const ragContext = await RAGMemory.getRelevantContext(query, 800);
        if (ragContext) {
          mockResult.localContext = ragContext;
        }
      }

      mercyHaptic.playPattern('neutralPulse', valence);
      return mockResult;
    }

    return { error: `Tool ${toolName} not implemented` };
  }

  static async processWithTools(prompt: string): Promise<string | null> {
    // Simple tool detection (expand with real LLM function calling later)
    const lower = prompt.toLowerCase();

    if (lower.includes('search') || lower.includes('find') || lower.includes('look up')) {
      return await this.callTool('web_search', { query: prompt });
    }

    if (lower.includes('image') || lower.includes('picture') || lower.includes('visual')) {
      return await this.callTool('search_images', { description: prompt });
    }

    if (lower.includes('code') || lower.includes('execute') || lower.includes('run')) {
      return await this.callTool('code_execution', { code: prompt });
    }

    // No tool detected → null (handled by direct LLM)
    return null;
  }
}

export default ToolCallingRouter;      return mockResult;
    }

    return { error: `Tool ${toolName} not implemented` };
  }

  static async processWithTools(prompt: string): Promise<string> {
    // Simple tool-calling loop (expand with real LLM function calling later)
    const valence = currentValence.get();

    // Example: detect if tool needed
    if (prompt.toLowerCase().includes('search') || prompt.toLowerCase().includes('image')) {
      const toolName = prompt.toLowerCase().includes('image') ? 'search_images' : 'web_search';
      const args = { query: prompt };
      const toolResult = await this.callTool(toolName, args);

      return `Tool result (\( {toolName}):\n \){JSON.stringify(toolResult, null, 2)}\n\nRathor reflects: ${prompt}`;
    }

    // No tool → direct response (handled by WebLLMEngine)
    return null;
  }
}

export default ToolCallingRouter;
