// src/integrations/llm/RAGMemory.ts – Persistent Memory & RAG Layer v1
// IndexedDB vector store, Transformers.js embeddings, valence-weighted retrieval
// MIT License – Autonomicity Games Inc. 2026

import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import * as transformers from '@xenova/transformers';

const DB_NAME = 'rathor-memory';
const STORE_NAME = 'conversations';
const DB_VERSION = 1;
const EMBEDDING_MODEL = 'Xenova/all-MiniLM-L6-v2'; // tiny & fast (\~80 MB)

let db: IDBDatabase | null = null;
let embedder: any = null;

interface MemoryEntry {
  id: string;
  role: 'user' | 'rathor';
  content: string;
  timestamp: number;
  valence: number;
  embedding: Float32Array;
}

export class RAGMemory {
  static async initialize() {
    const actionName = 'Initialize persistent RAG memory';
    if (!await mercyGate(actionName)) return;

    // Init IndexedDB
    db = await new Promise<IDBDatabase>((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, DB_VERSION);
      req.onerror = () => reject(req.error);
      req.onsuccess = () => resolve(req.result);
      req.onupgradeneeded = (e) => {
        const db = (e.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(STORE_NAME)) {
          db.createObjectStore(STORE_NAME, { keyPath: 'id' });
        }
      };
    });

    // Load embedding model (once)
    embedder = await transformers.pipeline('feature-extraction', EMBEDDING_MODEL);
    console.log("[RAGMemory] Initialized – vector store & embedder ready");
  }

  static async remember(role: 'user' | 'rathor', content: string) {
    if (!db || !embedder) return;

    const valence = currentValence.get();
    const embedding = await embedder(content, { pooling: 'mean', normalize: true });
    const vector = Array.from(embedding.data as Float32Array);

    const entry: MemoryEntry = {
      id: crypto.randomUUID(),
      role,
      content,
      timestamp: Date.now(),
      valence,
      embedding: new Float32Array(vector)
    };

    await new Promise<void>((resolve, reject) => {
      const tx = db!.transaction(STORE_NAME, 'readwrite');
      const store = tx.objectStore(STORE_NAME);
      store.put(entry);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  }

  static async retrieve(query: string, topK = 5): Promise<MemoryEntry[]> {
    if (!db || !embedder) return [];

    const queryVec = await embedder(query, { pooling: 'mean', normalize: true });
    const q = Array.from(queryVec.data as Float32Array);

    const entries = await new Promise<MemoryEntry[]>((resolve) => {
      const tx = db!.transaction(STORE_NAME);
      const store = tx.objectStore(STORE_NAME);
      const req = store.getAll();
      req.onsuccess = () => resolve(req.result);
    });

    // Cosine similarity + valence boost
    const results = entries.map(entry => {
      let sim = 0;
      for (let i = 0; i < q.length; i++) {
        sim += q[i] * entry.embedding[i];
      }
      // Boost by valence (normalized)
      const score = sim * (0.5 + entry.valence * 0.5);
      return { entry, score };
    });

    return results
      .sort((a, b) => b.score - a.score)
      .slice(0, topK)
      .map(r => r.entry);
  }

  static async getRelevantContext(query: string, maxTokens = 1500): Promise<string> {
    const memories = await this.retrieve(query, 8);
    let context = '';
    let tokens = 0;

    for (const m of memories) {
      const chunk = `${m.role}: ${m.content}\n`;
      const chunkTokens = chunk.length / 4; // rough estimate
      if (tokens + chunkTokens > maxTokens) break;
      context += chunk;
      tokens += chunkTokens;
    }

    return context.trim();
  }

  static async clearOldMemories(maxAgeDays = 90) {
    if (!db) return;

    const cutoff = Date.now() - maxAgeDays * 24 * 60 * 60 * 1000;
    const tx = db.transaction(STORE_NAME, 'readwrite');
    const store = tx.objectStore(STORE_NAME);
    const req = store.openCursor();

    req.onsuccess = (e) => {
      const cursor = (e.target as IDBRequest<IDBCursorWithValue>).result;
      if (cursor) {
        if (cursor.value.timestamp < cutoff) {
          cursor.delete();
        }
        cursor.continue();
      }
    };
  }
}

export default RAGMemory;
