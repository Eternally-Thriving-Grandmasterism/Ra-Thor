// src/sync/yjs-subdoc-sync-providers.ts – Yjs Subdocument Sync Providers Manager v1
// Per-subdoc persistence & network providers, lazy connect, high-latency handling, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';
import { IndexeddbPersistence } from 'y-indexeddb';
import { WebsocketProvider } from 'y-websocket';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const SUBDOC_PREFIX = 'mercy-yjs-subdoc-';
const RELAY_URL = import.meta.env.VITE_SYNC_RELAY_URL || 'wss://relay.rathor.ai';

export class YjsSubdocSyncProviders {
  private parentDoc: Y.Doc;
  private subdocPersistences = new Map<string, IndexeddbPersistence>();
  private subdocProviders = new Map<string, WebsocketProvider>();

  constructor(parentDoc: Y.Doc) {
    this.parentDoc = parentDoc;
  }

  /**
   * Get or create subdoc with its own persistence & network provider
   */
  async getOrCreateWithProviders(
    key: string,
    initialValue: any = {},
    requiredValence: number = MERCY_THRESHOLD
  ): Promise<Y.Doc | null> {
    const actionName = `Setup subdoc sync providers: ${key}`;
    if (!await mercyGate(actionName, key, requiredValence)) {
      return null;
    }

    // Persistence (IndexedDB per subdoc)
    const persistenceName = `\( {SUBDOC_PREFIX} \){key}`;
    let persistence = this.subdocPersistences.get(key);
    if (!persistence) {
      persistence = new IndexeddbPersistence(persistenceName, this.parentDoc);
      this.subdocPersistences.set(key, persistence);
    }

    // Network provider (websocket per subdoc room)
    let provider = this.subdocProviders.get(key);
    if (!provider && navigator.onLine && currentValence.get() > 0.9) {
      provider = new WebsocketProvider(RELAY_URL, persistenceName, this.parentDoc, {
        connectTimeout: 3000,
        maxBackoffTime: 15000
      });
      this.subdocProviders.set(key, provider);
    }

    // Subdoc itself
    const parentMap = this.parentDoc.getMap('subdocs');
    let subdoc = parentMap.get(key) as Y.Doc | undefined;

    if (!subdoc) {
      subdoc = new Y.Doc();
      parentMap.set(key, subdoc);
    }

    console.log(`[YjsSubdocSync] Providers ready for \( {key}: persistence= \){persistenceName}, network=${provider ? 'connected' : 'offline'}`);
    return subdoc;
  }

  /**
   * Clean up subdoc providers (disconnect & destroy persistence)
   */
  async cleanup(key: string) {
    const provider = this.subdocProviders.get(key);
    if (provider) {
      provider.destroy();
      this.subdocProviders.delete(key);
    }

    const persistence = this.subdocPersistences.get(key);
    if (persistence) {
      await persistence.destroy();
      this.subdocPersistences.delete(key);
    }

    console.log(`[YjsSubdocSync] Providers cleaned up for ${key}`);
  }
}

export const yjsSubdocSyncProviders = new YjsSubdocSyncProviders(/* pass global ydoc */);
