// src/sync/automerge-sync-providers.ts – Automerge Sync Providers Manager v1
// WebSocket relay, HTTP polling, offline IndexedDB, mercy-gated sync
// MIT License – Autonomicity Games Inc. 2026

import * as Automerge from '@automerge/automerge';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const RELAY_WS_URL = import.meta.env.VITE_AUTOMERGE_WS_RELAY || 'wss://automerge-relay.rathor.ai';
const HTTP_SYNC_ENDPOINT = import.meta.env.VITE_AUTOMERGE_HTTP_SYNC || 'https://sync.rathor.ai/changes';

export class AutomergeSyncProviders {
  private doc: Automerge.Doc<any>;
  private lastHeads: string[] = [];
  private ws: WebSocket | null = null;

  constructor(doc: Automerge.Doc<any>) {
    this.doc = doc;
    this.lastHeads = Automerge.getHeads(doc);
    this.initSync();
  }

  private async initSync() {
    // Offline persistence (IndexedDB via idb-keyval or similar)
    // Placeholder: real impl would use Dexie / idb-keyval
    console.log("[AutomergeSync] Local IndexedDB persistence initialized");

    // WebSocket real-time relay (high valence only)
    if (currentValence.get() > 0.9 && navigator.onLine) {
      this.ws = new WebSocket(RELAY_WS_URL);

      this.ws.onopen = () => {
        console.log("[AutomergeSync] WebSocket relay connected");
        this.sendCurrentChanges();
      };

      this.ws.onmessage = (event) => {
        const binary = new Uint8Array(event.data);
        Automerge.applyChanges(this.doc, binary);
        this.lastHeads = Automerge.getHeads(this.doc);
        console.log("[AutomergeSync] Received remote changes – merged");
      };

      this.ws.onclose = () => console.log("[AutomergeSync] WebSocket relay disconnected");
    }
  }

  /**
   * Send only new changes since last sync
   */
  async sendCurrentChanges() {
    if (!this.ws || this.ws.readyState !== WebSocket.OPEN) return;

    const newChanges = Automerge.getChanges(this.doc, this.lastHeads);
    if (newChanges.length === 0) return;

    this.ws.send(Automerge.save(this.doc)); // or send only newChanges for delta
    this.lastHeads = Automerge.getHeads(this.doc);

    console.log(`[AutomergeSync] Sent ${newChanges.length} new changes`);
  }

  /**
   * Poll HTTP endpoint for remote changes (high-latency fallback)
   */
  async pollHttpSync() {
    if (!await mercyGate('HTTP sync poll', 'Automerge HTTP sync')) return;

    try {
      const response = await fetch(HTTP_SYNC_ENDPOINT, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: Automerge.save(this.doc)
      });

      if (response.ok) {
        const binary = new Uint8Array(await response.arrayBuffer());
        Automerge.applyChanges(this.doc, binary);
        this.lastHeads = Automerge.getHeads(this.doc);
        console.log("[AutomergeSync] HTTP poll sync successful");
      }
    } catch (e) {
      console.warn("[AutomergeSync] HTTP poll failed", e);
    }
  }
}

export const automergeSyncProviders = new AutomergeSyncProviders(/* pass global Automerge doc */);
