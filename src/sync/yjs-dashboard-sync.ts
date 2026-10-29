// src/sync/yjs-dashboard-sync.ts
import * as Y from 'yjs';
import { WebsocketProvider } from 'y-websocket';
import { IndexeddbPersistence } from 'y-indexeddb';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const RELAY_URL = import.meta.env.VITE_YJS_RELAY_URL || 'wss://relay.rathor.ai';

export class YjsDashboardSync {
  private ydoc: Y.Doc;
  private provider: WebsocketProvider | null = null;
  private persistence: IndexeddbPersistence;

  constructor() {
    this.ydoc = new Y.Doc();
    this.persistence = new IndexeddbPersistence('mercy-dashboard-global', this.ydoc);

    // Only connect relay when online & high valence
    if (navigator.onLine && currentValence.get() > 0.9) {
      this.provider = new WebsocketProvider(RELAY_URL, 'mercy-dashboard-global', this.ydoc);
    }

    // Awareness: show other users' cursors / presence
    this.ydoc.awareness.setLocalStateField('user', {
      name: 'MercyUser',
      color: '#00ff88',
      valence: currentValence.get()
    });
  }

  async syncValence() {
    if (!await mercyGate('Sync dashboard valence')) return;

    const globalState = this.ydoc.getMap('global');
    globalState.set('valence', currentValence.get());

    console.log(`[YjsDashboardSync] Valence synced: ${currentValence.get().toFixed(8)}`);
  }

  getYDoc() {
    return this.ydoc;
  }
}

export const yjsDashboardSync = new YjsDashboardSync();

// Usage in dashboard component
// yjsDashboardSync.syncValence();
