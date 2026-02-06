// src/sync/subdocuments-manager.ts – sovereign Subdocuments Manager v1
// Creation, lazy loading, per-node naming, GC safety, mercy-gated
// MIT License – Autonomicity Games Inc. 2026

import * as Y from 'yjs';
import { IndexeddbPersistence } from 'y-indexeddb';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';

const MERCY_THRESHOLD = 0.9999999;
const SUBDOC_PREFIX = 'mercy-subdoc-';

export class SubdocumentsManager {
  private parentDoc: Y.Doc;
  private subdocs = new Map<string, Y.Doc>();
  private persistences = new Map<string, IndexeddbPersistence>();

  constructor(parentDoc: Y.Doc) {
    this.parentDoc = parentDoc;
  }

  /**
   * Get or create subdocument for a key (lazy + mercy-gated)
   */
  async getOrCreateSubdoc(key: string, requiredValence: number = MERCY_THRESHOLD): Promise<Y.Doc | null> {
    const actionName = `Create/Get subdoc: ${key}`;
    if (!await mercyGate(actionName, key, requiredValence)) {
      return null;
    }

    if (this.subdocs.has(key)) {
      return this.subdocs.get(key)!;
    }

    const subdoc = new Y.Doc({
      gc: true,
      gcFilter: () => true,
    });

    // Attach to parent
    const parentMap = this.parentDoc.getMap('subdocs');
    parentMap.set(key, subdoc);

    // Per-subdoc persistence (unique name per key)
    const persistenceName = `\( {SUBDOC_PREFIX} \){key}`;
    const persistence = new IndexeddbPersistence(persistenceName, subdoc);
    this.persistences.set(key, persistence);

    // Optional relay connection (high valence only)
    if (currentValence.get() > 0.95 && navigator.onLine) {
      // Example: y-websocket per subdoc
      // new WebsocketProvider(RELAY_URL, persistenceName, subdoc);
    }

    this.subdocs.set(key, subdoc);

    console.log(`[SubdocManager] Subdocument created/loaded: ${key} (persistence: ${persistenceName})`);
    return subdoc;
  }

  /**
   * Clean up subdoc (detach + destroy persistence)
   */
  async destroySubdoc(key: string) {
    if (!this.subdocs.has(key)) return;

    const subdoc = this.subdocs.get(key)!;
    const persistence = this.persistences.get(key);

    // Detach from parent
    const parentMap = this.parentDoc.getMap('subdocs');
    parentMap.delete(key);

    // Destroy persistence
    if (persistence) {
      await persistence.destroy();
      this.persistences.delete(key);
    }

    this.subdocs.delete(key);
    console.log(`[SubdocManager] Subdocument destroyed: ${key}`);
  }

  /**
   * Get all active subdocs (for monitoring)
   */
  getActiveSubdocs(): Map<string, Y.Doc> {
    return new Map(this.subdocs);
  }
}

export const subdocumentsManager = new SubdocumentsManager(/* pass global ydoc when initializing */);
