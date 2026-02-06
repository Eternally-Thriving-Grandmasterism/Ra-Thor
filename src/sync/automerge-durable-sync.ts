// src/sync/automerge-durable-sync.ts – Automerge Durable Document Sync Layer v1
// Per-document independent histories, compact binary sync, valence-aware change filtering
// MIT License – Autonomicity Games Inc. 2026

import * as Automerge from '@automerge/automerge';
import { currentValence } from '@/core/valence-tracker';
import { mercyGate } from '@/core/mercy-gate';
import mercyHaptic from '@/utils/haptic-utils';

const MERCY_THRESHOLD = 0.9999999;
const VALENCE_CHANGE_FILTER_PIVOT = 0.9;
const AUTMERGE_STORAGE_PREFIX = 'rathor-automerge-';

interface DocumentState<T> {
  doc: Automerge.AutomergeDocument<T>;
  lastSync: number;
  pendingChanges: Uint8Array[];
}

const documents = new Map<string, DocumentState<any>>();

export class AutomergeDurableSync {
  static async initialize<T>(docId: string, initialState: T) {
    const actionName = `Initialize Automerge document ${docId}`;
    if (!await mercyGate(actionName)) return;

    if (documents.has(docId)) {
      console.log(`[AutomergeSync] Document ${docId} already initialized`);
      return;
    }

    try {
      // Load from IndexedDB if exists
      const stored = await this.loadFromStorage(docId);
      let doc: Automerge.AutomergeDocument<T>;

      if (stored) {
        doc = Automerge.load<T>(stored);
        console.log(`[AutomergeSync] Loaded existing document ${docId}`);
      } else {
        doc = Automerge.from<T>(initialState);
        console.log(`[AutomergeSync] Created new document ${docId}`);
      }

      documents.set(docId, {
        doc,
        lastSync: Date.now(),
        pendingChanges: []
      });

      mercyHaptic.playPattern('cosmicHarmony', currentValence.get());
    } catch (e) {
      console.error(`[AutomergeSync] Initialization failed for ${docId}`, e);
    }
  }

  static async change<T>(docId: string, callback: (doc: T) => void) {
    const state = documents.get(docId);
    if (!state) {
      throw new Error(`Document ${docId} not initialized`);
    }

    const valence = currentValence.get();

    const [newDoc, changes] = Automerge.change(state.doc, callback);

    state.doc = newDoc;
    state.lastSync = Date.now();

    // Valence-aware change filtering
    if (valence > VALENCE_CHANGE_FILTER_PIVOT) {
      // High valence → sync immediately
      await this.sync(docId, changes);
    } else {
      // Queue for batch sync
      state.pendingChanges.push(changes);
      await this.persistToStorage(docId);
    }
  }

  private static async sync(docId: string, changes?: Uint8Array) {
    const state = documents.get(docId);
    if (!state) return;

    const payload = changes || Automerge.getChanges(state.doc, []);

    if (payload.length === 0) return;

    // TODO: send to relay / other peers (WebSocket, WebRTC, or ElectricSQL bridge)
    console.log(`[AutomergeSync] Syncing ${payload.length} bytes for ${docId}`);

    // Clear pending if full sync
    state.pendingChanges = [];
    await this.persistToStorage(docId);
  }

  private static async persistToStorage(docId: string) {
    const state = documents.get(docId);
    if (!state) return;

    const binary = Automerge.save(state.doc);
    const key = `\( {AUTMERGE_STORAGE_PREFIX} \){docId}`;

    try {
      await new Promise<void>((resolve, reject) => {
        const tx = indexedDB.open('rathor-mercy-db');
        tx.onsuccess = (e) => {
          const db = (e.target as IDBOpenDBRequest).result;
          const store = db.transaction('automerge-docs', 'readwrite').objectStore('automerge-docs');
          store.put({ id: docId, binary });
          resolve();
        };
        tx.onerror = () => reject(tx.error);
      });
    } catch (e) {
      console.error(`[AutomergeSync] Storage failed for ${docId}`, e);
    }
  }

  private static async loadFromStorage(docId: string): Promise<Uint8Array | null> {
    const key = `\( {AUTMERGE_STORAGE_PREFIX} \){docId}`;
    try {
      const binary = await new Promise<Uint8Array | null>((resolve) => {
        const tx = indexedDB.open('rathor-mercy-db');
        tx.onsuccess = (e) => {
          const db = (e.target as IDBOpenDBRequest).result;
          const store = db.transaction('automerge-docs').objectStore('automerge-docs');
          const req = store.get(docId);
          req.onsuccess = () => resolve(req.result?.binary || null);
          req.onerror = () => resolve(null);
        };
      });
      return binary;
    } catch (e) {
      console.error(`[AutomergeSync] Load from storage failed for ${docId}`, e);
      return null;
    }
  }

  static getDocument<T>(docId: string): Automerge.AutomergeDocument<T> | null {
    return documents.get(docId)?.doc as Automerge.AutomergeDocument<T> | null;
  }

  static async destroy(docId?: string) {
    if (docId) {
      documents.delete(docId);
      const key = `\( {AUTMERGE_STORAGE_PREFIX} \){docId}`;
      // Delete from IndexedDB (optional)
    } else {
      documents.clear();
    }
  }
}

export default AutomergeDurableSync;
